# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reproducible TP4/EP4 capacity arithmetic; estimates are not device evidence."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capacity():
    source_sha256 = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()

    def current_report(path):
        report = json.loads(path.read_text())
        provenance_path = Path(str(path) + ".provenance.json")
        provenance = json.loads(provenance_path.read_text())
        assert provenance["source_sha256"]["tt/multichip_decoder.py"] == source_sha256, (
            "stale capacity evidence",
            str(path),
        )
        assert report["passed"] is True
        return report

    config = json.loads((ROOT / "doc/functional_decoder/hf/config.json").read_text())["text_config"]
    h, e, i, k = (config[n] for n in ("hidden_size", "num_experts", "moe_intermediate_size", "num_experts_per_tok"))
    layers = config["num_hidden_layers"]
    full = config["layer_types"].count("full_attention")
    linear = layers - full
    context = config["max_position_embeddings"]
    # BFP8 tile:1024 mantissa bytes +64 shared exponent bytes.
    expert_format = layers * e * 3 * h * i * 1088 // (4 * 1024)
    kv = context * 256 * 2 * 2 * full
    recurrent = linear * 8 * 128 * 128 * 4
    convolution = linear * 3 * 2048 * 2
    # Conservative allowance covers native and ordered projection copies,
    # norms/router/LUTs, embedding and output-head shards, plus alignment.
    nonexpert_reserve = 3 * 1024**3
    # A replicated262K residual is1GiB. Bound simultaneous input, chunk-list,
    # concatenated attention, MoE outputs/final concat plus CCL/trace scratch.
    trace_activation_reserve = 6 * 1024**3
    total = 2 * expert_format + kv + 32 * (recurrent + convolution) + nonexpert_reserve + trace_activation_reserve
    census_path = ROOT / "doc/multichip_decoder/mixed_stack_selected.json"
    measured = None
    physical = 32 * 1024**3
    if census_path.exists():
        report = current_report(census_path)
        census = report["multichip"]["allocator_census"]
        physical = census[0]["capacity_bytes_per_device"]
        measured = report["multichip"]["loaded_decoder_bytes_per_device"]
        measured = dict(
            artifact=str(census_path.relative_to(ROOT)),
            linear_layer_bytes=measured["layer_0_loaded"],
            full_layer_bytes=measured["layer_3_loaded"],
            extrapolated_stack_weight_bytes=linear * measured["layer_0_loaded"] + full * measured["layer_3_loaded"],
            note="Constructor-resident allocations measured;40-layer extrapolation, not a loaded full-model claim.",
        )
        assert measured["extrapolated_stack_weight_bytes"] < 2 * expert_format + nonexpert_reserve
    assert total < physical
    context_evidence = []
    for name in ("context_l0_32k.json", "context_l3_32k.json", "context_fullmax.json"):
        evidence = ROOT / "doc/multichip_decoder" / name
        if evidence.exists():
            result = current_report(evidence)
            assert result["ownership_passed"] is True
            layer = 0 if name == "context_l0_32k.json" else 3
            expected_length = 262142 if name == "context_fullmax.json" else 32769
            expected_context = 262144 if name == "context_fullmax.json" else 65536
            signature = result["signature"]
            assert (signature["layer"], signature["length"], signature["context"], signature["decode_steps"]) == (
                layer,
                expected_length,
                expected_context,
                2,
            )
            values = [result["prefill_pcc"], *result["decode_pcc"], *result["state_pcc"].values()]
            assert all(
                isinstance(value, (int, float)) and math.isfinite(value) and 0.995 <= value <= 1.000000000001
                for value in values
            )
            assert result["multichip"]["decode_positions"] == [expected_length, expected_length + 1]
            context_evidence.append(
                dict(
                    artifact=str(evidence.relative_to(ROOT)),
                    signature=result["signature"],
                    prefill_pcc=result["prefill_pcc"],
                    decode_pcc=result["decode_pcc"],
                )
            )
    return dict(
        model="Qwen/Qwen3.6-35B-A3B",
        decoder_source_sha256=source_sha256,
        mesh=[1, 4],
        device_ids=[0, 1, 2, 3],
        status=(
            "conservative full-stack arithmetic; per-kind resident census and decoder context validation passed"
            if len(context_evidence) == 3
            else "conservative arithmetic with per-kind resident census; context device validation pending"
        ),
        context_validation=context_evidence,
        layers=layers,
        full_attention_layers=full,
        linear_attention_layers=linear,
        expert_count=e,
        top_k=k,
        expert_intermediate=i,
        tp_intermediate_tiles=i // 4 // 32,
        batch=32,
        expected_expert_union_fraction=1 - (1 - k / e) ** 32,
        hf_advertised_context=context,
        supported_context=context,
        capability_reduction=None,
        expert_format_bytes_per_device=expert_format,
        hybrid_expert_weight_bytes_per_device=2 * expert_format,
        nonexpert_weight_and_lut_reserve_bytes_per_device=nonexpert_reserve,
        weight_bytes_per_device_estimate=2 * expert_format + nonexpert_reserve,
        kv_cache_bytes_per_device=kv,
        kv_cache_dtype="BF16",
        kv_head_ownership=[0, 0, 1, 1],
        recurrent_bytes_per_request_device=recurrent,
        convolution_bytes_per_request_device=convolution,
        trace_activation_reserve_bytes_per_device=trace_activation_reserve,
        full_model_planned_bytes_per_device=total,
        physical_dram_bytes_per_device_assumption=32 * 1024**3,
        allocator_capacity_bytes_per_device=physical,
        measured_resident_weight_census=measured,
        planned_headroom_bytes_per_device=physical - total,
        cache_pool_contract="262144 aggregate cached tokens; up to 32 request recurrent slots. One request may use the entire context. This is not 32 simultaneous 262K requests.",
        note="Nonexpert reserve bounds measured layer weights plus future embedding/head shards. The 6 GiB trace/activation reserve bounds the replicated maximum-context layer working set; it is not a measured peak.",
    )


if __name__ == "__main__":
    plan = capacity()
    path = ROOT / "doc/multichip_decoder/memory_capacity_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    contract_path = ROOT / "doc/context_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["multichip_decoder_plan"] = plan
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: plan[k]
                for k in (
                    "layers",
                    "full_attention_layers",
                    "supported_context",
                    "full_model_planned_bytes_per_device",
                    "planned_headroom_bytes_per_device",
                )
            }
        )
    )
