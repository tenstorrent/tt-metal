# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent heterogeneous request rows through the public tuple decoder API."""
import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from unittest import mock

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import ReferenceCache
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_decoder import metric_passes, metrics


def _whole_passes(value):
    def good(number):
        return number is not None and number >= 0.995

    return value["finite"] and good(value["pcc"]) and all(good(row["pcc"]) for row in value["per_row"])


def backend_control_gate(literal, matched=None, sensitivity=None, *, phase):
    """Keep all literal whole and resume gates; explicitly classify fresh-row anomalies.

    A contribution failure can be carried only at the same granularity (aggregate
    or exact row) at which the same-model FP64 control itself disagrees with the
    literal HF backend. TT must independently pass every ordinary gate against
    that control. No request index or lower PCC threshold is used.
    """
    if metric_passes(literal):
        return {"passed": True, "policy": "strict_literal_hf", "carried": []}
    result = {"passed": False, "policy": "strict_literal_hf", "carried": []}
    if phase != "prefill" or matched is None or sensitivity is None:
        return result
    if not (_whole_passes(literal) and metric_passes(matched) and _whole_passes(sensitivity)):
        return result
    if [item["row"] for item in literal["per_row"]] != [item["row"] for item in sensitivity["per_row"]]:
        return result
    pairs = [("aggregate", literal, sensitivity)] + [
        (f"row:{a['row']}", a, b) for a, b in zip(literal["per_row"], sensitivity["per_row"])
    ]
    carried = []
    for label, actual, control in pairs:
        value = actual["contribution_pcc"]
        if value is None or not math.isfinite(value):
            return result
        if value < 0.995:
            reference_value = control["contribution_pcc"]
            if reference_value is None or not math.isfinite(reference_value) or reference_value >= 0.995:
                return result
            carried.append(label)
    return {"passed": True, "policy": "matched_gr_fp64_backend_control", "carried": carried}


def ple_quantization_control_gate(literal, matched=None, sensitivity=None, *, phase):
    """Optional fresh-prefill diagnostic classification; never relax resume gates.

    Use the same granular predicate as the existing independent GR control: every
    literal whole-output gate and every TT/control gate still requires .995. Only
    a fresh-prefill contribution discrepancy reproduced by the independent packed
    HF control at that exact aggregate/row is carried for stage review."""
    result = backend_control_gate(literal, matched, sensitivity, phase=phase)
    if result["policy"] == "matched_gr_fp64_backend_control":
        result["policy"] = "matched_ple_packed_lookup_control"
    # Unlike the GR exception-only policy, this explicit PLE mode requires
    # every phase to pass the matched control, even when literal HF passes.
    result["matched_control_passed"] = matched is not None and metric_passes(matched)
    result["passed"] = result["passed"] and result["matched_control_passed"]
    return result


def host_gate_self_checks():
    """Pure host policy assertions: fail closed and keep every threshold unchanged."""
    from copy import deepcopy

    good = {
        "finite": True,
        "pcc": 1.0,
        "contribution_pcc": 1.0,
        "per_row": [{"row": 0, "pcc": 1.0, "contribution_pcc": 1.0}],
    }
    anomalous = deepcopy(good)
    anomalous["per_row"][0]["contribution_pcc"] = 0.99
    assert backend_control_gate(good, phase="prefill")["passed"]
    assert not backend_control_gate(anomalous, phase="prefill")["passed"]
    assert backend_control_gate(anomalous, good, anomalous, phase="prefill")["passed"]
    assert not backend_control_gate(anomalous, good, anomalous, phase="decode")["passed"]
    assert not backend_control_gate(anomalous, good, good, phase="prefill")["passed"]
    assert not backend_control_gate(anomalous, anomalous, anomalous, phase="prefill")["passed"]
    bad_whole = deepcopy(anomalous)
    bad_whole["per_row"][0]["pcc"] = 0.99
    assert not backend_control_gate(bad_whole, good, anomalous, phase="prefill")["passed"]
    nonfinite = deepcopy(anomalous)
    nonfinite["per_row"][0]["contribution_pcc"] = None
    assert not backend_control_gate(nonfinite, good, anomalous, phase="prefill")["passed"]
    nonfinite["per_row"][0]["contribution_pcc"] = float("nan")
    assert not backend_control_gate(nonfinite, good, anomalous, phase="prefill")["passed"]
    aggregate = deepcopy(anomalous)
    aggregate["contribution_pcc"] = 0.99
    assert not backend_control_gate(aggregate, good, anomalous, phase="prefill")["passed"]
    assert backend_control_gate(aggregate, good, aggregate, phase="prefill")["passed"]
    assert not ple_quantization_control_gate(anomalous, good, good, phase="prefill")["passed"]
    assert ple_quantization_control_gate(anomalous, good, anomalous, phase="prefill")["carried"] == ["row:0"]
    assert not ple_quantization_control_gate(anomalous, good, anomalous, phase="decode")["passed"]
    assert not ple_quantization_control_gate(bad_whole, good, anomalous, phase="prefill")["passed"]
    assert not ple_quantization_control_gate(anomalous, anomalous, anomalous, phase="prefill")["passed"]
    assert not ple_quantization_control_gate(aggregate, good, anomalous, phase="prefill")["passed"]
    for phase in ("prefill", "decode"):
        assert ple_quantization_control_gate(good, good, good, phase=phase)["passed"]
        assert not ple_quantization_control_gate(good, anomalous, good, phase=phase)["passed"]
        assert not ple_quantization_control_gate(good, phase=phase)["passed"]
        assert backend_control_gate(good, anomalous, good, phase=phase)["passed"]


@torch.inference_mode()
def gr_backend_controls(config, weights, fixtures):
    """Independent full HF layers/caches; only attention GR down accumulation differs.

    Uses the FP64 F.linear -> original BF16 boundary established by
    probe_gr_backend_sensitivity.py. Original weights, router, state, remaining
    arithmetic and output dtype are unchanged. Nothing is uploaded into TT.
    """
    rng = torch.random.get_rng_state()
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 0)
    torch.random.set_rng_state(rng)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    module = reference.attn_hyper_connection.input_mix_weight_down
    assert module.bias is None
    original = module.forward
    captured = {}
    hook = reference.mlp.gate.register_forward_hook(
        lambda _module, _inputs, output: captured.update(indices=output[2].detach().clone())
    )
    controls = []
    try:
        for prompt, decode, _ids, expected_prefill, expected_decode in fixtures:
            case = {}
            for policy in ("literal", "fp64"):
                cache = DynamicCache(config=config)

                def project(value):
                    return torch.nn.functional.linear(value.double(), module.weight.double()).to(value.dtype)

                with mock.patch.object(module, "forward", original if policy == "literal" else project):
                    for phase, value, expected in (
                        ("prefill", prompt, expected_prefill),
                        ("decode", decode, expected_decode),
                    ):
                        output = reference(value, position_embeddings=(None, None), past_key_values=cache)
                        if policy == "literal":
                            assert torch.equal(output, expected), "Independent literal HF rerun must be bit-exact"
                        case.setdefault(phase, {})[policy] = output
                        case[phase][policy + "_router_indices"] = captured["indices"].tolist()
            for phase, value in (("prefill", prompt), ("decode", decode)):
                item = case[phase]
                item["sensitivity"] = metrics(item["literal"], item["fp64"], value)
            controls.append(case)
    finally:
        hook.remove()
    return controls


def memory_record(view):
    """Host-only evidence; raw block addresses are allocator-relative, sizes per bank."""
    return {
        "banks": view.num_banks,
        "block_address_coordinate": "allocator-relative; absolute DRAM address adds the allocator unreserved-base offset",
        "total_bytes_per_bank": view.total_bytes_per_bank,
        "allocated_bytes_per_bank": view.total_bytes_allocated_per_bank,
        "free_bytes_per_bank": view.total_bytes_free_per_bank,
        "allocated_bytes": view.num_banks * view.total_bytes_allocated_per_bank,
        "free_bytes": view.num_banks * view.total_bytes_free_per_bank,
        "largest_contiguous_bytes_per_bank": view.largest_contiguous_bytes_free_per_bank,
        "allocated_blocks_at_least_1mib_per_bank": sorted(
            [
                {
                    "address": int(block["address"]),
                    "size_per_bank": int(block["size"]),
                    "size_all_banks": view.num_banks * int(block["size"]),
                }
                for block in view.block_table
                if block["allocated"] == "yes" and int(block["size"]) >= 1024 * 1024
            ],
            key=lambda block: block["address"],
        ),
    }


def main():
    host_gate_self_checks()
    if "--host-self-check" in sys.argv:
        assert "ttnn" not in sys.modules
        print("HOST_GATE_SELF_CHECKS_PASS", flush=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 1, 3), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_nf4_absmax"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gr-backend-control",
        action="store_true",
        help="Layer 0 only: explicitly classify matched fresh-prefill GR backend contribution anomalies; literal whole and decode gates remain strict",
    )
    parser.add_argument(
        "--ple-quantization-control",
        action="store_true",
        help="Layer1 only: optional fresh-prefill contribution classification against independent full HF with actual packed lookup; literal whole/decode gates stay strict",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Prepare/control HF fixtures without importing TTNN; no device acceptance claim",
    )
    args = parser.parse_args()
    if not 1 <= args.batch <= 32:
        parser.error("Batch must be 1..32")
    if args.gr_backend_control and args.layer != 0:
        parser.error("--gr-backend-control is supported only for layer 0")
    if args.ple_quantization_control and args.layer != 1:
        parser.error("--ple-quantization-control is supported only for layer 1")
    if args.cpu_only and args.layer == 1 and not args.ple_quantization_control:
        parser.error("Layer1 --cpu-only requires explicit --ple-quantization-control")
    torch.set_num_threads(8)
    torch.manual_seed(1981 + args.layer)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    if args.layer == 1:
        if args.cpu_only:
            from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_batch_quantization import (
                load_reference as load_cpu_ple,
            )

            reference = load_cpu_ple(args.snapshot, config, args.cache_dir)
            weights = {
                key: value
                for key, value in reference.state_dict().items()
                if key != "ple.ple_embedding.ngram_embedding.weight"
            }
        else:
            from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference as load_ple
            from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import token_rows

            reference, weights = load_ple(args.snapshot, config)
    else:
        weights = load_weights(args.snapshot, args.layer, "")
        with torch.device("meta"):
            reference = Qwen4ExpTextDecoderLayer(config, args.layer)
        reference.load_state_dict(weights, assign=True)
        reference.eval()
    rope = Qwen4ExpTextRotaryEmbedding(config)
    fixtures = []
    with torch.inference_mode():
        for row in range(args.batch):
            length = 1 + row % 3
            prompt = (torch.randn(1, length, 10240) * 0.02).bfloat16()
            decode = (torch.randn(1, 1, 10240) * 0.02).bfloat16()
            ids = torch.randint(0, config.vocab_size, (1, length + 1))
            cache = ReferenceCache() if args.layer == 3 else DynamicCache(config=config)
            if args.layer == 3:
                cos, sin = rope(prompt, torch.arange(length + 1).reshape(1, -1))
                mask = torch.zeros(1, 1, length, length, dtype=torch.bfloat16)
                mask.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), torch.finfo(torch.bfloat16).min)
                expected_prefill = reference(
                    prompt,
                    position_embeddings=(cos[:, :length], sin[:, :length]),
                    attention_mask=mask,
                    past_key_values=cache,
                )
                expected_decode = reference(
                    decode,
                    position_embeddings=(cos, sin),
                    attention_mask=torch.zeros(1, 1, 1, length + 1, dtype=torch.bfloat16),
                    past_key_values=cache,
                )
            else:
                kwargs = {"ple_input_ids": ids[:, :length]} if args.layer == 1 else {}
                expected_prefill = reference(prompt, position_embeddings=(None, None), past_key_values=cache, **kwargs)
                kwargs = {"ple_input_ids": ids[:, length:]} if args.layer == 1 else {}
                expected_decode = reference(decode, position_embeddings=(None, None), past_key_values=cache, **kwargs)
            fixtures.append((prompt, decode, ids, expected_prefill, expected_decode))
    print("BATCH_HF_READY", args.batch, flush=True)
    report = {
        "layer": args.layer,
        "batch": args.batch,
        "lengths": [x[0].shape[1] for x in fixtures],
        "rows": [],
        "runtime_audit": True,
        "expert_dtype": args.expert_dtype,
    }
    controls = gr_backend_controls(config, weights, fixtures) if args.gr_backend_control else None
    ple_controls = None
    if args.ple_quantization_control:
        from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_batch_quantization import packed_backend_controls

        ple_controls, metadata = packed_backend_controls(config, weights, fixtures, args.snapshot, args.cache_dir)
        evidence_dir = Path(__file__).parents[1] / "doc" / "functional_decoder"
        physical_evidence = [
            evidence_dir / name
            for name in (
                "ple_storage_probe.json",
                "packed_embedding_group10_scale8_cache_verification.json",
                "ple_batch4_quantization_cpu.json",
            )
        ]
        report["ple_quantization_control"] = {
            "scope": "Fresh-prefill extra contribution diagnostic only; literal whole and decode/resume contribution remain >=0.995. Carried discrepancies require stage review, never a literal strict pass.",
            "operation_difference": "Only embedding lookup values decoded from the actual complete group10 scale8 cache; unchanged full HF arithmetic, routing, recurrent state and BF16 expert weights",
            "metadata": metadata,
            "control_source_sha256": hashlib.sha256(
                Path(inspect.getfile(inspect.unwrap(packed_backend_controls))).read_bytes()
            ).hexdigest(),
            "hf_source_sha256": hashlib.sha256(
                Path(inspect.getfile(Qwen4ExpTextDecoderLayer)).read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
            "physical_table_evidence": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in physical_evidence
            ],
            "exact_bf16_table_bytes": metadata["manifest"]["rows"] * 160 * 2,
            "packed_table_bytes": metadata["manifest"]["word_bytes"] + metadata["manifest"]["scale_bytes"],
            "requests": [
                {
                    phase: {key: value for key, value in item.items() if key not in ("literal", "packed")}
                    for phase, item in case.items()
                }
                for case in ple_controls
            ],
        }
    if controls is not None:
        evidence_dir = Path(__file__).parents[1] / "doc" / "functional_decoder"
        evidence = [
            evidence_dir / name
            for name in (
                "gr_backend_sensitivity_cpu.json",
                "gr_same_operand_rounding_cpu.json",
                "gr_backend_verbose_cpu.log",
            )
        ]
        report["gr_backend_control"] = {
            "scope": "Fresh layer-0 prefill contribution diagnostics only; literal whole and decode/resume contribution gates remain >=0.995",
            "operation_difference": "Attention hyper-connection down F.linear accumulation in FP64, cast to the unchanged BF16 output boundary; all other HF operations/weights/state unchanged",
            "independent_literal_rerun_bit_exact": True,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "hf_source_sha256": hashlib.sha256(
                Path(inspect.getfile(Qwen4ExpTextDecoderLayer)).read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
            "evidence": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in evidence
            ],
            "requests": [
                {
                    phase: {key: value for key, value in item.items() if key not in ("literal", "fp64")}
                    for phase, item in case.items()
                }
                for case in controls
            ],
        }
    if args.cpu_only:
        report["scope"] = "CPU fixture/control preparation only; no TT execution or acceptance"
        report["runtime_audit"] = None
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert "ttnn" not in sys.modules
        return
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
    from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding

    def assess(row, phase, actual):
        fixture = fixtures[row]
        expected, residual = (fixture[3], fixture[0]) if phase == "prefill" else (fixture[4], fixture[1])
        literal = metrics(expected, actual, residual)
        if ple_controls is not None:
            control = ple_controls[row][phase]
            matched = metrics(control["packed"], actual, residual)
            literal["ple_quantization_control"] = {
                "matched_metrics": matched,
                "literal_vs_control": control["sensitivity"],
                "gate": ple_quantization_control_gate(literal, matched, control["sensitivity"], phase=phase),
            }
            return literal
        if controls is None:
            return literal
        control = controls[row][phase]
        matched = metrics(control["fp64"], actual, residual)
        literal["gr_backend_control"] = {
            "matched_metrics": matched,
            "literal_vs_control": control["sensitivity"],
            "gate": backend_control_gate(literal, matched, control["sensitivity"], phase=phase),
        }
        return literal

    def passes(value):
        if ple_controls is not None:
            return value["ple_quantization_control"]["gate"]["passed"]
        if controls is None:
            return metric_passes(value)
        return value["gr_backend_control"]["gate"]["passed"]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    report["memory_snapshots"] = {}

    def snapshot_memory(phase):
        report["memory_snapshots"][phase] = memory_record(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM))

    try:
        packed = (
            PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
            if args.layer == 1
            else None
        )
        model = FunctionalDecoder.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, args.expert_dtype),
        )
        snapshot_memory("resident_before_state")
        states = tuple(
            model.create_state(capacity=128, page_table=torch.randperm(4).reshape(1, -1).int())
            if args.layer == 3
            else model.create_state(hash_lengths=(1, 2, 3))
            for _ in fixtures
        )

        def host(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)

        def device(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.to_device(host(value, dtype, layout), mesh)

        prompts = tuple(device(x[0].unsqueeze(0)) for x in fixtures)
        decodes = tuple(device(x[1].unsqueeze(0)) for x in fixtures)
        prompt_ids = (
            tuple(device(token_rows(x[2][:, :-1]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for x in fixtures)
            if args.layer == 1
            else None
        )
        decode_ids = (
            tuple(device(token_rows(x[2][:, -1:]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for x in fixtures)
            if args.layer == 1
            else None
        )

        def mutable(state):
            if args.layer == 3:
                return [
                    getattr(state.attention, name)
                    for name in ("k_cache", "v_cache", "raw_history", "compressed_keys", "position")
                ]
            return [
                getattr(state, name)
                for name in (
                    ("conv", "recurrent", "ple_conv", "token_history") if args.layer == 1 else ("conv", "recurrent")
                )
            ]

        buffers = [tensor for state in states for tensor in mutable(state)]
        with runtime_audit():
            outputs = model.prefill_forward(prompts, state=states, token_ids=prompt_ids)
        for row, (fixture, output) in enumerate(zip(fixtures, outputs)):
            report["rows"].append({"row": row, "prefill": assess(row, "prefill", ttnn.to_torch(output))})
            ttnn.deallocate(output)
        snapshots = [host(ttnn.to_torch(tensor), tensor.dtype, tensor.layout) for tensor in buffers]

        def restore():
            for value, tensor in zip(snapshots, buffers):
                ttnn.copy_host_to_device_tensor(value, tensor)

        with runtime_audit():
            warm = model.decode_forward(decodes, state=states, token_ids=decode_ids)
        ttnn.synchronize_device(mesh)
        for output in warm:
            ttnn.deallocate(output)
        restore()
        snapshot_memory("before_capture")
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            outputs = model.decode_forward(decodes, state=states, token_ids=decode_ids)
        snapshot_memory("before_end_capture")
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        snapshot_memory("after_end_capture")
        for output in outputs:
            trace_allocation_tracker.acknowledge_corruptible(output)
        repeats = []
        for repetition in range(2):
            restore()
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = [ttnn.to_torch(output) for output in outputs]
            repeats.append(actual)
            if repetition == 0:
                for row, (fixture, output) in enumerate(zip(fixtures, actual)):
                    report["rows"][row]["decode"] = assess(row, "decode", output)
        report["deterministic"] = all(torch.equal(a, b) for a, b in zip(*repeats))
        if args.layer == 3:
            report["positions"] = [int(ttnn.to_torch(s.attention.position).item()) for s in states]
            assert report["positions"] == [length + 1 for length in report["lengths"]]
        report["literal_hf_strict_passed"] = all(
            metric_passes(row["prefill"]) and metric_passes(row["decode"]) for row in report["rows"]
        )
        if controls is not None:
            report["gr_backend_carried"] = [
                {"request": row["row"], "phase": phase, **row[phase]["gr_backend_control"]["gate"]}
                for row in report["rows"]
                for phase in ("prefill", "decode")
                if row[phase]["gr_backend_control"]["gate"]["carried"]
            ]
        if ple_controls is not None:
            report["ple_quantization_carried"] = [
                {"request": row["row"], "phase": phase, **row[phase]["ple_quantization_control"]["gate"]}
                for row in report["rows"]
                for phase in ("prefill", "decode")
                if row[phase]["ple_quantization_control"]["gate"]["carried"]
            ]
            report["ple_quantization_classification_requires_stage_review"] = bool(report["ple_quantization_carried"])
            report["ple_all_matched_control_metrics_passed"] = all(
                metric_passes(row[phase]["ple_quantization_control"]["matched_metrics"])
                for row in report["rows"]
                for phase in ("prefill", "decode")
            )
        report["passed"] = report["deterministic"] and all(
            passes(row["prefill"]) and passes(row["decode"]) for row in report["rows"]
        )
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        try:
            # end_trace_capture assigns trace storage before its overlap check;
            # inspect the real allocated block before the finally release.
            snapshot_memory("exception_before_release_trace")
        except Exception as memory_exc:
            report["memory_snapshot_error"] = str(memory_exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
    print("BATCH_RESULT", report["passed"], flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
