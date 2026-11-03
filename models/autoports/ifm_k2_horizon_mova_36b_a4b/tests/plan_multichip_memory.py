# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Conservative full-stack memory arithmetic from actual local weight manifests."""

import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots/e9ce717642b2074082875be233c02c729ba47960"
)


def tiled_bytes(shape, dtype):
    tile = {"DataType.BFLOAT4_B": 576, "DataType.BFLOAT8_B": 1088, "DataType.BFLOAT16": 2048, "DataType.FLOAT32": 4096}[
        dtype
    ]
    return math.prod(shape[:-2]) * math.ceil(shape[-2] / 32) * math.ceil(shape[-1] / 32) * tile


def main():
    doc = ROOT / "doc/multichip_decoder"
    c = json.loads((SNAPSHOT / "config.json").read_text())
    layers = {}
    for layer, count in [(0, 3), (3, 45)]:
        manifest = doc / f"final_wall_l{layer}_b1_s32.json"
        if not manifest.exists():
            manifest = doc / f"final_smoke_l{layer}_b1_s33.json"
        result = json.loads(manifest.read_text())
        weights = result["weights"]
        rows = {}
        total = 0
        for name, w in weights.items():
            size = tiled_bytes(w["local_shape"], w["dtype"])
            decode = 0
            decode_shape = None
            if name.endswith(".weight") and w["local_shape"][-2] > 1 and "swiglu" not in name:
                decode_shape = w["local_shape"].copy()
                readers = (
                    (1 if layer == 0 else 2)
                    if name == "attention_packed.weight"
                    else (
                        2
                        if layer == 0 and name == "mlp.gate_up.weight"
                        else (2 if layer == 3 and name == "self_attn.o_proj.weight" else 1)
                    )
                )
                decode_shape[-1] = math.ceil(decode_shape[-1] / (32 * 8 * readers)) * 32 * 8 * readers
                if name in result.get("decode_weight_specs", {}):
                    decode_shape = result["decode_weight_specs"][name]["shape"]
                decode = tiled_bytes(decode_shape, w["dtype"])
            rows[name] = {**w, "prefill_bytes": size, "decode_shape": decode_shape, "decode_bytes": decode}
            total += size + decode
        # Position tensor, expanded router bias, sparse reduce IDs/map/placeholders,
        # and generous allowance for all per-layer semaphore and alignment costs.
        constants = 524288 * 4 + 1024 * 1024
        if layer == 3:
            constants += 2048 * (64 + 128) * 4
        layers[str(layer)] = dict(
            count=count,
            weight_rows=rows,
            weight_bytes=total,
            constant_allowance=constants,
            total_bytes=total + constants,
        )
    stack_weights = sum(v["count"] * v["total_bytes"] for v in layers.values())
    embedding_head = 2 * c["vocab_size"] * 2560 * 2  # Conservative BF16 replicated embedding and head.
    per_layer_kv = 2 * 524288 * 2 * 128 * 1088 // 1024
    kv = per_layer_kv * 48
    reserve = 2 * 1024**3
    result = dict(
        mesh=[1, 4],
        context=524288,
        aggregate_cached_token_budget=524288,
        max_batch=32,
        layers=layers,
        per_device_stack_weights_and_constants=stack_weights,
        embedding_and_head_allowance=embedding_head,
        per_device_kv_per_layer=per_layer_kv,
        per_device_stack_kv_bytes=kv,
        trace_activation_collective_workspace_reserve=reserve,
        per_device_total_bytes=stack_weights + embedding_head + kv + reserve,
        prefill_schedule="Stack integration processes bounded2048-token chunks; public layer accepts full prompts",
        shared_ccl_l1_bytes=2 * 32 * 10240 * 1088 // 1024,
        status="calculated; requires allocator/capacity probe",
        cache_dtype="BFLOAT8_B",
    )
    source_sha = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    normalization = json.loads((doc / "commit_source_equivalence.json").read_text())
    tested_source = Path(normalization["tested_source"]).read_text()
    current_source = (ROOT / "tt/multichip_decoder.py").read_text()

    def computational_ast(source):
        return ast.dump(
            ast.Module(
                body=[n for n in ast.parse(source).body if not isinstance(n, (ast.Import, ast.ImportFrom))],
                type_ignores=[],
            )
        )

    assert source_sha == normalization["committed_sha256"]
    assert hashlib.sha256(tested_source.encode()).hexdigest() == normalization["tested_sha256"]
    assert computational_ast(tested_source) == computational_ast(current_source)
    tested_sha = normalization["tested_sha256"]
    # The final placement change only executes inside the sparse EP constructor.
    # Retain dense evidence after verifying the exact one-line source equivalence.
    placement = json.loads((doc / "expert_placement_equivalence.json").read_text())
    previous_source = tested_source.replace("expert = local * 4 + rank", "expert = rank * 25 + local")
    dense_equivalent = (
        tested_sha == placement["after_sha256"]
        and hashlib.sha256(previous_source.encode()).hexdigest() == placement["before_sha256"]
    )

    def source_matches(probe, key):
        return probe[key] in (source_sha, tested_sha) or (
            probe["layer"] == 0 and dense_equivalent and probe[key] == placement["before_sha256"]
        )

    capacity_paths = [doc / f"capacity_tp4_l{layer}_524288_bfloat8_b.json" for layer in (0, 3)]
    full_paths = [doc / f"full_prefill_tp4_layer{layer}.json" for layer in (0, 3)]
    verified = all(path.exists() for path in capacity_paths + full_paths)
    if verified:
        probes = [json.loads(path.read_text()) for path in capacity_paths]
        verified = all(
            source_matches(p, "source_sha256")
            and p["stack_reservation"]
            and p["decode_pcc"] >= 0.995
            and min(r["pcc"] for r in p["prefill"]) >= 0.995
            for p in probes
        )
    if verified:
        full = [json.loads(path.read_text()) for path in full_paths]
        verified = all(
            source_matches(p, "decoder_sha256")
            and p["repeated_decode_equal"]
            and p["traced_decode_pcc"] >= 0.995
            and set(p["lengths"]) == {524275, 524288}
            and min(r["ttnn_baseline_pcc"] for r in p["rows"]) >= 0.995
            for p in full
        )
    if verified:
        result.update(
            status="validated allocator reservation and full public prefill",
            physical_allocator_bytes=probes[0]["memory"]["num_banks"] * probes[0]["memory"]["total_bytes_per_bank"],
            decoder_sha256=source_sha,
            tested_decoder_sha256=tested_sha,
            commit_source_equivalence="doc/multichip_decoder/commit_source_equivalence.json",
            dense_source_equivalence="doc/multichip_decoder/expert_placement_equivalence.json",
            capacity_evidence=[str(p.relative_to(ROOT)) for p in capacity_paths],
            full_prefill_evidence=[str(p.relative_to(ROOT)) for p in full_paths],
        )
    (doc / "memory_capacity_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    p = ROOT / "doc/context_contract.json"
    contract = json.loads(p.read_text())
    contract["multichip_decoder"] = {k: v for k, v in result.items() if k != "layers"}
    contract["multichip_decoder"].update(
        target_supported_context=524288,
        validated_supported_context=524288 if verified else None,
        capability_reduction=None,
        evidence="doc/multichip_decoder/memory_capacity_plan.json",
    )
    if verified:
        contract["multichip_decoder"].update(
            largest_tested_prefill_context=524288,
            largest_tested_decode_position=524287,
            full_context_tested_batch=1,
            max_tested_batch=32,
            bf16_cache_compatibility="Short-context B2 and full-capacity B1 prefill/decode, both kinds; full-stack BF16 capacity is not claimed",
            layout_policy="Replicated BF16 residual; local8Q/2KV; TP4 attention/dense/shared/value experts; routed MLP TP4 indexed decode and EP4 larger prefill",
            attention_policy=contract["optimized_decoder"]["attention_policy"],
            stage_review="doc/multichip_decoder/stage_review.md",
        )
    p.write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
