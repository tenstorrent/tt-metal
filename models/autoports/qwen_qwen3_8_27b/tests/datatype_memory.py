# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recompute full-context capacity from candidate tile sizes and inherited geometry."""

import argparse
import hashlib
import json
from pathlib import Path

from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy, load_precision

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"
TILE_BYTES = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    policy = load_precision(args.config)
    contract_path = ROOT / "doc/context_contract.json"
    contract = json.loads(contract_path.read_text())
    config = json.loads((ROOT / "doc/functional_decoder/hf_config.json").read_text())["text_config"]
    prior = json.loads((ROOT / "doc/optimized_multichip_decoder/memory_capacity_plan.json").read_text())
    context = config["max_position_embeddings"]
    assert context == policy["max_context"] == contract["hf_advertised_context"]
    resources = dict(contract["full_model"]["resources_bytes_per_device"])
    projection_bytes = 0
    mapping = {"mlp_gate_up": "gate", "mlp_down": "down", "attention": "attention", "output": "output"}
    for layer, kind in enumerate(config["layer_types"]):
        decoder = decoder_policy(policy, layer)
        for group, geometry in prior["layers"][kind]["projections"].items():
            tile = TILE_BYTES[decoder[mapping[group] + "_dtype"]]
            assert geometry["total_bytes"] % 576 == 0
            projection_bytes += geometry["total_bytes"] // 576 * tile
    resources.pop("decoder_projections_bfp4_both_layouts")
    resources["decoder_projections_selected_both_layouts"] = projection_bytes
    resources.pop("full_kv_bfp8")
    full = config["layer_types"].count("full_attention")
    # TP4 local KV heads=1, PAGE_SIZE=32, head_dim=256, two cache tensors/layer.
    resources["full_kv_selected"] = (
        full * 2 * ((context + 31) // 32) * (config["head_dim"] // 32) * TILE_BYTES[policy["kv_cache_dtype"]]
    )
    resources.pop("lm_head_bfp8_vocab_shard")
    resources.pop("lm_head_bfp8_dram_decode_copy")
    k, n = config["hidden_size"], config["vocab_size"] // 4
    tile = TILE_BYTES[policy["weight_groups"]["head"]]
    resources["lm_head_selected_vocab_shard"] = (k // 32) * (n // 32) * tile
    resources["lm_head_selected_dram_decode_copy"] = sum(
        (k // 32) * (((min(16384, n - start) + 511) // 512) * 512 // 32) * tile for start in range(0, n, 16384)
    )
    old = contract["optimized_full_model"]
    for key in (
        "history_persistent_max_bytes_per_device",
        "history_append_scratch_max_bytes_per_device",
        "history_cursor_and_alignment_reserve_bytes_per_device",
        "prefill_trace_persistent_reserve_bytes_per_device",
    ):
        resources[key] = old[key]
    total, physical = sum(resources.values()), old["device_dram_bytes"]
    assert total < physical, "Candidate cannot preserve the advertised full context"
    report = dict(
        config_id=policy["config_id"],
        supported_context=context,
        hf_advertised_context=context,
        capability_reduction=None,
        kv_dtype=policy["kv_cache_dtype"],
        page_size=32,
        local_kv_heads=1,
        head_dim=config["head_dim"],
        mesh=[1, 4],
        context_test_batch_size=1,
        public_sequence_alignment_requirement=None,
        resources_bytes_per_device=resources,
        conservative_total_bytes_per_device=total,
        device_dram_bytes=physical,
        headroom_bytes_per_device=physical - total,
        prefill_chunk_size=4096,
        arithmetic_scope="Full64 weights in prefill/decode layouts, selected KV and head; inherited conservative scratch/CCL/history reserves. Capacity arithmetic is not an execution test.",
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")},
    )
    path = DOC / ("memory_" + policy["config_id"] + ".json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    if args.select:
        contract["datatype_sweep"] = dict(report, evidence=str(path.relative_to(ROOT / "doc")))
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("config_id", "supported_context", "kv_dtype", "headroom_bytes_per_device")})
    )


if __name__ == "__main__":
    main()
