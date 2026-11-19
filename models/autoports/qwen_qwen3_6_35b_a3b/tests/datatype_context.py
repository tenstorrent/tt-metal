# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Recompute KV capacity arithmetic without lowering advertised capability."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "doc/context_contract.json"
    contract = json.loads(path.read_text())
    full = contract["full_model_validation"]
    capacity = full["capacity"]
    context = full["supported_context"]
    assert context == 262144
    # K and V each hold one local256-wide head in every one of10 full layers.
    tiles = 10 * 2 * (context // 32) * (256 // 32)
    physical = {dtype: tiles * tile_bytes for dtype, tile_bytes in [("bf16", 2048), ("bfp8", 1088)]}
    assert physical["bf16"] == capacity["full_attention_kv_bf16_bytes_per_device"]
    projected_free = contract["optimized_full_model_validation"]["historical_max_projected_free_bytes_per_device"]
    rows = {}
    for dtype, bytes_per_device in physical.items():
        rows[dtype] = dict(
            supported_context=context,
            aggregate_tokens=context,
            fixed_slots=32,
            local_kv_heads_per_device=1,
            full_attention_layers=10,
            kv_layout="TILE",
            kv_bytes_per_device=bytes_per_device,
            tile_bytes=2048 if dtype == "bf16" else 1088,
            free_dram_delta_vs_bf16_bytes_per_device=physical["bf16"] - bytes_per_device,
            projected_max_context_free_bytes_per_device=projected_free + physical["bf16"] - bytes_per_device,
            recurrent_state_dtype="fp32",
            convolution_state_dtype="bf16",
            capability_reduction=None,
            method="Exact KV tile payload arithmetic; stage7 historical maximum free-memory projection with baseline weights. No runtime peak or new maximum acceptance claimed.",
            prior_measurement="full_model/context_acceptance/final.json",
            later_stage_projection="optimized_full_model/context_memory_audit.json",
        )
    contract["datatype_sweep_capacity_candidates"] = rows
    path.write_text(json.dumps(contract, indent=2) + "\n")
    out = ROOT / "doc/datatype_sweep/context_candidates.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
