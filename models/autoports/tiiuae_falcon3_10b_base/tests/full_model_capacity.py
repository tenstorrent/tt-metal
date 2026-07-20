# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure the batch-32, 32K full-model DRAM capacity contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tt.generator import build_generator


def _dram_view(mesh_device) -> dict[str, int]:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "banks": int(view.num_banks),
        "total_bytes_per_device": int(view.num_banks * view.total_bytes_per_bank),
        "allocated_bytes_per_device": int(view.num_banks * view.total_bytes_allocated_per_bank),
        "free_bytes_per_device": int(view.num_banks * view.total_bytes_free_per_bank),
        "largest_contiguous_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
    }


def collect(model_dir: Path, output: Path, weight_cache_path: str) -> dict:
    mesh = None
    generator = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=512_000_000)
        baseline = _dram_view(mesh)
        generator = build_generator(
            model_dir,
            mesh,
            max_batch_size=32,
            max_context_len=32768,
            weight_cache_path=weight_cache_path,
        )
        after_weights = _dram_view(mesh)
        generator._ensure_kv_cache()
        after_cache = _dram_view(mesh)

        weight_bytes = after_weights["allocated_bytes_per_device"] - baseline["allocated_bytes_per_device"]
        cache_bytes = after_cache["allocated_bytes_per_device"] - after_weights["allocated_bytes_per_device"]
        combined_bytes = after_cache["allocated_bytes_per_device"] - baseline["allocated_bytes_per_device"]
        result = {
            "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
            "layers": generator.model.num_layers,
            "batch": generator.batch,
            "context_tokens": generator.model.max_cache_len,
            "page_block_tokens": generator.model.page_block_size,
            "pages_per_user": generator.pages_per_user,
            "physical_cache_blocks": generator.num_blocks,
            "cache_dtype": "BFP8_B paged, one local KV head/rank",
            "baseline": baseline,
            "after_weights": after_weights,
            "after_batch32_full_context_cache": after_cache,
            "measured_weight_bytes_per_device": weight_bytes,
            "measured_kv_cache_bytes_per_device": cache_bytes,
            "measured_weight_plus_kv_bytes_per_device": combined_bytes,
            "measured_remaining_bytes_per_device": after_cache["free_bytes_per_device"],
            "passed": bool(
                generator.model.num_layers == 40
                and generator.batch == 32
                and generator.model.max_cache_len == 32768
                and generator.pages_per_user == 1024
                and generator.num_blocks == 32768
                and weight_bytes > 0
                and cache_bytes > 0
                and combined_bytes < after_cache["total_bytes_per_device"]
                and after_cache["free_bytes_per_device"] > 0
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-10b-full-model-cache")
    args = parser.parse_args()
    result = collect(args.model_dir, args.output, args.weight_cache_path)
    if not result["passed"]:
        raise SystemExit("batch-32 full-context capacity gate failed")


if __name__ == "__main__":
    main()
