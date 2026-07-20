# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exercise the non-aligned public maximum-context boundary through chunked prefill."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import build_generator


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
        generator = build_generator(
            model_dir,
            mesh,
            max_context_len=32768,
            weight_cache_path=weight_cache_path,
        )
        memory_after_weights = _dram_view(mesh)
        generator._ensure_kv_cache()
        memory_after_cache = _dram_view(mesh)
        prompt_len = 32767
        prompt = [1] * prompt_len
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        generated = generator.generate(prompt, 2, enable_trace=True, sampling_mode="device")
        ttnn.synchronize_device(mesh)
        elapsed_s = time.perf_counter() - start
        memory_after_execution = _dram_view(mesh)

        page_table = generator._page_table_host
        mapped_pages = int((page_table[0] >= 0).sum().item())
        final_physical_page = int(page_table[0, mapped_pages - 1].item())
        layer_rank_checks = []
        for layer, (key_cache, value_cache) in enumerate(generator._kv_cache):
            key_ranks = ttnn.get_device_tensors(key_cache)
            value_ranks = ttnn.get_device_tensors(value_cache)
            for rank, (key_rank, value_rank) in enumerate(zip(key_ranks, value_ranks, strict=True)):
                key_page = ttnn.to_torch(key_rank)[final_physical_page]
                value_page = ttnn.to_torch(value_rank)[final_physical_page]
                layer_rank_checks.append(
                    {
                        "layer": layer,
                        "rank": rank,
                        "key_page_abs_sum": float(key_page.float().abs().sum()),
                        "value_page_abs_sum": float(value_page.float().abs().sum()),
                    }
                )
        all_layer_rank_pages_nonzero = all(
            check["key_page_abs_sum"] > 0 and check["value_page_abs_sum"] > 0 for check in layer_rank_checks
        )
        current_positions = (
            generator._trace_inputs is not None
            and ttnn.to_torch(ttnn.get_device_tensors(generator._trace_inputs[1])[0]).reshape(-1).tolist()
        )
        rotary_positions = (
            generator._trace_inputs is not None
            and ttnn.to_torch(ttnn.get_device_tensors(generator._trace_inputs[2])[0]).reshape(-1).tolist()
        )
        result = {
            "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
            "layers_executed": generator.model.num_layers,
            "public_prompt_tokens": prompt_len,
            "requested_tokens": 2,
            "padded_tokens_owned_by_generator": 32768,
            "prefill_chunks": 16,
            "prefill_internal_row_chunks_per_layer": 32,
            "prefill_internal_row_chunks_total": 32 * generator.model.num_layers,
            "chunk_tokens": 2048,
            "page_block_tokens": generator.model.page_block_size,
            "mapped_pages": mapped_pages,
            "expected_pages": 1024,
            "final_physical_page": final_physical_page,
            "all_layer_rank_final_page_checks": layer_rank_checks,
            "checked_cache_tensors": len(layer_rank_checks) * 2,
            "all_layer_rank_final_pages_nonzero": all_layer_rank_pages_nonzero,
            "generated_tokens": [int(token) for token in generated],
            "trace_stats": dict(generator.trace_stats),
            "current_positions_after_decode": current_positions,
            "rotary_positions_after_decode": rotary_positions,
            "memory_after_weights": memory_after_weights,
            "memory_after_cache": memory_after_cache,
            "memory_after_execution": memory_after_execution,
            "elapsed_s": elapsed_s,
            "passed": (
                generator.model.num_layers == 28
                and mapped_pages == 1024
                and len(layer_rank_checks) == 28 * 4
                and all_layer_rank_pages_nonzero
                and len(generated) == 2
                and generator.trace_stats["replays"] == 1
                and current_positions == [32768]
                and rotary_positions == [32768]
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
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    args = parser.parse_args()
    result = collect(args.model_dir, args.output, args.weight_cache_path)
    if not result["passed"]:
        raise SystemExit("full-context coverage failed")


if __name__ == "__main__":
    main()
