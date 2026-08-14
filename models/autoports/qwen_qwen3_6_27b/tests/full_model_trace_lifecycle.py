# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused allocator-stability check for repeated split-trace setup/release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def dram_snapshot(mesh):
    ttnn.synchronize_device(mesh)
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    return {
        "allocated_bytes": banks * int(view.total_bytes_allocated_per_bank),
        "free_bytes": banks * int(view.total_bytes_free_per_bank),
        "largest_contiguous_bytes_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--layer-indices", type=int, nargs="+", default=[0, 3])
    args = parser.parse_args()
    if args.iterations < 3:
        raise ValueError("at least three iterations are required to compare two post-warmup releases")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=128,
            batch=1,
            layer_indices=args.layer_indices,
        )
        cycles = []
        for iteration in range(args.iterations):
            generator.setup_token_out_decode([220], [64])
            live = dram_snapshot(mesh)
            generator.token_out_decode_step(readback=False)
            generator._release_traces()
            released = dram_snapshot(mesh)
            cleared = all(
                getattr(generator, name) is None
                for name in (
                    "_trace_token",
                    "_trace_position",
                    "_trace_active_mask",
                    "_trace_active_state_mask",
                    "_trace_page_table",
                    "_trace_logits",
                    "_trace_sampled",
                    "_compat_token",
                    "_compat_position",
                    "_compat_logits",
                )
            )
            cycles.append({"iteration": iteration, "live": live, "released": released, "aliases_cleared": cleared})
            print(json.dumps(cycles[-1]), flush=True)

        stable_fields = ("allocated_bytes", "free_bytes", "largest_contiguous_bytes_free_per_bank")
        reference = cycles[1]["released"]
        allocator_stable = all(
            all(cycle["released"][field] == reference[field] for field in stable_fields) for cycle in cycles[2:]
        )
        result = {
            "iterations": args.iterations,
            "layer_indices": args.layer_indices,
            "cycles": cycles,
            "post_warmup_allocator_stable": allocator_stable,
            "all_aliases_cleared": all(cycle["aliases_cleared"] for cycle in cycles),
        }
        if not result["post_warmup_allocator_stable"] or not result["all_aliases_cleared"]:
            raise AssertionError(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
