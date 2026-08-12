# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Precision-locked real-weight LM-head program-config sweep."""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import build_generator
from models.autoports.tiiuae_falcon3_7b_base.tt.optimized_decoder import (
    _advisor_matmul_program_config,
    _dram_matmul_program_config,
    _sharded_memory_config,
)
from models.common.readiness_check.schema import load_reference


def _timed(mesh, fn, iterations: int) -> float:
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - start) * 1000.0 / iterations


def collect(model_dir: Path, reference: Path, output: Path, *, iterations: int, weight_cache_path: str) -> dict:
    if not ttnn.CONFIG.throw_exception_on_fallback:
        raise RuntimeError("LM-head sweep requires TTNN throw_exception_on_fallback=true")
    prompt = load_reference(reference).entries[0].prompt_tokens[0].tolist()[:128]
    mesh = None
    generator = None
    results = []
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=512_000_000)
        generator = build_generator(
            model_dir,
            mesh,
            override_num_layers=1,
            max_context_len=32768,
            weight_cache_path=weight_cache_path,
        )
        # K is 96 tiles over the inherited 32-core residual shard, leaving
        # three K tiles/core.  The only legal divisors are therefore 1 and 3.
        for block_w in (1, 3):
            generator.reset()
            generator.model.lm_head.config.program_configs[0] = _dram_matmul_program_config(
                32,
                generator.model.hidden_size,
                generator.model.local_vocab_size,
                generator.model.layers[-1].residual_grid,
                in0_block_w=block_w,
            )
            generated = generator.generate(prompt, 2, sampling_mode="device", enable_trace=True)
            model_ms = _timed(
                mesh,
                lambda: ttnn.execute_trace(mesh, generator._trace_model_id, cq_id=0, blocking=False),
                iterations,
            )
            pair_ms = _timed(mesh, generator._replay_split_sampling, iterations)
            results.append(
                {
                    "in0_block_w": block_w,
                    "model_trace_ms_per_token": model_ms,
                    "token_out_ms_per_token": pair_ms,
                    "generated_tokens": generated,
                }
            )
        winner = min(results, key=lambda item: item["token_out_ms_per_token"])
        generator.reset()
        selected_weight = generator.model.lm_head.output_weights[0]
        generator.model.lm_head.output_weights[0] = ttnn.to_memory_config(selected_weight, ttnn.DRAM_MEMORY_CONFIG)
        adapted = []
        for cores, grid, block_w, per_core_n in (
            (8, ttnn.CoreGrid(x=8, y=1), 12, 128),
            (16, ttnn.CoreGrid(x=8, y=2), 6, 64),
        ):
            generator.reset()
            generator.model.lm_head.config.input_memcfg = _sharded_memory_config(32, 3072, grid)
            generator.model.lm_head.config.program_configs[0] = _advisor_matmul_program_config(
                grid=(int(grid.x), int(grid.y)),
                in0_block_w=block_w,
                per_core_n=per_core_n,
                out_subblock_w=4,
            )
            candidate = {
                "cores": cores,
                "grid": [int(grid.x), int(grid.y)],
                "in0_block_w": block_w,
                "per_core_n": per_core_n,
                "includes_terminal_reshard": True,
            }
            try:
                generated = generator.generate(prompt, 2, sampling_mode="device", enable_trace=True)
                candidate.update(
                    {
                        "token_out_ms_per_token": _timed(mesh, generator._replay_split_sampling, iterations),
                        "generated_tokens": generated,
                        "result": "measured",
                    }
                )
            except RuntimeError as error:
                candidate.update({"result": "blocked", "exact_error": str(error).split("backtrace:", 1)[0].strip()})
            adapted.append(candidate)
        result = {
            "run_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "fallback_policy": "throw_exception_on_fallback=true (asserted by runner)",
            "command_environment": "TTNN_CONFIG_OVERRIDES={throw_exception_on_fallback:true}",
            "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
            "weights": "real Falcon3 LM head BFP4, LoFi compute, BFP8 output",
            "fixed_geometry": "32-core inherited residual shard; K=96 tiles, 3 K tiles/core; N=1024 tiles, 32 N tiles/core",
            "grid_constraint": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig exposes no grid or output-subblock fields; changing the 32-core grid would require resharding the preserved inter-layer residual at the terminal boundary",
            "iterations": iterations,
            "candidates": results,
            "winner": winner,
            "adapted_terminal_layout_candidates": adapted,
            "passed": bool(
                len({tuple(item["generated_tokens"]) for item in results}) == 1
                and adapted[0]["result"] == "blocked"
                and adapted[1]["result"] == "measured"
                and adapted[1]["generated_tokens"] == winner["generated_tokens"]
                and adapted[1]["token_out_ms_per_token"] > winner["token_out_ms_per_token"]
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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    args = parser.parse_args()
    result = collect(
        args.model_dir,
        args.reference,
        args.output,
        iterations=args.iterations,
        weight_cache_path=args.weight_cache_path,
    )
    if not result["passed"]:
        raise SystemExit("LM-head program-config sweep changed greedy tokens")


if __name__ == "__main__":
    main()
