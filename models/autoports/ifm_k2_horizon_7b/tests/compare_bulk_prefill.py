"""Full36 integration of current-policy bulk-prefill block/grid winners."""

import argparse
import json
import statistics
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import K2Generator
from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_full_model_policy import OptimizedFullModelDecoder


def patches(mode):
    stack = ExitStack()
    # Retain the original control after promotion of the winning production
    # hook; all candidate changes below are applied on that same control.
    stack.enter_context(
        patch.object(
            OptimizedFullModelDecoder,
            "_prefill_fused_matmul_config",
            MultichipDecoder._prefill_fused_matmul_config,
        )
    )
    if mode == "baseline":
        return stack
    original_agmm = ttnn.experimental.all_gather_minimal_matmul_async
    original_mm = ttnn.experimental.minimal_matmul

    def agmm(x, w, **kwargs):
        small = x.shape[2] <= 512
        if kwargs["fuse_swiglu"] and (small or w.dtype == ttnn.bfloat4_b):
            kwargs["config"] = ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=16,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            )
        elif mode == "all" and not kwargs["fuse_swiglu"] and x.dtype == ttnn.bfloat16 and not small:
            kwargs["config"] = ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=8,
                N_block_size=16,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
            kwargs["num_workers_per_link"] = 4
        return original_agmm(x, w, **kwargs)

    def mm(x, w, **kwargs):
        if mode in ("mlp_down", "mlp_down_m2", "all") and x.shape[2] > 512 and tuple(w.shape) == (3072, 4096):
            kwargs["config"] = ttnn.MinimalMatmulConfig(
                M_block_size=4 if mode == "mlp_down" else 2,
                K_block_size=16,
                N_block_size=16,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            )
        return original_mm(x, w, **kwargs)

    stack.enter_context(patch.object(ttnn.experimental, "all_gather_minimal_matmul_async", agmm))
    stack.enter_context(patch.object(ttnn.experimental, "minimal_matmul", mm))
    return stack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[257, 4096])
    parser.add_argument("--modes", nargs="+", default=["baseline", "mlp", "mlp_down", "all", "baseline"])
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(16)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    result = {"pass": False, "layers": 36, "generation_len": 128, "records": []}
    try:
        gen = K2Generator(mesh)
        for length in args.lengths:
            prompt = gen.tokenizer.encode(
                "The sky appears blue because sunlight scatters in the atmosphere. " * length
            )[:length]
            baseline_tokens = None
            for mode in args.modes:
                gen._release_traces()
                with patches(mode):
                    gen.generate(prompt, 128)
                    runs = []
                    for _ in range(args.trials):
                        tokens = gen.generate(prompt, 128)
                        runs.append(gen.last_perf.copy())
                    if baseline_tokens is None:
                        baseline_tokens = tokens
                    row = {
                        "prompt_len": length,
                        "mode": mode,
                        "runs": runs,
                        "ttft_ms": statistics.median(r["ttft_seconds"] for r in runs) * 1000,
                        "all_tokens_equal_baseline": tokens == baseline_tokens,
                        "matching_tokens": sum(a == b for a, b in zip(tokens, baseline_tokens)),
                        "tokens": tokens,
                    }
                    result["records"].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(
                        "BULK_FULL36",
                        json.dumps({k: v for k, v in row.items() if k not in ("runs", "tokens")}),
                        flush=True,
                    )
            gen._release_traces()
        result["pass"] = True
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
