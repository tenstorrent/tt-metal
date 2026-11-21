# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score head ([rows, 2048] @ [2048, 32], fp32 out): minimal_matmul (N = 1 tile, so one row of cores
works) vs a 1D matmul that splits the rows over every core. Accuracy and traced time.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_score.py [--m 16384]
"""
import argparse
import time

import torch

import ttnn

K, N, REPS = 2048, 32, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    args = ap.parse_args()
    M = args.m
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    grid = dev.compute_with_storage_grid_size()
    torch.manual_seed(0)
    x = torch.randn(1, 1, M, K)
    w = torch.randn(K, N) / K**0.5
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def old():
        return ttnn.experimental.minimal_matmul(
            tx,
            tw,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=16,
                K_block_size=8,
                N_block_size=1,
                subblock_h=4,
                subblock_w=1,
                compute_with_storage_grid_size=grid,
            ),
            compute_kernel_config=ckc,
            dtype=ttnn.float32,
        )

    mt = M // 32
    cores = grid.x * grid.y
    per_core_m = -(-mt // cores)

    def new(kb):
        return ttnn.matmul(
            tx,
            tw,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=kb,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=per_core_m,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
            compute_kernel_config=ckc,
            dtype=ttnn.float32,
        )

    ref = ttnn.to_torch(tx).float() @ ttnn.to_torch(tw).float()

    def timed(fn):
        fn()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        outs = [fn() for _ in range(REPS)]
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            best = min(best, (time.perf_counter() - t) / REPS * 1e6)
        ttnn.release_trace(dev, tid)
        for o in outs:
            ttnn.deallocate(o)
        return best

    for name, fn in [("minimal_matmul", old)] + [
        (f"1D in0_block_w={kb}", (lambda kb=kb: new(kb))) for kb in (2, 4, 8, 16)
    ]:
        try:
            out = ttnn.to_torch(fn()).float().reshape(ref.shape)
            err = (out - ref).abs().max().item()
            print(f"{name}: max|err| {err:.4g}  {timed(fn):.1f} us")
        except Exception as e:
            print(f"{name}: failed {str(e).splitlines()[0][:200]}")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
