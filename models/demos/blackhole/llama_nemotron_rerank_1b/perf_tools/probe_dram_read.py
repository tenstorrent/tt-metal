# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Weight-block DRAM read rate per core: per-tile TensorAccessor reads vs bank-grouped stateful reads, on
``--cores`` cores at once (13 = a column of weight readers), via the op's device kernel time.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_dram_read.py --kb 16 --nb 8
"""
import argparse
import os
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops import _mm_grid as mg

KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "dram_read_bench.cpp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=int, default=16)
    ap.add_argument("--nb", type=int, default=8)
    ap.add_argument("--cores", type=int, default=13)
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    K, N = 64, args.nb * args.cores
    w = ttnn.from_torch(torch.randn(K * 32, N * 32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    page = 1088
    gx = 13
    cs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c % gx, c // gx), ttnn.CoreCoord(c % gx, c // gx)) for c in range(args.cores)]
    )
    for mode in (0, 1):
        kern = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[args.kb, args.nb, N, K, args.reps, mode, page] + mg.accessor_args(w),
            runtime_args=[((c % gx, c // gx), [w.buffer_address(), c * args.nb]) for c in range(args.cores)],
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[kern], cbs=[mg.cb(0, args.kb * args.nb, ttnn.bfloat8_b, cs)], semaphores=[]
        )
        run = lambda: ttnn.generic_op([w, w], prog)
        run()
        ttnn.synchronize_device(dev)
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            run()
            ttnn.synchronize_device(dev)
            best = min(best, time.perf_counter() - t)
        blk = args.kb * args.nb * page
        cyc = best * 1.35e9 / args.reps
        print(
            f"mode {mode} ({'per-tile accessor' if mode == 0 else 'bank-grouped stateful'}): {cyc:.0f} cycles per "
            f"{args.kb}x{args.nb} block ({blk/1024:.0f} KB) = {blk/cyc:.1f} B/cycle per core on {args.cores} cores",
            flush=True,
        )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
