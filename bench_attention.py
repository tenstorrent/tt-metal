# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device time for our attention kernels, next to ttnn's SDPA at the same shape.

READ THE CORE COUNTS BEFORE THE NUMBERS. Our kernels are single-core; ttnn's SDPA is written
for the whole grid. The absolute figures are therefore not a like-for-like comparison of
implementations -- they compare one core against sixty-four. The per-core column is the closer
thing to a fair read, and even that flatters us, because ttnn is solving the general problem
(batches, heads, GQA, arbitrary sequence lengths) while these kernels do one head at one shape.

What the numbers ARE good for: a baseline to optimise our kernels against, and a sanity check
that we are in the right order of magnitude rather than a hundred times off.

    python bench_attention.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_bench import bench, show

TILE = 32


def bench_ours_flash(device, sq, sk_total, dt, chunks):
    import test_unified_flash as flash

    # Build once outside the timed region by capturing the closure the test uses.
    call = lambda: flash.run(device, sq, sk_total, dt, chunks, True)
    return bench(device, call, iters=20, warmup=3, match="flash_attention.cpp")


def bench_ours_nonflash(device, sq, sk, dt):
    import test_unified_attention as attn

    call = lambda: attn.run(device, sq, sk, dt, True)
    return bench(device, call, iters=20, warmup=3, match="attention.cpp")


def bench_ttnn(device, s, d, q_chunk, k_chunk, cores=None):
    q = torch.randn([1, 1, s, d])
    k = torch.randn([1, 1, s, d])
    v = torch.randn([1, 1, s, d])
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tq, tk, tv = ttnn.from_torch(q, **kw), ttnn.from_torch(k, **kw), ttnn.from_torch(v, **kw)

    # Pinning the grid is what makes this comparable: at S=128 with one head there are only a
    # few q-chunks of work, so the full grid leaves most cores idle and any per-core figure
    # derived from it is fiction. One core against one core is the honest comparison.
    grid = ttnn.CoreCoord(1, 1) if cores == 1 else device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )

    call = lambda: ttnn.transformer.scaled_dot_product_attention(
        tq, tk, tv, is_causal=True, program_config=pc, compute_kernel_config=ckc
    )
    stats = bench(device, call, iters=20, warmup=3, match="sdpa")
    return stats, grid


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--s", type=int, default=128, help="sequence length (queries and keys)")
    p.add_argument("--d", type=int, default=64, help="head dim")
    args = p.parse_args(argv)

    sq = args.s // TILE
    dt = args.d // TILE

    device = ttnn.open_device(device_id=0)
    try:
        print(f"\n=== S={args.s} D={args.d}, causal, one head ===\n")

        # Ours, non-flash: needs sq*sk <= 8 tiles, so only small S fits in one shot.
        if sq * sq <= 8 and sq * dt <= 8:
            st = bench_ours_nonflash(device, sq, sq, dt)
            show("ours: attention (1 core)", st)
        else:
            print(f"ours: attention (1 core)          skipped -- sq*sk={sq * sq} tiles over the 8-tile DST budget")

        # Ours, flash: chunk K so sq*sk_chunk <= 8.
        for chunks in (1, 2, 4):
            if sq % chunks or (sq // chunks) * sq > 8 or sq * dt > 8:
                continue
            st = bench_ours_flash(device, sq, sq, dt, chunks)
            show(f"ours: flash, {chunks} chunk(s) (1 core)", st)

        # ttnn, whole grid.
        for pin in (1, None):
            for qc, kc in ((32, args.s), (args.s, args.s)):
                try:
                    st, grid = bench_ttnn(device, args.s, args.d, qc, kc, cores=pin)
                    n = grid.x * grid.y
                    label = f"ttnn: SDPA q{qc}/k{kc} ({n} core{'s' if n > 1 else ''})"
                    show(label, st)
                except Exception as e:  # noqa: BLE001 -- report and continue; configs are picky
                    print(f"ttnn: SDPA q{qc}/k{kc} pin={pin}        unsupported: {str(e).splitlines()[0][:80]}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
