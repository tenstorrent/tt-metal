# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure what one pass costs, by running out = in through N identity passes.

Not a correctness test of anything interesting -- every pass is a copy, so the
answer is the input and any deviation means the plumbing is broken.  The point is
the SLOPE: with the math pinned at zero, runtime against N is the cost of one L1
round trip of `tiles` tiles plus its CB handshake.  That number decides whether
fusing passes in a real kernel is worth the implementation cost.

    python test_unified_passcost.py --tiles 8
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/passcost.cpp"

CB_IN, CB_OUT = 0, 16
CB_SCRATCH = range(1, 8)


def run(device, passes, tiles=8, seed=0):
    shape = [1, tiles, 32, 32]

    torch.manual_seed(seed)
    a = (0.5 + 1.5 * torch.rand(shape)).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()

    ct_args = [tiles]
    for t in (ta, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    rt_args = [ta.buffer_address(), tout.buffer_address()]

    cbs = [make_cb(cb, core_ranges, num_pages=2 * tiles) for cb in (CB_IN, CB_OUT, *CB_SCRATCH)]

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=cores,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        defines=[("PASSES", str(passes))],
    )

    out = ttnn.generic_op([ta, tout], program)
    return ttnn.to_torch(out).to(torch.float32), a.to(torch.float32)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", type=int, default=8)
    p.add_argument("--max-passes", type=int, default=8)
    args = p.parse_args(argv)

    from unified_bench import bench

    device = ttnn.open_device(device_id=0)
    rows, failed = [], []
    try:
        for n in range(1, args.max_passes + 1):
            got, want = run(device, n, args.tiles)
            err = (got - want).abs().max().item()
            if err != 0.0:
                # A copy chain must be bit-exact; anything else is broken plumbing.
                logger.error(f"passes={n}: identity chain altered the data, max |diff| = {err}")
                failed.append(n)
            st = bench(device, lambda n=n: run(device, n, args.tiles), iters=20, warmup=3, match="passcost.cpp")
            rows.append((n, st["median_us"]))
    finally:
        ttnn.close_device(device)

    logger.info(f"identity passes over {args.tiles} tiles:")
    prev = None
    for n, us in rows:
        delta = f"  +{us - prev:6.2f}us" if prev is not None else ""
        logger.info(f"  passes={n}  median={us:7.2f}us{delta}")
        prev = us
    if len(rows) >= 2:
        slope = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
        logger.info(f"slope = {slope:.2f}us per pass ({slope / args.tiles:.3f}us per tile per pass)")

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
