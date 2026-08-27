# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the two-stage reduction tree on device: sum an ht x wt tile block down its
rows, gather each column's results, and sum those too.

Exercises the reduce fusion kind (unified_kernels/reduction_tree.cpp), where
metal's reduce folds a tile grid along one axis -- within AND across tiles -- and
the packer zeroes everything that is not part of the result. Also the only test
that drives noc_core_write: the column gather is one push per core into row 0.

Every core reads the SAME input block, so all columns compute the same answer.
That is why the kernel's roots can share one output block index without clobbering
each other -- a kernel with per-column inputs would need to index the output by
column. in1 is unused by the kernel; it exists only in the accessor arg layout.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_reduction.py --ht 4 --wt 4 --grid-h 2 --grid-w 2

Reducing 128x128 (4x4 tiles) over rows leaves a single valid 1x128 row living in a
1x4 tile block, which is what --ht 4 --wt 4 checks.
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, dfb_output, run_unified_spec, unified_program_spec

KERNEL = "unified_kernels/reduction_tree.cpp"

TILE = 32


def run(device, ht=4, wt=4, grid_h=2, grid_w=2, num_blocks=1, op="sum", single_stage=False, seed=0):
    torch.manual_seed(seed)
    # One input block per (block, column): column x reads b * grid_w + x, so every
    # column reduces different data and a mis-indexed column shows up as garbage.
    # Values kept small and centred so a 32*ht-deep bfloat16 sum stays accurate.
    a = ((torch.rand([1, 1, num_blocks * grid_w * ht * TILE, wt * TILE]) - 0.5) * 0.5).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    # One reduced tile-row per block, holding every column's result side by side:
    # column x lands at block index b * grid_w + x. Only row 0 of each carries the
    # answer. Sized this way so a column writing the wrong slot is visible.
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, num_blocks * TILE, grid_w * wt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))])
    cores = [ttnn.CoreCoord(x, y) for y in range(grid_h) for x in range(grid_w)]

    named_ct_args = [("num_blocks", num_blocks), ("in_ht", ht), ("in_wt", wt), ("num_cores_y", grid_h)]
    defines = (
        ([("RT_MAX", "1")] if op == "max" else [])
        + ([("RT_MEAN", "1")] if op == "mean" else [])
        # max and mean are only correct single-stage: stage 2 would fold the
        # zeros the packer wrote outside stage 1's result row.
        + ([("RT_SINGLE_STAGE", "1")] if single_stage else [])
    )

    # Endpoint roles, read straight off the kernel:
    #   in0     noc_load<0>            -> DM thread 0 fills it, compute reads it
    #   scaler  fill_reduce_scaler<1>  -> DM thread 1 fills it once, compute re-reads it
    #   tmp0    compute stores it, then DM thread 0 drains it -- into the gather in the
    #           two-stage shape, or straight to DRAM when single-stage. Either way thread 0
    #           for the gather and thread 1 for the direct store, so the role differs.
    #   tmp1    the gather destination: DM thread 0 writes it, compute reduces it. Declared
    #           even single-stage, where nothing touches it, because the kernel declares its
    #           Storage unconditionally -- exactly as the circular buffer was allocated
    #           unconditionally before.
    #   out     compute stores it, DM thread 1 drains it
    dfbs = [
        dfb("in0", ht * wt),
        # tmp0's drain differs by shape -- the gather takes it on thread 0, the single-stage
        # store on thread 1 -- and both spellings live in the source, so it is stated.
        dfb_output("tmp0", thread=1 if single_stage else 0, num_pages=wt),
        dfb("tmp1", wt * grid_h),
        dfb("scaler", 1),
        dfb("out", wt),
    ]

    logger.info(
        f"running reduction tree: op={op} single_stage={single_stage} ht={ht} wt={wt} "
        f"grid={grid_h}x{grid_w} num_blocks={num_blocks} ({ht * TILE}x{wt * TILE} per core)"
    )

    tensors = {"in0": ta, "out": tout}
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        tensors=tensors,
        named_compile_time_args=named_ct_args,
        defines=defines,
        name="reduction_tree",
    )
    run_unified_spec(device, spec, tensors)
    out = tout
    got_full = ttnn.to_torch(out).to(torch.float32)

    af = a.to(torch.float32)
    got_rows, want_rows = [], []
    for b in range(num_blocks):
        for x in range(grid_w):
            # Column x reduces input block b * grid_w + x down its rows; the grid_h
            # cores of the column all read that block, so the gather sums grid_h
            # copies of it. Distinct data per column, so the check catches a column
            # reading or writing the wrong index.
            k = b * grid_w + x
            block = af[0, 0, k * ht * TILE : (k + 1) * ht * TILE, :]
            if op == "max":
                folded = block.max(dim=0).values
            elif op == "mean":
                folded = block.mean(dim=0)
            else:
                folded = block.sum(dim=0)
            # Two-stage sums grid_h copies of the column's result; single-stage is
            # just the fold, so there is nothing to scale.
            want_rows.append(folded if single_stage else folded * grid_h)
            got_rows.append(got_full[0, 0, b * TILE, x * wt * TILE : (x + 1) * wt * TILE])

    # Everything outside row 0 of each block is the packer's zeroing contract.
    masked = torch.cat([got_full[0, 0, b * TILE + 1 : (b + 1) * TILE, :].flatten() for b in range(num_blocks)])
    return torch.stack(got_rows), torch.stack(want_rows), masked.abs().max().item()


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ht", type=int, default=4, help="input block height in tiles")
    p.add_argument("--wt", type=int, default=4, help="input block width in tiles")
    p.add_argument("--grid-h", type=int, default=2, help="core rows: the gather's height")
    p.add_argument("--grid-w", type=int, default=2, help="core columns, each reducing independently")
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--op", choices=["sum", "max", "mean"], default="sum")
    p.add_argument("--single-stage", action="store_true", help="stage 1 only -- required for max/mean, see the kernel")
    p.add_argument("--pcc", type=float, default=0.99)
    p.add_argument("--rtol", type=float, default=0.05, help="PCC cannot see a wrong scaler; this can")
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        got, want, masked = run(
            device, args.ht, args.wt, args.grid_h, args.grid_w, args.num_blocks, args.op, args.single_stage
        )
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    # PCC is invariant to a global scale factor, so on its own it cannot tell a
    # mean from a sum -- a mean fed the wrong scaler is exactly N times too large
    # and still correlates perfectly. Relative magnitude is what catches that.
    scale = max(want.abs().max().item(), 1e-6)
    rel = (got - want).abs().max().item() / scale
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")
    logger.info(f"max |got - want| / max|want| = {rel:.4f} (threshold {args.rtol})")
    logger.info(f"got [0,:4]  = {got[0, :4].tolist()}")
    logger.info(f"want[0,:4]  = {want[0, :4].tolist()}")
    # A reduction that leaked into the masked rows is wrong even at a good PCC.
    logger.info(f"max |masked rows| = {masked:.6f}  (0 means the packer zeroed them)")

    if measured < args.pcc or rel > args.rtol:
        logger.error("FAIL")
        return 1
    if masked != 0.0:
        logger.error("FAIL: reduction wrote outside row 0")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
