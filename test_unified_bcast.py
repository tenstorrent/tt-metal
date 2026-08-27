# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Broadcast: {add, sub, mul} x {Rows, Cols, Both}, measured rather than assumed.

This test DETERMINES which metal call each axis lowers to. Metal's own documentation
cannot settle it: for BroadcastType::COL it asserts both that B is "a single tile with a
filled 0-column" (B indexed by h) and that the result is C[h,w] = A[h,w] + B[w] (B indexed
by w). Those are opposite claims. Only the fused bias pinned the Rows direction before
this, and nothing in the repo used the Cols one at all.

Two properties make the test unable to pass wrongly, and both are load-bearing:

  NON-SQUARE block (2x3 tiles). The Rows vector is then Shape<1,3> and the Cols vector
  Shape<2,1> -- different types, so an axis mix-up on our side fails to COMPILE.

  Vector entries VARY along the vector's length. This catches the real risk. If
  Axis::Cols emitted an op that reads B as a row, a column vector offers only v[0] at
  [0,0] with zeros across its row 0, so the result would collapse to `a + v[0]` in one
  tile-column and `a + 0` elsewhere. A constant-valued vector would hide exactly that.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_bcast.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/bcast.cpp"
TILE = 32

OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}


def vec_tensor(axis, ht, wt):
    """The vector, laid out as the axis requires, with entries that VARY so a wrong
    direction cannot coincidentally agree."""
    if axis == "Rows":
        # Shape<1, wt> tiles: one row of values, meaningful in row 0 of each tile.
        vals = 1.0 + torch.arange(wt * TILE, dtype=torch.float32) * 0.03
        t = torch.zeros([1, 1, TILE, wt * TILE])
        t[0, 0, 0, :] = vals
        return t.to(torch.bfloat16), vals
    if axis == "Cols":
        # Shape<ht, 1> tiles: one column of values, meaningful in column 0 of each tile.
        vals = 1.0 + torch.arange(ht * TILE, dtype=torch.float32) * 0.03
        t = torch.zeros([1, 1, ht * TILE, TILE])
        t[0, 0, :, 0] = vals
        return t.to(torch.bfloat16), vals
    # Both: a single value at [0, 0].
    t = torch.zeros([1, 1, TILE, TILE])
    t[0, 0, 0, 0] = 1.75
    return t.to(torch.bfloat16), torch.tensor([1.75])


def reference(op, axis, a, vals):
    f = OPS[op]
    if axis == "Rows":
        return f(a, vals.unsqueeze(0))  # per column, same for every row
    if axis == "Cols":
        return f(a, vals.unsqueeze(1))  # per row, same for every column
    return f(a, vals[0])


def run(device, op, axis, ht=2, wt=3, then_sfpu=False, seed=0):
    torch.manual_seed(seed)
    a = (1.0 + torch.rand([1, 1, ht * TILE, wt * TILE])).to(torch.bfloat16)
    vec, vals = vec_tensor(axis, ht, wt)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tv = ttnn.from_torch(vec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, ht * TILE, wt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core_ranges, cores = single_core()
    named_ct_args = [("ht", ht), ("wt", wt)]

    vec_pages = {"Rows": wt, "Cols": ht, "Both": 1}[axis]
    cbs = [
        make_cb(CB_BLOCK, core_ranges, num_pages=ht * wt),
        make_cb(CB_VEC, core_ranges, num_pages=vec_pages),
        make_cb(CB_OUT, core_ranges, num_pages=ht * wt),
    ] + ([make_cb(CB_TMP, core_ranges, num_pages=ht * wt)] if then_sfpu else [])

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        named_compile_time_args=named_ct_args,
        tensors={"block": ta, "vec": tv, "out": tout},
        defines=[(f"BC_AXIS_{axis.upper()}", "1"), (f"BC_OP_{op.upper()}", "1")]
        + ([("BC_THEN_SFPU", "1")] if then_sfpu else []),
    )
    run_unified_spec(device, spec, {"block": ta, "vec": tv, "out": tout})
    out = tout

    got = ttnn.to_torch(out).to(torch.float32)[0, 0]
    want = reference(op, axis, a.to(torch.float32)[0, 0], vals.to(torch.float32))
    if then_sfpu:
        want = torch.relu(want + want)
    return got, want


def err(got, want):
    return ((got - want).abs().max() / want.abs().max()).item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=0.02)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for axis in ("Rows", "Cols", "Both"):
            for op in ("add", "sub", "mul"):
                got, want = run(device, op, axis)
                e = err(got, want)
                ok = e <= args.tol
                logger.info(f"{op:3s} bcast<Axis::{axis:5s}>  err={e:.5f}  {'ok' if ok else 'FAIL'}")
                if not ok:
                    failed.append((op, axis))

        # A broadcast leaves the unpacker in broadcast mode; an SFPU op afterwards must put
        # it back. Phase 4's per-leaf copy_tile_to_dst_init_short is what should do that.
        for axis in ("Rows", "Cols", "Both"):
            got, want = run(device, "sub", axis, then_sfpu=True)
            e = err(got, want)
            ok = e <= args.tol
            logger.info(f"sub bcast<Axis::{axis:5s}> then SFPU  err={e:.5f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(("sub-then-sfpu", axis))
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
