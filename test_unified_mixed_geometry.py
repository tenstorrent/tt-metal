# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two tile geometries in one body: the first in-tree coverage of the row form.

Nothing else in `unified_kernels/` uses a sub-tile geometry at all -- no kernel spells
`Tiled<>` and no launcher passes `dfb(..., tile=)` -- so the row-form support that
unified_blaze_integration_spec.md B3 records as FIXED has had no regression test here. Its
measurements were taken through throwaway probes and through blaze, neither of which is in
this repo.

WHAT THIS PINS. Tile geometry is programmed ONLY by the kernel's init, for the unpacker and
the packer together, and the operand whose geometry the init did not name comes back wrong.
With the init on the 32x32 pair, the row store loses face 1 entirely -- a 1x32 tile is two
faces of 1x16, so that is half the row. See A3, which this test CORRECTS: the defect is not
`pack_to` failing to carry geometry between stores. A body containing only the row store
fails identically, and reprogramming the packer does not repair it.

THIS TEST PASSES WHILE THE DEFECT IS PRESENT, which needs saying out loud. It asserts the
CURRENT behaviour, so it holds the finding still rather than turning the suite red over a
defect nobody has fixed yet. Run it with --expect-fixed once a fix lands: that asserts the
numbers are right instead, and is the switch to flip when the per-pass geometry re-init A3
now asks for exists. Either way it fails if the behaviour CHANGES, which is the point of
pinning it.

Checked face by face, not by PCC. A whole-row PCC reads 0.6-0.8 with half the values exactly
wrong, and that is a number people explain away; "face 1: 0/16" is not.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_mixed_geometry.py             # pins the defect
    python test_unified_mixed_geometry.py --matrix    # the five bodies that localised it
    python test_unified_mixed_geometry.py --expect-fixed
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/mixed_geometry.cpp"
TILE = 32
ROW_TILE = ttnn.Tile([1, TILE])
FACE = 16  # a row-form tile is two faces of 1x16, and face 1 is the one that goes missing

# The five bodies that localised this, with what each MEASURED on a Wormhole n150. `full` is
# None where that body does not run the 32x32 store at all.
#
#   row_only   drop the 32x32 pass, so there is no geometry transition left to blame
#   row_init   point the kernel's init at the ROW pair instead of the 32x32 one
#   reinit     re-init the SFPU for the row pair mid-body, through the raw API
#
# Read it downwards: whichever geometry the init names is the one that comes back right, in
# BOTH directions, and a mid-body re-init repairs both. Row 2 is what rules out A3's stated
# mechanism -- one store, no transition, same lost face.
VARIANTS = [
    # name                                  row_only row_init reinit   full  face0 face1
    ("init on 32x32, both passes", False, False, False, True, True, False),
    ("init on 32x32, row pass alone", True, False, False, None, True, False),
    ("init on row, row pass alone", True, True, False, None, True, True),
    ("init on row, both passes", False, True, False, False, True, True),
    ("init on 32x32, re-init for row", False, False, True, True, True, True),
]


def run(device, row_only=False, row_init=False, reinit=False, seed=0):
    torch.manual_seed(seed)
    # Strictly positive and away from zero: recip is ill-conditioned near it.
    full = (0.5 + 1.5 * torch.rand([1, 1, TILE, TILE])).to(torch.bfloat16)
    row = (0.5 + 1.5 * torch.rand([1, 1, 1, TILE])).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    t_in = ttnn.from_torch(full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    t_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )
    # The row pair carries a 1x32 TILE, which is what makes its pages 64 bytes rather than
    # 2048 and what the kernel's Tiled<Tile<1, 32>, ...> has to agree with. The host
    # round-trip of such a tensor is exact on BOTH faces -- checked separately -- so a lost
    # face here is the kernel's and not ttnn's.
    t_in_row = ttnn.from_torch(
        row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram, tile=ROW_TILE
    )
    t_out_row = ttnn.from_torch(
        torch.zeros([1, 1, 1, TILE], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
        tile=ROW_TILE,
    )

    core_ranges, _ = single_core()
    tensors = {"in": t_in, "out": t_out, "in_row": t_in_row, "out_row": t_out_row}
    defines = (
        ([("MG_ROW_ONLY", "1")] if row_only else [])
        + ([("MG_ROW_INIT", "1")] if row_init else [])
        + ([("MG_REINIT", "1")] if reinit else [])
    )

    # Roles come off the kernel's declarations: in/in_row are filled by DM thread 0, out and
    # out_row drained by thread 1. The two row buffers state their tile, and the endpoint's
    # own static_assert holds the kernel's Shape to it.
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=[
            dfb("in", 2),
            dfb("out", 2),
            dfb("in_row", 2, tile=ROW_TILE),
            dfb("out_row", 2, tile=ROW_TILE),
        ],
        tensors=tensors,
        defines=defines or None,
        name=f"mixed_geometry{'_ro' if row_only else ''}{'_ri' if row_init else ''}{'_re' if reinit else ''}",
    )
    run_unified_spec(device, spec, tensors)

    got_full = ttnn.to_torch(t_out).to(torch.float32)
    got_row = ttnn.to_torch(t_out_row).to(torch.float32).flatten()
    want_full = torch.reciprocal(full.to(torch.float32))
    want_row = torch.reciprocal(row.to(torch.float32)).flatten()
    err_full = (got_full - want_full).abs() / want_full.abs().clamp(min=1e-6)
    err_row = (got_row - want_row).abs() / want_row.abs().clamp(min=1e-6)
    return err_full, err_row


def measure(device, row_only, row_init, reinit, tol):
    err_full, err_row = run(device, row_only=row_only, row_init=row_init, reinit=reinit)
    ok_row = err_row <= tol
    full = None if row_only else bool((err_full <= tol).all())
    return full, int(ok_row[:FACE].sum()), int(ok_row[FACE:].sum())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rel-err", type=float, default=0.02, help="max elementwise relative error")
    p.add_argument("--matrix", action="store_true", help="run all five bodies against the recorded table")
    p.add_argument("--expect-fixed", action="store_true", help="assert the numbers are RIGHT (post-fix)")
    args = p.parse_args(argv)

    cases = VARIANTS if args.matrix else VARIANTS[:1]
    device = ttnn.open_device(device_id=0)
    try:
        measured = [measure(device, c[1], c[2], c[3], args.rel_err) for c in cases]
    finally:
        ttnn.close_device(device)

    failed = []
    for case, (full, face0, face1) in zip(cases, measured):
        name, row_only = case[0], case[1]
        want = (None if row_only else True, True, True) if args.expect_fixed else tuple(case[4:])
        shown = "n/a" if full is None else ("exact" if full else "WRONG")
        logger.info(f"{name:34s} 32x32={shown:5s}  face0={face0}/{FACE}  face1={face1}/{FACE}")
        if (full, face0 == FACE, face1 == FACE) != want:
            failed.append(name)

    if failed:
        logger.error(f"FAIL: {failed}")
        if args.expect_fixed:
            logger.error("the per-pass geometry re-init A3 asks for is absent, or is not sufficient")
        else:
            logger.error("behaviour CHANGED from what this test pins -- re-run --matrix and update A3")
        return 1

    if args.expect_fixed:
        logger.info("PASS")
    else:
        logger.warning(
            "PASS -- and this is a test that passes while the defect is PRESENT: the row store "
            "loses face 1, half the row, whenever the init named the other geometry. See "
            "unified_blaze_integration_spec.md A3, and use --expect-fixed once it is fixed."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
