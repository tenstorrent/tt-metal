# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two tile geometries in one body: the first in-tree coverage of the row form.

Nothing else in `unified_kernels/` uses a sub-tile geometry at all -- no kernel spells
`Tiled<>` and no launcher passes `dfb(..., tile=)` -- so the row-form support that
unified_blaze_integration_spec.md B3 records as FIXED has had no regression test here. Its
measurements were taken through throwaway probes and through blaze, neither of which is in
this repo.

WHAT IT COVERS. Tile geometry reaches the hardware through hw_configure, which the kernel's
init runs once for the ONE operand pair it names; the per-pass calls carry formats and the op
and never carried geometry. So a pass over an operand of another geometry read and wrote
through the init's descriptor. `unpack_geometry_to` / `pack_geometry_to` in math.hpp
reprogram the descriptors per pass, and this is what holds them to it.

WHAT IT LOOKED LIKE BEFORE THE FIX, because the numbers are the diagnosis:

    init on 32x32, both passes         32x32 1024/1024   row face1  0/16
    init on 32x32, row pass alone            n/a         row face1  0/16
    init on row, row pass alone              n/a         row face1 16/16
    init on row, both passes           32x32  277/1024   row face1 16/16
    init on 32x32, re-init mid-body    32x32 1024/1024   row face1 16/16

Whichever geometry the init named came back right, in BOTH directions -- row 4 is the same
defect pointing the other way. Row 2 is what ruled out A3's first diagnosis (that `pack_to`
failed to carry geometry between stores): one store, no transition, the same lost face. A
1x32 tile is two faces of 1x16, so a lost face 1 is half the row.

Checked face by face, not by PCC. A whole-row PCC reads 0.6-0.8 with half the values exactly
wrong, and that is a number people explain away; "face 1: 0/16" is not.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_mixed_geometry.py             # the primary body
    python test_unified_mixed_geometry.py --matrix    # all five, the ones that localised it
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
# All five are exact now, which is the point: the fix does not depend on which geometry the
# init happened to name, and the mid-body re-init in the last body is redundant rather than
# load-bearing. The docstring records what each of them measured BEFORE the fix, which is
# where the diagnosis came from -- keep the bodies even though they now agree, because a
# regression would show up in exactly one of them and the row tells you which half broke.
VARIANTS = [
    # name                                  row_only row_init reinit   full  face0 face1
    ("init on 32x32, both passes", False, False, False, True, True, True),
    ("init on 32x32, row pass alone", True, False, False, None, True, True),
    ("init on row, row pass alone", True, True, False, None, True, True),
    ("init on row, both passes", False, True, False, True, True, True),
    ("init on 32x32, re-init for row", False, False, True, True, True, True),
    # The FPU path, whose operands are configured in fpu_seed_init rather than in the leaf's
    # emit. Same two geometries, same expectation; a + a rather than recip so the pass stays
    # on the FPU.
    ("FPU both passes", False, False, False, True, True, True, "fpu"),
    # The matmul path: a 32x32 product, then B3's row-form LHS, whose operands are a MIXED
    # pair under matmul's reversed order. All-ones inputs, so a 32-deep product is exactly
    # 32 in bfloat16 and a lost face reads as 0 rather than as a tolerance question.
    # The matmul path. One shape per body, because two matmul shapes in one body is a
    # separate limitation -- see the kernel, and A3.
    ("matmul, row-form", True, False, False, None, True, True, "matmul"),
]


def run(device, row_only=False, row_init=False, reinit=False, fpu=False, matmul=False, seed=0):
    torch.manual_seed(seed)
    # Strictly positive and away from zero: recip is ill-conditioned near it.
    if matmul:
        full = torch.ones([1, 1, TILE, TILE], dtype=torch.bfloat16)
        row = torch.ones([1, 1, 1, TILE], dtype=torch.bfloat16)
    else:
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
        + ([("MG_FPU", "1")] if fpu else [])
        + ([("MG_MATMUL", "1")] if matmul else [])
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
        name=f"mixed_geometry{'_ro' if row_only else ''}{'_ri' if row_init else ''}{'_re' if reinit else ''}{'_fpu' if fpu else ''}{'_mm' if matmul else ''}",
    )
    run_unified_spec(device, spec, tensors)

    got_full = ttnn.to_torch(t_out).to(torch.float32)
    got_row = ttnn.to_torch(t_out_row).to(torch.float32).flatten()
    if matmul:
        f = full.to(torch.float32)[0, 0]
        want_full = (f @ f).reshape(1, 1, TILE, TILE)
        want_row = (row.to(torch.float32).reshape(1, TILE) @ f).flatten()
    else:
        ref = (lambda x: x + x) if fpu else torch.reciprocal
        want_full = ref(full.to(torch.float32))
        want_row = ref(row.to(torch.float32)).flatten()
    err_full = (got_full - want_full).abs() / want_full.abs().clamp(min=1e-6)
    err_row = (got_row - want_row).abs() / want_row.abs().clamp(min=1e-6)
    return err_full, err_row


def measure(device, row_only, row_init, reinit, tol, fpu=False, matmul=False):
    err_full, err_row = run(device, row_only=row_only, row_init=row_init, reinit=reinit, fpu=fpu, matmul=matmul)
    ok_row = err_row <= tol
    full = None if row_only else bool((err_full <= tol).all())
    return full, int(ok_row[:FACE].sum()), int(ok_row[FACE:].sum())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rel-err", type=float, default=0.02, help="max elementwise relative error")
    p.add_argument("--matrix", action="store_true", help="run all five bodies against the recorded table")
    args = p.parse_args(argv)

    cases = VARIANTS if args.matrix else VARIANTS[:1]
    device = ttnn.open_device(device_id=0)
    try:
        measured = [
            measure(
                device, c[1], c[2], c[3], args.rel_err,
                fpu=(len(c) > 7 and c[7] == "fpu"),
                matmul=(len(c) > 7 and c[7] == "matmul"),
            )
            for c in cases
        ]
    finally:
        ttnn.close_device(device)

    failed = []
    for case, (full, face0, face1) in zip(cases, measured):
        name, row_only = case[0], case[1]
        want = tuple(case[4:7])
        shown = "n/a" if full is None else ("exact" if full else "WRONG")
        logger.info(f"{name:34s} 32x32={shown:5s}  face0={face0}/{FACE}  face1={face1}/{FACE}")
        if (full, face0 == FACE, face1 == FACE) != want:
            failed.append(name)

    if failed:
        logger.error(f"FAIL: {failed}")
        logger.error(
            "a lost face on the row store means the per-pass descriptor reprogramming in "
            "math.hpp (unpack_geometry_to / pack_geometry_to) is not reaching this path; a "
            "lost 32x32 store means it reprogrammed and did not put it back"
        )
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
