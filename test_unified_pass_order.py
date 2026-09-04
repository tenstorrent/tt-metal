# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tile geometry across a SEQUENCE of passes, with the init held right.

`test_unified_mixed_geometry.py` varies which geometry the init names for a body of one or
two passes. This varies which passes run and in what order, with the init always naming the
shapes of the pass under test -- so the pass under test is never the one the init got wrong,
and whatever breaks it is the pass beside it.

WHY IT EXISTS. blaze's u_flash_kda is still wrong at the row form after `f1d762a3707` and
`eb17b41d7fd`: s_out PCC 0.675, o 0.830. Its body is

    bcast -> matmul -> SFPU -> matmul -> matmul

longer than anything in unified_kernels/. Bisecting it stage by stage through blaze pointed
at two things neither commit covers. Both are here with no blaze in the picture, so they can
be worked from this repo. See unified_blaze_integration_spec.md A3.

FINDING 1 -- the broadcast path never programmed operand geometry -- FIXED. Rows 1 and 2:
one bcast pass whose operands are ALL 32x32, with the init pointed at the 32x32 pair and then
at the row pair. Nothing in the body touches a row-form buffer, so the result could not
legitimately depend on that, and it did: WRONG at both tile counts, 0.958560 through blaze.
`Strategy<BcastFusion>` carried `reconfig_data_format` and the broadcast mode and no
descriptor call, so the pass ran on whatever the init left. It now calls
`unpack_geometry_to(block, vec)` beside the reconfig, and rows 1 and 2 are exact.

FINDING 2 -- a matmul does not restore its operand descriptors after another pass moves
them. Rows 3 to 6, all with `matmul_init<Row, Blk>` naming the matmul's own shapes:

    matmul alone                    exact
    bcast, then the matmul          BROKEN     <- was exact before finding 1 was fixed
    SFPU, then the matmul           BROKEN
    matmul, then the SFPU           exact, both

THIS TABLE CHANGED WHEN FINDING 1 WAS FIXED, and the change is the most useful thing in it.
Row 4 used to be exact, and it was exact only because the bcast programmed nothing. Now that
it programs its own 32x32 descriptors, the matmul behind it fails identically to the matmul
behind the SFPU pass -- 6/16 and 8/16 at wt=1, 0/64 on both faces at wt=4. So this is not
"the SFPU path's reprogram is not seen by the FPU path": ANY pass which programs operand
geometry ahead of a matmul breaks it.

And it is the SAME missing call, not a stale memo. The matmul strategy has two
`unpack_geometry_to(node.in1_dfb, node.in0_dfb)` sites (math.hpp:1789 and :2010) and BOTH sit
behind a bias epilogue or an `if (reload)` accumulator path, so neither runs for a plain
single matmul. The unconditional entry -- math.hpp:1953, whose own comment says it programs
the block dimensions "rather than trusting matmul_init to still be in effect" -- calls
`matmul_block_init` with no descriptor call beside it. So a matmul programs its block
dimensions per pass and its operand geometry never, and it is right only while nothing has
moved what matmul_init left. That is the next thing to fix, and rows 4 and 5 are the two
witnesses for it.

Still distinct from the matmul limitation A3 records: that one is two matmul SHAPES in one
body, and rows 4 and 5 have a single matmul with a non-matmul pass in front.

ROWS 7 AND 8 ARE THE TRAP, and they are the reason this test runs every row at TWO tile
counts rather than one. A mid-body re-init -- `ckernel::init_sfpu` naming the operands the
matmul is about to use -- makes the matmul come back exact at wt=1 and does NOT fix it at
wt=4:

    SFPU, re-init, then matmul    wt=1  face0=16/16 face1=16/16     <- looks fixed
    SFPU, re-init, then matmul    wt=4  face0=29/64 face1=35/64     <- is not

wt is the matmul's kt_dim, so wt=1 is one k-step and wt=4 walks four, and a k-step re-reads
srcB. `init_sfpu(in, out)` forwards to `unpack_hw_configure(in, in)`, which programs srcA
AND srcB from one buffer -- fine for a copy or an SFPU pass, wrong for a matmul whose two
operands have different geometry, and only VISIBLE once srcB is read more than once. A
wt=1-only test reports this as a working workaround.

The `pack_to_forget()` that mixed_geometry's MG_REINIT pairs it with makes no difference in
either direction, which rows 7 and 8 exist to say.

So there is no kernel-level workaround: the one-operand re-init is the only form `u::`
exposes, `matmul_init` cannot be re-run (compute_kernel_hw_startup is MMIO plus a pack-sync
init and the second call hangs), and a raw two-operand `llk_unpack_hw_configure(a, b)` from
a kernel body is not something to ship. It has to be fixed inside the FPU path.

Note for anyone reading A3's blaze bisect: an attempt there read 0.599 -> 0.396 and was
recorded as "no workaround", then re-measured as working. Both readings were partial -- the
first measured a body with a trailing copy() after the matmul, the second a single-tile
matmul. The two tile counts here are what settle it.

THE EXPECTATIONS PIN TODAY'S BEHAVIOUR, so this suite is GREEN while the defect is present
and says loudly which rows are wrong. `--expect-fixed` is the switch to flip when the fix
lands; it asserts every row exact, which is what the fix has to deliver.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_pass_order.py                  # all eight rows
    python test_unified_pass_order.py --expect-fixed    # assert them all exact
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/pass_order_geometry.cpp"
TILE = 32
ROW_TILE = ttnn.Tile([1, TILE])
FACE = 16  # a row-form tile is two faces of 1x16; face 1 is the one that goes missing

# name, defines, and what each buffer is EXPECTED to be today. `None` means the body does
# not store that buffer, so it is not a measurement.
#
#   sq   the 32x32 output: exact / wrong
#   row  the row-form output, checked face by face -- a whole-row PCC reads 0.6-0.8 with
#        half the values exactly wrong, which is a number people explain away
# Every row runs at BOTH tile counts. wt is the matmul's kt_dim, so wt=1 is a single
# k-step and wt=4 walks four -- and a k-step re-reads srcB. That distinction decides
# whether the mid-body re-init is a workaround or a trap, and a wt=1-only test says the
# wrong thing about it: see the note on rows 7 and 8.
TILE_COUNTS = [1, 4]

VARIANTS = [
    # ---- Finding 1: an all-32x32 bcast, and the init is the only variable ----
    ("bcast alone, init on 32x32", ["PO_BCAST"], True, None, True, None),
    # Exact since the bcast strategy programs its own descriptors. It was WRONG at both tile
    # counts before that, which was finding 1: an all-32x32 body whose answer depended on
    # what the init named.
    ("bcast alone, init on row", ["PO_BCAST", "PO_ROW_INIT"], True, None, True, None),
    # ---- Finding 2: the matmul names its own shapes throughout ----
    ("matmul alone", [], None, True, None, True),
    # Both halves flipped when finding 1 was fixed, and the row half flipped the WRONG way.
    # The 32x32 store is exact now: it programs its own descriptors rather than inheriting
    # the row geometry matmul_init named. And the matmul after it is now broken exactly as
    # it is after the SFPU pass below -- because the bcast used to disturb nothing only by
    # virtue of programming nothing. So finding 2 is not about the SFPU path: ANY pass that
    # programs operand geometry ahead of a matmul breaks it, and fixing finding 1 is what
    # made that visible.
    ("bcast, then matmul", ["PO_BCAST_FIRST"], True, False, True, False),
    ("SFPU, then matmul", ["PO_SFPU_FIRST"], True, False, True, False),
    ("matmul, then SFPU", ["PO_SFPU_LAST"], True, True, True, True),
    # ---- The mid-body re-init, which LOOKS like a repair at wt=1 and is not one ----
    ("SFPU, re-init, then matmul", ["PO_SFPU_FIRST", "PO_REINIT"], True, True, True, False),
    ("SFPU, re-init w/o forget, matmul", ["PO_SFPU_FIRST", "PO_REINIT_NO_FORGET"], True, True, True, False),
]


def run(device, defines, nt, seed=0):
    torch.manual_seed(seed)
    H = W = nt * TILE
    # Strictly positive and away from zero: recip is ill-conditioned near it, and a matmul
    # of positives cannot cancel a lost face into looking right.
    blk = (0.5 + 1.5 * torch.rand([1, 1, H, W])).to(torch.bfloat16)
    # The broadcast vector is one column of TILES; only column 0 of each is read.
    vec = (0.5 + 1.5 * torch.rand([1, 1, H, TILE])).to(torch.bfloat16)
    row = (0.5 + 1.5 * torch.rand([1, 1, 1, W])).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def dev(t, tile=None):
        kw = {"tile": tile} if tile is not None else {}
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram, **kw)

    t_in, t_vec = dev(blk), dev(vec)
    t_out = dev(torch.zeros([1, 1, H, W], dtype=torch.bfloat16))
    # The row pair carries a 1x32 TILE, which is what makes its pages 64 bytes rather than
    # 2048 and what the kernel's Tiled<Tile<1, 32>, ...> has to agree with.
    t_in_row = dev(row, ROW_TILE)
    t_out_row = dev(torch.zeros([1, 1, 1, W], dtype=torch.bfloat16), ROW_TILE)

    core_ranges, _ = single_core()
    tensors = {"in": t_in, "vec": t_vec, "out": t_out, "in_row": t_in_row, "out_row": t_out_row}
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=[
            dfb("in", nt * nt),
            dfb("vec", nt),
            dfb("out", nt * nt),
            dfb("in_row", nt, tile=ROW_TILE),
            dfb("out_row", nt, tile=ROW_TILE),
        ],
        named_compile_time_args=[("ht", nt), ("wt", nt)],
        tensors=tensors,
        defines=[(d, "1") for d in defines] or None,
        name=f"pass_order_{nt}" + ("_" + "_".join(d.lower()[3:] for d in defines) if defines else "_mm"),
    )
    run_unified_spec(device, spec, tensors)

    b, v, r = (x.to(torch.float32) for x in (blk, vec, row))
    # The bcast is column 0 of each of the vector's TILES, spread across the block's
    # columns; the SFPU pass is recip. Which of the two wrote `out` depends on the body.
    col = v[0, 0, :, 0].reshape(1, 1, H, 1)
    want_bcast = b * col
    want_recip = torch.reciprocal(b)
    want_row = (r.reshape(1, W) @ b[0, 0]).flatten()

    got_sq = ttnn.to_torch(t_out).to(torch.float32)
    got_row = ttnn.to_torch(t_out_row).to(torch.float32).flatten()
    want_sq = want_bcast if ("PO_BCAST" in defines or "PO_BCAST_FIRST" in defines) else want_recip
    return (
        (got_sq - want_sq).abs() / want_sq.abs().clamp(min=1e-6),
        (got_row - want_row).abs() / want_row.abs().clamp(min=1e-6),
    )


def measure(device, defines, nt, tol):
    """Returns (square exact?, face-0 correct, face-1 correct, per-face total).

    Counted PER TILE and summed: the row output is `nt` tiles of 1x32, each two faces of
    1x16, so face 1 of tile 2 is elements 80..95 and not something at the end. Slicing the
    flat row in half instead reports face1=112/16 at nt=4, which is not a number.
    """
    err_sq, err_row = run(device, defines, nt)
    ok = err_row <= tol
    f0 = sum(int(ok[i * TILE : i * TILE + FACE].sum()) for i in range(nt))
    f1 = sum(int(ok[i * TILE + FACE : (i + 1) * TILE].sum()) for i in range(nt))
    return bool((err_sq <= tol).all()), f0, f1, FACE * nt


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rel-err", type=float, default=0.02, help="max elementwise relative error")
    p.add_argument(
        "--expect-fixed",
        action="store_true",
        help="assert every row exact, instead of pinning today's behaviour",
    )
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        measured = [[measure(device, c[1], nt, args.rel_err) for nt in TILE_COUNTS] for c in VARIANTS]
    finally:
        ttnn.close_device(device)

    failed = []
    for variant, per_nt in zip(VARIANTS, measured):
        name, defines = variant[0], variant[1]
        for nt, (sq, face0, face1, per_face) in zip(TILE_COUNTS, per_nt):
            # Expectations are per tile count: index 2/3 are wt=1, 4/5 are wt=4 where the
            # variant gives them, else the wt=1 pair applies to both.
            want_sq, want_row = (variant[2], variant[3]) if nt == 1 or len(variant) < 6 else (variant[4], variant[5])
            row_ok = face0 == per_face and face1 == per_face
            logger.info(
                f"{name:33s} wt={nt}  "
                f"blk={('exact' if sq else 'WRONG') if want_sq is not None else 'n/a':5s}  "
                f"row: {f'face0={face0}/{per_face} face1={face1}/{per_face}' if want_row is not None else 'n/a'}"
            )
            # None stays None under --expect-fixed: the body does not store that buffer, so
            # there is nothing there to be right.
            exp_sq = (True if want_sq is not None else None) if args.expect_fixed else want_sq
            exp_row = (True if want_row is not None else None) if args.expect_fixed else want_row
            if (exp_sq is not None and sq != exp_sq) or (exp_row is not None and row_ok != exp_row):
                failed.append(f"{name} (wt={nt})")

    if failed:
        logger.error(f"FAIL: {failed}")
        logger.error(
            "'bcast alone, init on row' wrong means the broadcast path is not programming "
            "its operands' geometry and inherits the init's. 'SFPU, then matmul' wrong "
            "means the FPU path's geometry memo was not invalidated when the SFPU path "
            "reprogrammed the same descriptor. See unified_blaze_integration_spec.md A3."
        )
        return 1
    logger.info("PASS" + (" (all exact)" if args.expect_fixed else " (defect pinned as expected)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
