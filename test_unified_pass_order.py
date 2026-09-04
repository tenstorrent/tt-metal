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

FINDING 2 -- a matmul did not restore its operand descriptors after another pass moved
them -- FIXED. Rows 3 to 6, all with `matmul_init<Row, Blk>` naming the matmul's own shapes.
What they measured before the fix:

    matmul alone                    exact
    bcast, then the matmul          BROKEN     <- was exact before finding 1 was fixed
    SFPU, then the matmul           BROKEN
    matmul, then the SFPU           exact, both

`Strategy<FPUFusion>::run` and `::run_banded` now call
`unpack_geometry_to(node.in1_dfb, node.in0_dfb)` beside the `matmul_block_init` at their
entry. That is EXACTLY the call matmul_init makes for the unpack side, reversed to match:
`compute_kernel_hw_startup<SrcOrder::Reverse>` resolves src_a_cb = icb1 and src_b_cb = icb0
before calling `llk_unpack_hw_configure(src_a_cb, src_b_cb)`, so in1 is srcA and in0 is srcB
(compute_kernel_hw_startup.h:60-64). All sixteen rows are exact.

THIS TABLE CHANGED WHEN FINDING 1 WAS FIXED, and the change is the most useful thing in it.
Row 4 used to be exact, and it was exact only because the bcast programmed nothing. Now that
it programs its own 32x32 descriptors, the matmul behind it fails identically to the matmul
behind the SFPU pass -- 6/16 and 8/16 at wt=1, 0/64 on both faces at wt=4. So this is not
"the SFPU path's reprogram is not seen by the FPU path": ANY pass which programs operand
geometry ahead of a matmul breaks it.

And it is the SAME missing call, not a stale memo. The matmul strategy has two
`unpack_geometry_to(node.in1_dfb, node.in0_dfb)` sites (math.hpp:1797 and :2018) and BOTH sit
behind a bias epilogue or an `if (reload)` accumulator path, so neither runs for a plain
single matmul. The unconditional entry is math.hpp:1961, and its own comment has already
named this defect: it programs the block dimensions per pass rather than trusting matmul_init
because "a broadcast, a reduction or an SFPU pass reconfigures the unpack and math units for
itself, so a matmul that FOLLOWS one -- as attention's second matmul does -- would otherwise
run against another op's state and return garbage". Exactly right, and the call it reaches
for carries the block dimensions and the formats and not the tile descriptors. So a matmul
programmed its block dimensions per pass and its operand geometry never, and was right only
while nothing had moved what matmul_init left. Rows 4 and 5 are the two witnesses, and the
call now sits directly under that comment.

Both fixes were measured the way this repo measures things. The bcast call is free on the
matmul sweep and costs +0.6% on the d64 prefill configs, which is the work. The matching
one-liner in `bias_finish` was backed out instead: +0.44% on every L1-mode matmul cell for a
call the sweep never executes, because `via_bias` is a runtime bool so the whole epilogue is
EMITTED for L1 mode and elided for Dst.

The matmul entry cost +1.16% mean and +7.2% worst as first written, and the shape of that
number is what gave it away: a FLAT 0.14us on every cell, +4.5% on the smallest and +0.1% on
the largest. A one-off, not per-tile work -- `matmul_init` and `compute_init` program the
descriptors at kernel entry and did not tell the memo, so the first pass of every homogeneous
kernel reprogrammed what was already in place. `unpack_geometry_assume` / `pack_geometry_assume`
record it instead, and the cost fell to

    mean +0.30%   worst +3.1% at matmul l1/kb1/kt8/rt1/ct1

against an A/A noise floor of mean -0.04% and worst 1.4%. What remains is the memo compare
itself -- two table loads and two compares per pass boundary -- and it lands on the L1 cells
rather than the Dst ones because L1 mode enters `run` once per k-block plus the finish pass,
so it crosses more boundaries. Folding the geometry word to a literal via a `pack_to<DfbId>()`
style template form is the outstanding way to recover it.

ON CODE SIZE, because the matmul entry is on every matmul and that was the worry. It costs
nothing. Measured on test_unified_flash.py's q-loop (sq=4, nq=4, sk=4) config with
LIGHTWEIGHT_KERNEL_ASSERTS on, which is the largest program either suite builds:

    HEAD                          73488
    the two calls added           73488     <- no change
    ... and marked noinline       74080     <- +592, so they are NOT marked noinline

A `noinline` on the geometry helpers was tried, on the theory that a helper inlined into
every `Strategy<FPUFusion>::run` instantiation was the cost. It is not: LTO already shares
what it can, and forcing the call out of line costs more than it saves. Reverted.

SEPARATELY, AND NOT CAUSED BY ANY OF THIS: those numbers are all over the 70656 kernel config
buffer, so `test_unified_flash.py` cannot launch that config with lightweight asserts enabled
-- at HEAD, 2832 bytes over, before any of this landed. It passes with asserts off, which is
why it went unnoticed, and it fails in `run_unified_tests.sh` because the sweep turns asserts
on. That is its own defect and wants its own fix; the biggest single symbol in that build is
`kernel_main` at 10376 bytes, and each distinct matmul SHAPE costs about 6.5KB across its
IPA-CP clones. Disabling those clones is not the lever -- `-fno-ipa-cp-clone` makes the same
translation unit 41% bigger, because each clone is a specialisation that replaces a larger
general version rather than duplicating it.

Still distinct from the matmul limitation A3 records: that one is two matmul SHAPES in one
body, and rows 4 and 5 have a single matmul with a non-matmul pass in front.

ROWS 7 AND 8 WERE THE TRAP, and they are the reason this test runs every row at TWO tile
counts rather than one. A mid-body re-init -- `ckernel::init_sfpu` naming the operands the
matmul was about to use -- made the matmul come back exact at wt=1 and did NOT fix it at
wt=4:

    SFPU, re-init, then matmul    wt=1  face0=16/16 face1=16/16     <- looked fixed
    SFPU, re-init, then matmul    wt=4  face0=29/64 face1=35/64     <- was not

Both are exact at both counts now, and the re-init in them is REDUNDANT rather than
load-bearing: the matmul entry programs its own operands, so there is nothing left for a
hand re-init to repair.

wt is the matmul's kt_dim, so wt=1 is one k-step and wt=4 walks four, and a k-step re-reads
srcB. `init_sfpu(in, out)` forwards to `unpack_hw_configure(in, in)`, which programs srcA
AND srcB from one buffer -- fine for a copy or an SFPU pass, wrong for a matmul whose two
operands have different geometry, and only VISIBLE once srcB is read more than once. A
wt=1-only test reports this as a working workaround.

The `pack_to_forget()` that mixed_geometry's MG_REINIT pairs it with makes no difference in
either direction, which rows 7 and 8 exist to say.

There was no kernel-level workaround, which is why it was fixed inside the FPU path: the
one-operand re-init is the only form `u::` exposes, `matmul_init` cannot be re-run
(compute_kernel_hw_startup is MMIO plus a pack-sync init and the second call hangs), and a
raw two-operand `llk_unpack_hw_configure(a, b)` from a kernel body is not something to
ship.

Note for anyone reading A3's blaze bisect: an attempt there read 0.599 -> 0.396 and was
recorded as "no workaround", then re-measured as working. Both readings were partial -- the
first measured a body with a trailing copy() after the matmul, the second a single-tile
matmul. The two tile counts here are what settle it.

EVERY ROW MUST NOW BE EXACT. This suite pinned today's behaviour while the two defects were
live, and `--expect-fixed` was the switch to flip when they were fixed; both are fixed, so
the pins are gone and the flag with them. A wrong number here is a regression, and the row it
appears on says which half broke -- which is the whole reason the bodies were kept after they
stopped disagreeing.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_pass_order.py     # eight bodies, each at two tile counts
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
    # name                                defines                        stores sq / row
    # ---- Finding 1: an all-32x32 bcast, and the init is the only variable ----
    ("bcast alone, init on 32x32", ["PO_BCAST"], True, None),
    # This is the finding-1 body. It was WRONG at both tile counts: an all-32x32 broadcast
    # whose answer depended on what the init named.
    ("bcast alone, init on row", ["PO_BCAST", "PO_ROW_INIT"], True, None),
    # ---- Finding 2: the matmul names its own shapes throughout ----
    ("matmul alone", [], None, True),
    # These two are the finding-2 bodies, and they broke in opposite eras. "SFPU, then
    # matmul" was broken from the start. "bcast, then matmul" was EXACT until finding 1 was
    # fixed and broke the moment the bcast started programming its own descriptors -- it was
    # never a control, only a pass that happened to disturb nothing.
    ("bcast, then matmul", ["PO_BCAST_FIRST"], True, True),
    ("SFPU, then matmul", ["PO_SFPU_FIRST"], True, True),
    ("matmul, then SFPU", ["PO_SFPU_LAST"], True, True),
    # ---- The mid-body re-init, which LOOKED like a repair at wt=1 and was not one ----
    # Both were 29/64 and 35/64 at wt=4 while reading a clean 16/16 at wt=1. They are exact
    # at both counts now, and REDUNDANT rather than load-bearing: the matmul entry programs
    # its own operands, so nothing is left for a hand re-init to repair. Kept because a
    # regression would show here first, and because the wt=1 / wt=4 disagreement is the
    # reason this suite runs every body at two tile counts.
    ("SFPU, re-init, then matmul", ["PO_SFPU_FIRST", "PO_REINIT"], True, True),
    ("SFPU, re-init w/o forget, matmul", ["PO_SFPU_FIRST", "PO_REINIT_NO_FORGET"], True, True),
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
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        measured = [[measure(device, c[1], nt, args.rel_err) for nt in TILE_COUNTS] for c in VARIANTS]
    finally:
        ttnn.close_device(device)

    failed = []
    for variant, per_nt in zip(VARIANTS, measured):
        name, defines, stores_sq, stores_row = variant
        for nt, (sq, face0, face1, per_face) in zip(TILE_COUNTS, per_nt):
            row_ok = face0 == per_face and face1 == per_face
            logger.info(
                f"{name:33s} wt={nt}  "
                f"blk={('exact' if sq else 'WRONG') if stores_sq else 'n/a':5s}  "
                f"row: {f'face0={face0}/{per_face} face1={face1}/{per_face}' if stores_row else 'n/a'}"
            )
            # Every body that stores a buffer must be exact in it. There is no per-body
            # expectation left to carry: the two findings this suite was written for are
            # both fixed, so a wrong number here is a REGRESSION rather than a pin.
            if (stores_sq and not sq) or (stores_row and not row_ok):
                failed.append(f"{name} (wt={nt})")

    if failed:
        logger.error(f"FAIL: {failed}")
        logger.error(
            "a wrong 32x32 store on a bcast body means Strategy<BcastFusion> is not "
            "programming its operands' descriptors. A wrong row store on a body with a "
            "pass in front of the matmul means the matmul entry is not putting its own "
            "descriptors back -- and if that shows only at wt=4, whatever programmed them "
            "did it from ONE buffer, so srcB is wrong from the second k-step on. See "
            "unified_blaze_integration_spec.md A3."
        )
        return 1
    logger.info("PASS (all exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
