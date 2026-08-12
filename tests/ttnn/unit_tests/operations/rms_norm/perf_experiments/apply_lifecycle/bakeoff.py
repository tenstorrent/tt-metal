# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Driver for the apply-lifecycle bake-off (idea I11).

Every option is CORRECTNESS-GATED first (PCC + median(got/true) against a torch
reference at the pinned bf16/HiFi2/fp32_dest_acc_en=False contract). A failing option
is reported BROKEN and its timing is not used. Perf is measured, never asserted.
"""

from __future__ import annotations

import traceback

import ttnn

from bench import P_BULK, P_CALLER_STRIDED, P_PERCHUNK, P_PERTILE, check, make_inputs, measure

# label -> (blk, out_policy, normed_policy)
OPTIONS = {
    # The op's current approach, verbatim: two chains, blk=1, PerTile/PerTile on both
    # the internal cb_normed and the output CB.
    "baseline": (1, P_PERTILE, P_PERTILE),
    # (a) LIFECYCLE alone, blk=1. PerChunk at blk=1 is definitionally PerTile, so the
    # lifecycle-only questions are: bulk on the INTERNAL scratch (integration-safe:
    # cb_normed is already sized R*WC and consumed by the next chain in this kernel),
    # and bulk on the OUTPUT (integration-UNSAFE, measured only as the round-1
    # reference point / upper bound).
    "normed_bulk": (1, P_PERTILE, P_BULK),
    "both_bulk": (1, P_BULK, P_BULK),
    # (b) BLOCKING, in the integration-safe PerChunk form on the output.
    "perchunk_blk2": (2, P_PERCHUNK, P_BULK),
    "perchunk_blk4": (4, P_PERCHUNK, P_BULK),
    "perchunk_blk6": (6, P_PERCHUNK, P_BULK),
    "perchunk_blk8": (8, P_PERCHUNK, P_BULK),
    # Fully-safe variant: PerChunk on BOTH CBs (no bulk anywhere, so no CB resizing at
    # all — cb_normed keeps whatever depth it has).
    "perchunk_both_blk4": (4, P_PERCHUNK, P_PERCHUNK),
    "perchunk_both_blk8": (8, P_PERCHUNK, P_PERCHUNK),
    "perchunk_both_blk3": (3, P_PERCHUNK, P_PERCHUNK),
    "perchunk_both_blk2": (2, P_PERCHUNK, P_PERCHUNK),
    # The op's OUT_STRIDED leg (caller-reserved, strided pack): no lifecycle lever,
    # blocking only. `strided_blk1` is that leg's honest baseline.
    "strided_blk1": (1, P_CALLER_STRIDED, P_PERTILE),
    "strided_blk4": (4, P_CALLER_STRIDED, P_PERCHUNK),
    "strided_blk8": (8, P_CALLER_STRIDED, P_PERCHUNK),
    # Upper bound: bulk output + blocking (what round 1 measured; NOT integration-safe).
    "bulk_blk4": (4, P_BULK, P_BULK),
    "bulk_blk8": (8, P_BULK, P_BULK),
}


def main(device, shapes, options, iters=(1, 21), grid=(1, 1)):
    results = {}
    for rows_t, cols in shapes:
        tt, ref, hw = make_inputs(device, rows_t, cols, grid=grid)
        print(f"\n=== (rows_t={rows_t}, cols={cols}) grid={grid} ===", flush=True)
        print(f"{'option':<22} {'ns/chunk':>10} {'T_lo':>8} {'T_hi':>9} {'pcc':>10} {'ratio':>9} {'amax':>9}")
        for label in options:
            blk, outp, normp = OPTIONS[label]
            if blk > cols:
                print(f"{label:<22} {'skip (blk>cols)':>10}")
                continue
            try:
                kw = dict(rows_t=rows_t, cols=cols, blk=blk, out_policy=outp, normed_policy=normp, grid=grid)
                lo, hi = iters
                t_lo, out_lo = measure(device, tt, hw, iters=lo, **kw)
                pcc, ratio, amax = check(out_lo, ref, hw, grid=grid)
                ttnn.deallocate(out_lo)  # each measure allocates a fresh output shard
                t_hi, out_hi = measure(device, tt, hw, iters=hi, **kw)
                ttnn.deallocate(out_hi)
                slope = (t_hi - t_lo) / (hi - lo)
                ok = pcc > 0.999 and abs(ratio - 1.0) < 0.02
                results[(rows_t, cols, label)] = (slope, pcc, ok)
                print(
                    f"{label:<22} {slope:>10.1f} {t_lo:>8.0f} {t_hi:>9.0f} {pcc:>10.6f} {ratio:>9.6f} {amax:>9.4f}"
                    + ("" if ok else "   <-- BROKEN")
                )
            except Exception:
                traceback.print_exc()
                print(f"{label:<22}  EXCEPTION")
    return results
