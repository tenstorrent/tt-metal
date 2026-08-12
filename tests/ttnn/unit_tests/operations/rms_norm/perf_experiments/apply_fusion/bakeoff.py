# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Driver for the apply-fusion bake-off. One device session per invocation.

Every option is CORRECTNESS-GATED first (PCC + median(got/true) against a torch
reference at the pinned precision); a variant that fails the gate is reported as
BROKEN and its timing is not used. Perf is measured, never asserted.

Usage (from a tt-probe script):

    import sys; sys.path.insert(0, "<this dir>")
    from bakeoff import main
    main(shapes=[(1, 3)], options=["baseline", "fused_rstd"], iters=(1, 21))
"""

from __future__ import annotations

import traceback

import ttnn

from apply_bench import check, make_inputs, measure

# label -> (variant, blk, out_bulk, reconfig_mode)
#   reconfig_mode: 1 = data-format reconfig Enabled on every operand (what the op does
#   today), 2 = Enabled on the FIRST stage only, 0 = off everywhere.
OPTIONS = {
    # The op's current approach, verbatim: two chains, blk=1, the per-tile output
    # streaming quantum, reconfig Enabled on both passes.
    "baseline": ("baseline", 1, False, 1),
    # Same two chains, bulk output lifecycle (isolates the per-tile push cost).
    "baseline_bulk": ("baseline", 1, True, 1),
    # Same two chains, reconfig dropped on the second pass / everywhere.
    "baseline_rc2off": ("baseline", 1, False, 2),
    "baseline_noreconfig": ("baseline", 1, False, 0),
    # Same two chains with DEST-lane blocking.
    "baseline_blk4": ("baseline", 4, True, 1),
    "baseline_blk8": ("baseline", 8, True, 1),
    "baseline_blk8_rc2off": ("baseline", 8, True, 2),
    "baseline_blk6": ("baseline", 6, True, 1),
    "baseline_blk2": ("baseline", 2, True, 1),
    # ONE chain / one DEST window / one pack, rstd expanded (ROWS_T tile-ops of prep).
    "fused_rstd": ("fused_rstd", 1, False, 1),
    "fused_rstd_blk4": ("fused_rstd", 4, True, 1),
    "fused_rstd_blk8": ("fused_rstd", 8, True, 1),
    "fused_rstd_blk8_rc2off": ("fused_rstd", 8, True, 2),
    "fused_rstd_noreconfig": ("fused_rstd", 1, False, 0),
    # DEST-lane / block-iteration diagnostic ladder for the dest-reuse correctness bug.
    "fused_rstd_blk2": ("fused_rstd", 2, True, 1),
    "fused_rstd_bulk": ("fused_rstd", 1, True, 1),
    "fused_rstd_blk3": ("fused_rstd", 3, True, 1),
    "fused_rstd_blk5": ("fused_rstd", 5, True, 1),
    "fused_rstd_blk6": ("fused_rstd", 6, True, 1),
    # NOTE (measured, device HANG): blk > 1 must NOT be paired with the PerTile output
    # policy. eltwise_chain's elem_apply_pack reserves ONE page per block iteration
    # (eltwise_chain.inl:2785) but packs `inner_count` tiles, so the CB overruns and the
    # kernel deadlocks. The chain does not static_assert this pairing. Every blocked
    # option below therefore uses the bulk (Upfront/AtEnd) output lifecycle.
    # Same fusion, GAMMA expanded instead (COLS tile-ops of prep).
    "fused_gamma": ("fused_gamma", 1, False, 1),
    "fused_gamma_blk8": ("fused_gamma", 8, True, 1),
    "fused_gamma_blk6": ("fused_gamma", 6, True, 1),
    # ONE chain, no expansion: gamma broadcast into a second DEST lane, SFPU combine.
    "fused_sfpu": ("fused_sfpu", 1, False, 1),
    "fused_sfpu_blk4": ("fused_sfpu", 4, True, 1),
    # rstd folded into gamma (one-tile-row blocks only).
    "fold_gamma": ("fold_gamma", 1, False, 1),
    "fold_gamma_blk8": ("fold_gamma", 8, True, 1),
    "fold_gamma_blk6": ("fold_gamma", 6, True, 1),
}


def main(shapes, options, iters=(1, 21), grid=(1, 1), pcc_floor=0.999, ratio_tol=0.02, dtype="bf16"):
    device = ttnn.open_device(device_id=0)
    try:
        for rows_t, cols in shapes:
            tt, ref, hw = make_inputs(device, rows_t, cols, grid=grid, dtype=dtype)
            print(
                f"\n=== SHAPE rows_t={rows_t} cols={cols} grid={grid} dtype={dtype} (block = {rows_t * cols} tiles) ==="
            )
            for label in options:
                variant, blk, out_bulk, reconfig = OPTIONS[label]
                if variant == "fold_gamma" and rows_t != 1:
                    print(f"RESULT {dtype} {rows_t}x{cols} {label:22s} SKIP inexpressible (needs ROWS_T==1)")
                    continue
                kw = dict(
                    rows_t=rows_t,
                    cols=cols,
                    variant=variant,
                    blk=blk,
                    out_bulk=out_bulk,
                    reconfig=reconfig,
                    grid=grid,
                    dtype=dtype,
                )
                try:
                    lo, hi = iters
                    ns_lo, out = measure(device, tt, hw, iters=lo, **kw)
                    pcc, ratio, amax = check(out, ref, hw, grid=grid)
                    ok = pcc >= pcc_floor and abs(ratio - 1.0) <= ratio_tol
                    ttnn.deallocate(out)
                    ns_hi, out_hi = measure(device, tt, hw, iters=hi, **kw)
                    ttnn.deallocate(out_hi)
                    slope = (ns_hi - ns_lo) / (hi - lo) if (ns_hi and ns_lo) else float("nan")
                    print(
                        f"RESULT {dtype} {rows_t}x{cols} {label:22s} {'PASS' if ok else 'BROKEN'} "
                        f"pcc={pcc:.6f} ratio_med={ratio:.6f} amax={amax:.5f} "
                        f"ns@{lo}={ns_lo} ns@{hi}={ns_hi} per_block={slope:.1f}"
                    )
                except Exception:  # a variant that will not compile / hangs is DATA
                    print(f"RESULT {dtype} {rows_t}x{cols} {label:22s} ERROR")
                    traceback.print_exc()
    finally:
        ttnn.close_device(device)
