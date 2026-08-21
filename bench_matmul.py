# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep matmul block shapes against ttnn, on ONE core, looking for holes.

Two kinds of hole, and the sweep reports both:

  EXPRESSIBLE   shapes the library refuses. Asked by TRYING, not by re-deriving the
                rule in Python: a shape the library rejects raises out of the JIT, and
                the message is recorded. A Python copy of the constraint would go stale
                the moment the library's changed, and would then lie in exactly the
                direction that matters.
  RATE          shapes that work but are slow relative to ttnn.

The axes mirror ttnn's own single-core matmul microbenchmark
(tests/scripts/test_moreh_microbenchmark.py -> test_compute_mm): output rows and columns
in tiles, the inner dimension, and how the running total is carried. Both sides are pinned
to one core and HiFi2, since the comparison is meaningless otherwise -- ttnn's default
fidelity is not ours.

K is swept as kt, the inner dimension of ONE matmul call, and not as a count of k-blocks.
An earlier version had it the other way round, which manufactured a hole that was not
there: it only ever varied k_blocks at small kt, so the 1.3us those blocks cost read as a
property of blocked matmul instead of a cost we were inflicting on ourselves. kt is not a
DST dimension -- DST budgets the output block, rt*ct, and matmul_block accumulates its
whole k-loop into the same registers -- so kt runs as far as L1 will hold the operand
blocks, which is past any single-core transformer's hidden size. Blocking is still
reachable with --k-blocks, for the case where K genuinely does not fit, but it is a knob
and not an axis.

The ceiling this sweep does find is L1: the operands are rt*kt and kt*ct tiles, so a large
kt against a large rt or ct exhausts the allocator. That is a real limit and is reported
like any other hole.

    python bench_matmul.py
    python bench_matmul.py --rt 1 2 4 --ct 1 2 4 --kt 4 64
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_bench import bench

TILE = 32
HIFI2 = {"math_fidelity": ttnn.MathFidelity.HiFi2, "math_approx_mode": False}


def classify(exc):
    """Turn a failure into a short reason, keyed off the library's own assert text."""
    text = str(exc)
    if "per-acquire DST budget" in text:
        return "DST: rt*ct over budget on the accumulating path"
    if "wider than the DST budget in a SINGLE row" in text:
        return "DST: ct over 8, no row band fits"
    if "no device records matched" in text:
        return "no profiler records"
    # Not a library refusal: the shape is legal and the chip is too small for it. This is
    # the limit that actually bounds kt, so it gets named rather than reported as a bare
    # exception type.
    if "Out of Memory" in text or "not enough space" in text.lower() or "circular buffer" in text.lower():
        return "L1: operands do not fit (rt*kt + kt*ct tiles)"
    if "static assertion" in text or "static_assert" in text:
        return "rejected by a static_assert"
    return type(exc).__name__


def ours(device, matmul, rt, ct, kt, k_blocks, mode, min_pcc):
    """(us, note). note is non-empty when the shape is not expressible OR not measurable."""
    try:
        got, want = matmul.run(device, rt, ct, kt, k_blocks=k_blocks, mode=mode, fidelity=HIFI2)
    except Exception as exc:  # noqa: BLE001 - the library refusing a shape IS the result
        return None, classify(exc)
    # A wrong answer would make a timing meaningless, so gate before timing -- but gate on
    # PCC, the same measure the test suite uses. A max-relative-error bound cannot do this
    # job once kt is an axis: bf16 accumulation error grows with the number of terms, so the
    # peak-relative deviation at 1x4 runs 0.017 at kt=8 and 0.077 at kt=128 while PCC holds
    # above 0.999. A 0.05 bound on it reported correct large-K shapes as WRONG.
    measured = matmul.pcc(got, want)
    if measured < min_pcc:
        return None, f"WRONG (pcc {measured:.4f})"
    # A failure to MEASURE is recorded, not raised. A sweep exists to explore a space where
    # failures are the interesting part, and one unmeasurable cell should not throw away the
    # other hundred and fifty.
    try:
        us = bench(
            device,
            lambda: matmul.run(device, rt, ct, kt, k_blocks=k_blocks, mode=mode, fidelity=HIFI2),
            iters=8,
            warmup=2,
            match="unified_kernels/matmul.cpp",
        )["median_us"]
    except Exception as exc:  # noqa: BLE001
        return None, classify(exc)
    return us, ""


def theirs(device, m, n, k, cache={}):
    """ttnn.matmul on one core at HiFi2, in microseconds."""
    key = (m, n, k)
    if key in cache:
        return cache[key]
    a = ttnn.from_torch(
        torch.randn([1, 1, m, k], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = ttnn.from_torch(
        torch.randn([1, 1, k, n], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2)
    try:
        us = bench(
            device,
            lambda: ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1), compute_kernel_config=ckc),
            iters=8,
            warmup=2,
            match="operations/matmul",
        )["median_us"]
    except Exception:  # noqa: BLE001 - record the gap rather than abort the sweep
        us = None
    cache[key] = us
    return us


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rt", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--ct", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--kt", type=int, nargs="+", default=[1, 2, 8, 32, 64])
    # A knob, not an axis -- see the module docstring. Blocking K is for a K too large for
    # L1; it costs 1.36us a block at no benefit otherwise.
    p.add_argument("--k-blocks", type=int, default=1)
    p.add_argument("--modes", nargs="+", default=["dst", "l1", "single"], choices=["dst", "l1", "single"])
    p.add_argument("--pcc", type=float, default=0.99, help="correctness gate before a cell is timed")
    args = p.parse_args(argv)

    import test_unified_matmul as matmul

    device = ttnn.open_device(device_id=0)
    rows = []
    try:
        for mode in args.modes:
            # The single-shot path is one k-block by definition; it is the banded one,
            # and the reason it is in the sweep is that its shape limits differ.
            kb = 1 if mode == "single" else args.k_blocks
            for kt in args.kt:
                for rt in args.rt:
                    for ct in args.ct:
                        us, note = ours(device, matmul, rt, ct, kt, kb, mode, args.pcc)
                        ref = theirs(device, rt * TILE, ct * TILE, kt * kb * TILE) if us is not None else None
                        rows.append((mode, kb, kt, rt, ct, us, ref, note))
    finally:
        ttnn.close_device(device)

    logger.info("one core, HiFi2 both sides. MACs = rt*ct*kt*k_blocks tile-multiplies.")
    logger.info(
        f"  {'mode':4s} {'kb':>3s} {'kt':>3s} {'rt':>3s} {'ct':>3s} {'MACs':>5s} "
        f"{'ours':>9s} {'ttnn':>9s} {'ratio':>6s}  note"
    )
    holes = []
    for mode, kb, kt, rt, ct, us, ref, note in rows:
        macs = rt * ct * kt * kb
        if us is None:
            logger.info(f"  {mode:4s} {kb:3d} {kt:3d} {rt:3d} {ct:3d} {macs:5d} {'-':>9s} {'-':>9s} {'-':>6s}  {note}")
            holes.append(("expressible", mode, kb, kt, rt, ct, note))
            continue
        r = us / ref if ref else None
        logger.info(
            f"  {mode:4s} {kb:3d} {kt:3d} {rt:3d} {ct:3d} {macs:5d} {us:7.2f}us "
            f"{(f'{ref:7.2f}us' if ref else '        -'):>9s} {(f'{r:.2f}x' if r else '-'):>6s}"
        )
        if r and r > 2.0:
            holes.append(("rate", mode, kb, kt, rt, ct, f"{r:.2f}x"))

    # Per-MAC cost is the shape-independent view: a shape whose per-MAC rate is far off
    # the best one we achieve is losing to overhead, not to arithmetic.
    good = [(rt * ct * kt * kb, us) for mode, kb, kt, rt, ct, us, ref, note in rows if us]
    if good:
        best = min(us / macs for macs, us in good)
        logger.info(f"  best per-MAC cost achieved: {best:.3f}us")
        for mode, kb, kt, rt, ct, us, ref, note in rows:
            if not us:
                continue
            per = us / (rt * ct * kt * kb)
            if per > 3 * best:
                logger.info(f"    {mode} kb={kb} kt={kt} {rt}x{ct}: {per:.3f}us/MAC, {per / best:.1f}x the best")

    logger.info(f"holes found: {len(holes)}")
    for h in holes:
        logger.info(f"  {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
