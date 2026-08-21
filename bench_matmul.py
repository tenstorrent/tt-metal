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
in tiles, the inner dimension, how many k-blocks the total is accumulated over, and how
the running total is carried. Both sides are pinned to one core and HiFi2, since the
comparison is meaningless otherwise -- ttnn's default fidelity is not ours.

    python bench_matmul.py
    python bench_matmul.py --rt 1 2 4 --ct 1 2 4 --kt 4
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
    if "static assertion" in text or "static_assert" in text:
        return "rejected by a static_assert"
    return type(exc).__name__


def ours(device, matmul, rt, ct, kt, k_blocks, mode):
    """(us, note). note is non-empty when the shape is not expressible OR not measurable."""
    try:
        got, want = matmul.run(device, rt, ct, kt, k_blocks=k_blocks, mode=mode, fidelity=HIFI2)
    except Exception as exc:  # noqa: BLE001 - the library refusing a shape IS the result
        text = str(exc)
        if "per-acquire DST budget" in text:
            return None, "DST: rt*ct over budget on the accumulating path"
        if "wider than the DST budget in a SINGLE row" in text:
            return None, "DST: ct over 8, no row band fits"
        if "static assertion" in text or "static_assert" in text:
            return None, "rejected by a static_assert"
        return None, classify(exc)
    # A wrong answer would make a timing meaningless, so gate before timing.
    scale = max(want.abs().max().item(), 1e-6)
    rel = (got - want).abs().max().item() / scale
    if rel > 0.05:
        return None, f"WRONG (rel {rel:.3f})"
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
    p.add_argument("--kt", type=int, nargs="+", default=[2, 8])
    p.add_argument("--k-blocks", type=int, nargs="+", default=[1, 4])
    p.add_argument("--modes", nargs="+", default=["dst", "l1", "single"], choices=["dst", "l1", "single"])
    args = p.parse_args(argv)

    import test_unified_matmul as matmul

    device = ttnn.open_device(device_id=0)
    rows = []
    try:
        for mode in args.modes:
            # The single-shot path is one k-block by definition; it is the banded one,
            # and the reason it is in the sweep is that its shape limits differ.
            k_blocks = [1] if mode == "single" else args.k_blocks
            for kb in k_blocks:
                for kt in args.kt:
                    for rt in args.rt:
                        for ct in args.ct:
                            us, note = ours(device, matmul, rt, ct, kt, kb, mode)
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
