#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate a bounded, reproducible auto-matmul tuning corpus (a manifest for
``tune_matmul_cache.py``).

The corpus is a deduped cross of a small, bounded set of M/K/N values (tile-aligned +
powers-of-two, plus any explicitly-listed model shapes) with the requested dtypes and
layouts. It is intentionally bounded -- this is the reproducible seed set the offline
cache + predictor index are built from, not an exhaustive scrape.

Example:
    python3 tools/auto_matmul/generate_corpus.py --out corpus.json
    python3 tools/auto_matmul/generate_corpus.py --out corpus.json \\
        --dtypes bfloat16 bfloat8_b --extra-shapes 2048x2880x5120 2048x4096x2880
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Bounded default axis values (in elements).  Tile-aligned (multiples of 32) and a few
# powers-of-two / non-power-of-two regimes so the corpus spans square / narrow / wide.
_DEFAULT_MKN = [32, 128, 512, 1024, 2048, 2880, 4096, 8192]
# Cap the dense cross so the corpus stays bounded regardless of axis growth.
_MAX_CASES = 4096


def _parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in text.lower().replace("x", ",").split(",") if p]
    if len(parts) != 3:
        raise ValueError(f"--extra-shapes entries must be MxKxN, got {text!r}")
    return parts[0], parts[1], parts[2]


def _regime(m: int, n: int) -> str:
    lo, hi = min(m, n), max(m, n)
    if lo > 0 and hi / lo > 8:
        return "narrow"
    return "square"


def _dense_shapes(values: list[int], *, include_narrow: bool) -> list[tuple[int, int, int]]:
    shapes: set[tuple[int, int, int]] = set()
    for k in values:
        # square/general: M == N sweep over K
        for mn in values:
            shapes.add((mn, k, mn))
        if include_narrow:
            # narrow: a small M (decode-like) and a small N (projection-like) per K
            for big in values:
                shapes.add((32, k, big))  # small M
                shapes.add((big, k, 32))  # small N
    return sorted(shapes)


def _build_cases(shapes: list[tuple[int, int, int]], dtypes: list[str], layouts: list[str]) -> list[dict]:
    cases: list[dict] = []
    seen: set[tuple] = set()
    for m, k, n in shapes:
        for dtype in dtypes:
            for layout in layouts:
                key = (m, k, n, dtype, layout)
                if key in seen:
                    continue
                seen.add(key)
                cases.append(
                    {
                        "name": f"{_regime(m, n)}_{m}x{k}x{n}_{dtype}",
                        "lhs_shape": [1, 1, m, k],
                        "rhs_shape": [1, 1, k, n],
                        "dtype": dtype,
                        "layout": layout,
                        "memory_config": "DRAM_MEMORY_CONFIG",
                    }
                )
                if len(cases) >= _MAX_CASES:
                    return cases
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a bounded auto-matmul tuning corpus manifest.")
    parser.add_argument("--out", required=True, help="Path to write the manifest JSON.")
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16"], help="Dtypes to cross (e.g. bfloat16 bfloat8_b).")
    parser.add_argument("--layouts", nargs="+", default=["TILE_LAYOUT"], help="Layouts to cross.")
    parser.add_argument("--values", nargs="+", type=int, default=_DEFAULT_MKN, help="M/K/N axis values (bounded).")
    parser.add_argument("--no-narrow", action="store_true", help="Skip the narrow (small-M / small-N) regime shapes.")
    parser.add_argument(
        "--extra-shapes", nargs="*", default=[], help="Explicit MxKxN model shapes to include (e.g. 2048x2880x5120)."
    )
    args = parser.parse_args()

    shapes = set(_dense_shapes(sorted(set(args.values)), include_narrow=not args.no_narrow))
    for text in args.extra_shapes:
        shapes.add(_parse_shape(text))

    cases = _build_cases(sorted(shapes), args.dtypes, args.layouts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"cases": cases}, indent=2, sort_keys=True))
    print(f"Wrote {len(cases)} cases to {out_path} (bounded at {_MAX_CASES}).")


if __name__ == "__main__":
    main()
