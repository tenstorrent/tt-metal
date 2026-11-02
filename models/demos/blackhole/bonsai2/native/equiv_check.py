# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-format equivalence: do PQ2_0, Q2_0 and PTQ1_0 hold the same weights?

The three GGUFs are three encodings of one model. If they agree, then
everything downstream can be cross-checked against any of them -- notably the
stock llama.cpp ground truth for Q2_0 becomes usable for PQ2_0, and a single
engine with a group-size parameter serves all three.

What "agree" means here:

  codes     identical int8 trits, position by position
  scales    Q2_0's two 64-weight scales inside one PQ2_0 128-group must equal
            the PQ2_0 scale (the plan's dedupe check).  If they differ, Q2_0
            holds independently quantized weights and needs its own accuracy
            comparison.

Also reports the group-size cost the plan predicts: Q2_0 carries twice as many
scales, so about 6% more bytes for the same weights.

Usage::

    python -m models.demos.blackhole.bonsai2.native.equiv_check \
        --pq2_0 .../PQ2_0.gguf --q2_0 .../Q2_0.gguf [--ptq1_0 .../PTQ1_0.gguf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .gguf_ternary import TERNARY_TYPES, read_gguf, read_ternary


def _names(gguf) -> list[str]:
    return sorted(n for n, t in gguf.tensors.items() if t.ggml_type in TERNARY_TYPES)


def compare_codes(a, b, name: str) -> tuple[float, float]:
    """Fraction of matching trits and the largest absolute difference."""
    if a.shape != b.shape:
        return 0.0, np.inf
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(np.mean(diff == 0)), float(diff.max())


def compare_scales(a_scales: np.ndarray, a_g: int, b_scales: np.ndarray, b_g: int) -> dict:
    """Compare two scale tables over the same K, reporting both relationships.

    ``b`` is the finer group (or equal).  Reports:
      sub_equal   every fine scale equals its coarse parent (dedupe holds)
      sub_max_rel largest relative deviation
      coarse_mean the coarse table's mean scale, for scale-of-error context
    """
    k = a_scales.shape[1] * a_g
    if b_scales.shape[1] * b_g != k:
        raise ValueError(f"K mismatch: {a_scales.shape[1]}x{a_g} vs {b_scales.shape[1]}x{b_g}")
    fine_g, coarse_g = min(a_g, b_g), max(a_g, b_g)
    fine = a_scales.astype(np.float64) if a_g == fine_g else b_scales.astype(np.float64)
    coarse = b_scales.astype(np.float64) if a_g == fine_g else a_scales.astype(np.float64)
    ratio = coarse_g // fine_g
    if ratio * fine_g != coarse_g:
        raise ValueError(f"group sizes {a_g} and {b_g} are not commensurate")
    parent = np.repeat(coarse, ratio, axis=1)
    denom = np.maximum(np.abs(parent), 1e-30)
    rel = np.abs(fine - parent) / denom
    return {
        "sub_equal": bool(np.array_equal(fine, parent)),
        "sub_max_rel": float(rel.max()),
        "sub_mean_rel": float(rel.mean()),
        "coarse_mean_abs": float(np.abs(coarse).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pq2_0", required=True)
    ap.add_argument("--q2_0", required=True)
    ap.add_argument("--ptq1_0", default=None)
    ap.add_argument("--limit", type=int, default=0, help="check only the first N tensors per file")
    ap.add_argument("--tensors", default="", help="comma-separated GGUF tensor names to restrict to")
    args = ap.parse_args()

    files = {"pq2_0": Path(args.pq2_0), "q2_0": Path(args.q2_0)}
    if args.ptq1_0:
        files["ptq1_0"] = Path(args.ptq1_0)
    ggufs = {k: read_gguf(v) for k, v in files.items()}

    names = _names(ggufs["pq2_0"])
    if args.tensors:
        want = set(args.tensors.split(","))
        names = [n for n in names if n in want]
    if args.limit:
        names = names[: args.limit]

    print(f"comparing {len(names)} ternary tensors across {len(ggufs)} files")
    for k, g in ggufs.items():
        have = set(_names(g))
        missing = [n for n in names if n not in have]
        if missing:
            print(f"  {k}: {len(missing)} tensors missing, e.g. {missing[:3]}")

    stats = {k: {"code_match": [], "scale": [], "bytes": 0} for k in ("pq2_0", "q2_0", "ptq1_0")}
    failures = 0

    for name in names:
        t_pq = read_ternary(ggufs["pq2_0"], name)
        for other in ("q2_0", "ptq1_0"):
            if other not in ggufs:
                continue
            t_o = read_ternary(ggufs[other], name)
            match, maxdiff = compare_codes(t_pq.codes, t_o.codes, name)
            sc = compare_scales(t_pq.scales, t_pq.group, t_o.scales, t_o.group)
            stats[other]["code_match"].append(match)
            stats[other]["scale"].append(sc["sub_max_rel"])
            if match < 1.0 or not sc["sub_equal"]:
                failures += 1
                if failures <= 12:
                    print(
                        f"  {name:28s} vs {other}: codes {match * 100:.4f}% maxdiff {maxdiff:g}  "
                        f"scales equal={sc['sub_equal']} maxrel={sc['sub_max_rel']:.3e} "
                        f"(coarse |s| mean {sc['coarse_mean_abs']:.3e})"
                    )

    # Byte accounting over the tensors actually compared.
    for key, g in ggufs.items():
        total = sum(g.tensors[n].nbytes for n in names if n in g.tensors)
        stats[key]["bytes"] = total
        print(f"  {key}: {total / 1e9:.4f} GB for these {len(names)} tensors")

    print()
    for other in ("q2_0", "ptq1_0"):
        if other not in ggufs:
            continue
        cm = stats[other]["code_match"]
        sm = stats[other]["scale"]
        print(
            f"{other}: {len(cm)} tensors, codes identical in {sum(1 for x in cm if x == 1.0)}/{len(cm)}, "
            f"mean trit match {np.mean(cm) * 100:.6f}%, scales deduped in {sum(1 for x in sm if x == 0.0)}/{len(sm)}, "
            f"max scale rel dev {max(sm):.3e}"
        )

    base = stats["pq2_0"]["bytes"]
    for other in ("q2_0", "ptq1_0"):
        if other not in ggufs:
            continue
        b = stats[other]["bytes"]
        print(f"  {other} vs pq2_0 bytes: {b / base * 100:.2f}%")

    if failures:
        print(f"\n{failures} tensor/format pairs differ; see the lines above.")
    else:
        print("\nall compared tensors are bit-identical in codes and scales; one engine serves all formats.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
