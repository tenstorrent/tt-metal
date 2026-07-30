#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare two judged runs PAIRED on ``doc["Record ID"]``, and say whether the gap is real.

Aggregates mislead on 198 questions. A run that scored 72.1% against another's 69.5% looks like a
2.6 pp regression; paired, it was 18 questions one way and 13 the other, two-sided p ~ 0.47 -- noise.
Binomial sigma at n=198, p=0.7 is already 3.3 pp, so anything under ~6 pp needs the paired test before
it means anything.

The join key is ``Record ID``: lm_eval reshuffles the choices per run, so ``doc_hash`` matches only 18
of 198 and letters are not comparable across runs. ``llm_judge.py --out`` writes the Record ID as
``id``, so its verdict files join directly.

Usage::

    compare_runs.py baseline_verdicts.jsonl candidate_verdicts.jsonl
    compare_runs.py a.jsonl b.jsonl --labels "CI 73.2%" "fullcanvas"
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    rows = {}
    for line in path.open(errors="replace"):
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def two_sided_p(a: int, b: int) -> float:
    """Exact binomial sign test on the discordant pairs -- the paired analogue of a t-test here."""
    n = a + b
    if n == 0:
        return 1.0
    lo = min(a, b)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2**n
    return min(1.0, 2 * tail)


def correct(v: dict) -> bool:
    return bool(v.get("judge_letter")) and v["judge_letter"] == v.get("gold_letter")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--labels", nargs=2, default=None)
    args = ap.parse_args()

    a, b = load(args.baseline), load(args.candidate)
    la, lb = args.labels or (args.baseline.stem, args.candidate.stem)

    # Judge errors (a refused question) are excluded from BOTH sides so the denominators match.
    common = [k for k in a if k in b and not a[k].get("error") and not b[k].get("error")]
    if not common:
        raise SystemExit("no Record IDs in common -- are both files llm_judge --out verdicts?")
    dropped = (len(a) - len(common), len(b) - len(common))
    print(f"paired on Record ID: {len(common)} questions in both")
    if any(dropped):
        print(f"  ({dropped[0]} / {dropped[1]} excluded as unmatched or judge-errored)")
    print(f"\n{'':36s} {la[:22]:>22s}  {lb[:22]:>22s}")

    def row(name, fn):
        x = sum(1 for k in common if fn(a[k]))
        y = sum(1 for k in common if fn(b[k]))
        n = len(common)
        print(f"  {name:34s} {x:>5}/{n} {100*x/n:>6.1f}%   {y:>5}/{n} {100*y/n:>6.1f}%")

    row("lm_eval scored correct", lambda v: v.get("regex_correct"))
    row("judge-confirmed correct", correct)
    row("stated an answer", lambda v: v.get("answered"))
    row("regex credited a non-answer", lambda v: v.get("regex_correct") and not v.get("answered"))
    row("non-English (drift)", lambda v: v.get("language") not in ("en", None))

    both = sum(1 for k in common if correct(a[k]) and correct(b[k]))
    a_only = [k for k in common if correct(a[k]) and not correct(b[k])]
    b_only = [k for k in common if not correct(a[k]) and correct(b[k])]
    p = two_sided_p(len(a_only), len(b_only))
    print(f"\npaired flips on judge-confirmed correctness")
    print(
        f"  both correct {both}   {la} only {len(a_only)}   {lb} only {len(b_only)}   "
        f"neither {len(common) - both - len(a_only) - len(b_only)}"
    )
    print(f"  discordant {len(a_only) + len(b_only)} ({len(a_only)} vs {len(b_only)})  two-sided p = {p:.3f}")
    print("  => NOT distinguishable from run-to-run noise" if p > 0.05 else "  => significant at p<0.05")

    print("\nfailure modes")
    ma = collections.Counter(a[k].get("failure_mode") for k in common)
    mb = collections.Counter(b[k].get("failure_mode") for k in common)
    for mode in sorted(set(ma) | set(mb), key=lambda m: -(ma.get(m, 0) + mb.get(m, 0))):
        print(f"  {str(mode):24s} {ma.get(mode,0):>5}                  {mb.get(mode,0):>5}")

    bad_a = {k for k in common if not a[k].get("answered")}
    bad_b = {k for k in common if not b[k].get("answered")}
    shared = sorted(bad_a & bad_b)
    print(f"\nnon-answers: {la} {len(bad_a)}, {lb} {len(bad_b)}, BOTH {len(shared)}")
    if shared:
        print("  recurring across both runs -- question-specific, not random:")
        for k in shared:
            print(f"    {k}: {a[k].get('failure_mode')} / {b[k].get('failure_mode')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
