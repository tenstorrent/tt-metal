#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Score a TT GPQA run against the CUDA reference ON THE SAME QUESTIONS.

A TT run that stopped early -- an engine death, a killed stage, a partial CI eval -- can only be
compared to the reference restricted to the questions it actually answered. **The answered set is a
prefix, not a random sample**, so comparing it to the reference's full-198 figure is not a
conservative approximation, it is a wrong number in an unknown direction. On the 2026-07-28 run that
restriction moved the reference from 70.45% to 68.03%, which is the difference between "TT is 6 pp
behind" and "TT is 9 pp ahead".

THE JOIN KEY IS ``doc["Record ID"]``. The two obvious keys both fail:

* ``doc_hash`` looks canonical and matched **18 of 198** -- lm_eval reshuffles the multiple-choice
  options per run and the hash covers the shuffled ``choices``/``answer`` fields.
* ``doc_id`` happened to line up, but nothing in the file says so, and a differently seeded run
  breaks it silently.

For the same reason **letters are not comparable across runs**: the correct answer is "(A)" in one
run and "(B)" in another for the same question. So this never compares letters. It uses each run's
own ``exact_match``, which lm_eval computed against that run's own shuffle -- and the reference file
stores the correct answer TEXT rather than its letter, so a mismatch is detectable.

Usage::

    compare_same_questions.py /home/zni/dg_runs/cot_rerun
    compare_same_questions.py <run_dir> --stage smoke
    compare_same_questions.py <path/to/samples_*.jsonl>

Reports the TT score and the reference score over the same questions, the gap in binomial sigma,
per-question agreement, and -- as the floor any gap has to clear -- the reference's own rep-to-rep
agreement on that same subset.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "gpu_reference_by_record.json"


def load_tt(target: Path, stage: str):
    """Per-Record-ID TT results from an lm_eval samples file, or the newest one under a run dir."""
    if target.is_dir():
        candidates = sorted(target.rglob(f"{stage}/**/samples_*.jsonl")) or sorted(target.rglob("samples_*.jsonl"))
        if not candidates:
            sys.exit(f"no samples_*.jsonl under {target}")
        path = candidates[-1]
    else:
        path = target

    per_record = {}
    for line in path.open():
        row = json.loads(line)
        # One record per FILTER, so every raw count is doubled. flexible-extract is the scored one.
        if row.get("filter") not in (None, "flexible-extract"):
            continue
        doc = row.get("doc") or {}
        record = doc.get("Record ID")
        if record is None:
            sys.exit(f"{path} has no doc['Record ID']; cannot join without it")
        text = (row.get("resps") or [[""]])[0][0]
        per_record[record] = {
            "correct": bool(row.get("exact_match")),
            "empty": not text.strip(),
            "answer_text": str(doc.get("Correct Answer", "")).strip(),
            "domain": doc.get("High-level domain"),
        }
    return path, per_record


def rate(flags):
    n = len(flags)
    if not n:
        return 0, 0, 0.0, 0.0
    c = sum(1 for f in flags if f)
    p = c / n
    return c, n, 100.0 * p, 100.0 * math.sqrt(p * (1 - p) / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="run directory or a samples_*.jsonl")
    ap.add_argument("--stage", default="full", help="which stage's samples to prefer (default: full)")
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument(
        "--include-empty",
        action="store_true",
        help="count empty answers as wrong instead of excluding them (they are usually a serving "
        "failure, not a model answer -- see read_engine_health in watch_gpqa.py)",
    )
    args = ap.parse_args()

    if not args.reference.exists():
        sys.exit(f"no reference at {args.reference}")
    ref_blob = json.loads(args.reference.read_text())
    ref = ref_blob["per_record"]
    reps = sorted(k for k in next(iter(ref.values())) if k.startswith("rep"))

    path, tt = load_tt(args.target, args.stage)
    print(f"TT samples:  {path}")
    print(
        f"reference:   {args.reference}  ({', '.join(f'{r} {ref_blob['meta'][r]['score']*100:.2f}%' for r in reps)} over 198)"
    )

    common = [r for r in tt if r in ref]
    missing = [r for r in tt if r not in ref]
    if missing:
        print(f"  !! {len(missing)} TT record(s) absent from the reference; excluded")
    if not common:
        sys.exit("no Record ID overlap -- the two runs are not on the same dataset")

    # The correct-answer TEXT is shuffle-independent, so it is a real check that the join is sound.
    mismatched = [r for r in common if tt[r]["answer_text"] and tt[r]["answer_text"] != ref[r]["answer_text"]]
    if mismatched:
        print(f"  !! {len(mismatched)} record(s) disagree on the correct answer TEXT -- the join is")
        print("     unsound (different dataset revision?). Refusing to report a comparison.")
        for r in mismatched[:3]:
            print(f"       {r}: TT {tt[r]['answer_text']!r} vs ref {ref[r]['answer_text']!r}")
        return 1

    answered = [r for r in common if args.include_empty or not tt[r]["empty"]]
    empty = len(common) - len([r for r in common if not tt[r]["empty"]])
    print()
    print(f"questions in this TT run:   {len(common)} of 198")
    if empty:
        note = "counted as wrong" if args.include_empty else "EXCLUDED"
        print(f"  of which empty answers:   {empty}  ({note})")
    print(f"compared on:                {len(answered)} questions")
    if len(answered) < 30:
        print(f"  (n={len(answered)} is small: one question is worth {100.0/max(1,len(answered)):.0f} pp)")

    ct, nt, pt, et = rate([tt[r]["correct"] for r in answered])
    print()
    print(f"  {'TT':<22} {ct:3d}/{nt:<3d} = {pt:6.2f}%  +/- {et:.2f} pp")
    ref_scores = []
    for rep in reps:
        c, n, p, _ = rate([ref[r][rep] for r in answered])
        ref_scores.append(p)
        print(f"  {'reference ' + rep:<22} {c:3d}/{n:<3d} = {p:6.2f}%")
    bar = sum(ref_scores) / len(ref_scores)
    gap = pt - bar
    sigma = gap / et if et else 0.0
    print(f"  {'reference mean':<22} {'':8} {bar:6.2f}%   <- the bar, on THESE questions")
    print()
    verdict = (
        "indistinguishable from the reference"
        if abs(sigma) < 2
        else ("ABOVE the reference" if gap > 0 else "BELOW the reference")
    )
    print(f"  gap: {gap:+.2f} pp = {sigma:+.2f} sigma  -> {verdict}")

    # Agreement, and the floor it has to clear: the reference does not even agree with itself.
    print()
    for rep in reps:
        a, n, p, _ = rate([tt[r]["correct"] == ref[r][rep] for r in answered])
        print(f"  per-question agreement, TT vs {rep}: {a}/{n} = {p:.0f}%")
    if len(reps) >= 2:
        a, n, p, _ = rate([ref[r][reps[0]] == ref[r][reps[1]] for r in answered])
        print(f"  {reps[0]} vs {reps[1]}: {a}/{n} = {p:.0f}%   <- the reference's own noise floor")

    # Is the answered subset skewed? It is a prefix, so this is worth checking, not assuming.
    full = Counter(ref[r]["domain"] for r in common)
    sub = Counter(ref[r]["domain"] for r in answered)
    if len(answered) < len(common):
        print()
        print("  subject mix (a partial run's answered set is a prefix, not a sample):")
        for k in sorted(full):
            share_sub = 100.0 * sub.get(k, 0) / max(1, len(answered))
            share_full = 100.0 * full[k] / len(common)
            print(
                f"    {str(k):<12} {sub.get(k,0):3d}/{full[k]:<3d}   {share_sub:4.0f}% of compared vs {share_full:4.0f}% of set"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
