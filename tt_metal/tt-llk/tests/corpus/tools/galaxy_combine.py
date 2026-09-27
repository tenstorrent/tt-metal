#!/usr/bin/env python3
"""Fail-closed combiner for galaxy_shard.sh slice verdicts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def combine(out: Path, npar: int, space: int, op: str, golden: bool, shard_failed: bool):
    covered = 0
    expected_per_slice = space // npar if npar > 0 and space % npar == 0 else None
    all_equal = True
    numeric_ok = True
    witness = []
    missing = []
    for chip in range(npar):
        slice_dir = out / f"slice-{chip}"
        verdict_path = slice_dir / f"{op}-VERDICT.txt"
        if not verdict_path.exists():
            missing.append(chip)
            all_equal = False
            numeric_ok = False
            continue
        verdict_text = verdict_path.read_text()
        match = re.search(r"covered=(\d+)", verdict_text)
        slice_covered = int(match.group(1)) if match else 0
        covered += slice_covered
        if expected_per_slice is None or slice_covered != expected_per_slice:
            missing.append(chip)
            all_equal = False
            numeric_ok = False
        if "VERDICT=BIT-EXACT-ALL-INPUTS" not in verdict_text:
            all_equal = False
            bands = re.search(r"witness_bands=(\[.*\])", verdict_text)
            witness.append((chip, bands.group(1) if bands else "?"))
        if golden:
            correctness = slice_dir / f"{op}-CORRECTNESS-VERDICT.txt"
            if not correctness.exists() or "NUMERIC_GATE=PASS" not in correctness.read_text():
                numeric_ok = False
    full = covered == space and not missing and not shard_failed
    if all_equal and full:
        verdict = "BIT-EXACT-ALL-INPUTS"
    elif not all_equal and not missing:
        verdict = "DIVERGENT"
    else:
        verdict = "INCOMPLETE"
    summary = (
        f"OP={op} VERDICT={verdict} slices={npar} covered={covered} "
        f"(full {space}={covered == space}) missing={missing} witness={witness} "
        f"numeric_gate={'PASS' if numeric_ok else 'FAIL'} "
        "numeric_contract=TOLERANCE-ONLY-NOT-ULP-CERTIFIED"
    )
    return summary, verdict == "BIT-EXACT-ALL-INPUTS" and numeric_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("npar", type=int)
    parser.add_argument("space", type=int)
    parser.add_argument("op")
    parser.add_argument("golden", type=int, choices=(0, 1))
    parser.add_argument("shard_rc", type=int)
    args = parser.parse_args()
    summary, passed = combine(
        args.out, args.npar, args.space, args.op, bool(args.golden), bool(args.shard_rc)
    )
    print(summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
