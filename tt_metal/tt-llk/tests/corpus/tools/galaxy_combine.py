#!/usr/bin/env python3
"""Fail-closed combiner for galaxy_shard.sh slice verdicts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def verdict_tokens(text: str) -> dict[str, str]:
    """Parse whitespace-delimited key=value fields without substring matches."""
    pairs = re.findall(r"(?:^|\s)([A-Za-z_]+)=([^\s]+)", text)
    if len({key for key, _ in pairs}) != len(pairs):
        return {}
    return dict(pairs)


def combine(
    out: Path,
    npar: int,
    space: int,
    op: str,
    golden: bool,
    shard_failed: bool,
    failed_chips: "set[int] | None" = None,
):
    covered = 0
    # A zero-sized slice is not a covered slice: with space==0 (or space<npar)
    # every per-slice check below compares 0 against 0 and passes, so empty
    # slices used to combine into BIT-EXACT-ALL-INPUTS having enumerated
    # nothing.  Require a positive space AND a positive slice.
    expected_per_slice = (
        space // npar if npar > 0 and space > 0 and space % npar == 0 else None
    )
    if expected_per_slice is not None and expected_per_slice < 1:
        expected_per_slice = None
    all_equal = True
    numeric_ok = True
    witness = []
    # Chips the driver saw exit non-zero are invalid by identity, whatever
    # verdict file happens to be sitting in their slice dir.
    invalid = set(failed_chips or ())
    if invalid:
        all_equal = False
        numeric_ok = False
    for chip in range(npar):
        slice_dir = out / f"slice-{chip}"
        verdict_path = slice_dir / f"{op}-VERDICT.txt"
        if not verdict_path.exists():
            invalid.add(chip)
            all_equal = False
            numeric_ok = False
            continue
        verdict_text = verdict_path.read_text()
        tokens = verdict_tokens(verdict_text)
        try:
            slice_start = int(tokens["start"])
            slice_total = int(tokens["total"])
            slice_covered = int(tokens["covered"])
        except (KeyError, ValueError):
            slice_start = slice_total = slice_covered = -1
        covered += slice_covered
        expected_start = chip * expected_per_slice if expected_per_slice is not None else -1
        if (
            tokens.get("OP") != op
            or expected_per_slice is None
            or slice_start != expected_start
            or slice_total != expected_per_slice
            or slice_covered != expected_per_slice
        ):
            invalid.add(chip)
            all_equal = False
            numeric_ok = False
        if not (tokens.get("VERDICT") or "").startswith("BIT-EXACT"):
            all_equal = False
            bands = re.search(r"witness_bands=(\[.*\])", verdict_text)
            witness.append((chip, bands.group(1) if bands else "?"))
        if golden:
            correctness = slice_dir / f"{op}-CORRECTNESS-VERDICT.txt"
            correctness_tokens = (
                verdict_tokens(correctness.read_text()) if correctness.exists() else {}
            )
            if (
                correctness_tokens.get("OP") != op
                or correctness_tokens.get("NUMERIC_GATE") != "PASS"
                or correctness_tokens.get("ULP_ADMISSION") != "PASS"
            ):
                numeric_ok = False
    invalid_list = sorted(invalid)
    full = covered == space and not invalid_list and not shard_failed
    if all_equal and full:
        verdict = "BIT-EXACT-ALL-INPUTS"
    elif not all_equal and not invalid_list:
        verdict = "DIVERGENT"
    else:
        verdict = "INCOMPLETE"
    numeric_status = (
        "NOT_REQUESTED" if not golden else ("PASS" if numeric_ok else "FAIL")
    )
    summary = (
        f"OP={op} VERDICT={verdict} slices={npar} covered={covered} "
        f"(full {space}={covered == space}) invalid={invalid_list} witness={witness} "
        f"numeric_gate={numeric_status} "
        "numeric_contract=TOLERANCE-PLUS-SAME-ORACLE-ULP-NONREGRESSION-"
        "NOT-ABSOLUTE-ULP-CERTIFIED"
    )
    return summary, verdict.startswith("BIT-EXACT") and numeric_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("npar", type=int)
    parser.add_argument("space", type=int)
    parser.add_argument("op")
    parser.add_argument("golden", type=int, choices=(0, 1))
    parser.add_argument("shard_rc", type=int)
    parser.add_argument(
        "--failed-chips",
        default="",
        help="comma-separated chip indices whose slice process exited non-zero; "
        "their ranges are marked invalid regardless of any verdict file present",
    )
    args = parser.parse_args()
    failed_chips = {int(x) for x in args.failed_chips.split(",") if x.strip()}
    summary, passed = combine(
        args.out,
        args.npar,
        args.space,
        args.op,
        bool(args.golden),
        bool(args.shard_rc),
        failed_chips,
    )
    print(summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
