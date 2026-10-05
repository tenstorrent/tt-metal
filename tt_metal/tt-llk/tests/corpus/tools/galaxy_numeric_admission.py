#!/usr/bin/env python3
"""Aggregate Galaxy golden sidecars into one semantic-uplift admission.

Bit equivalence and numerical quality answer different questions.  The former
is copied from galaxy_combine's campaign verdict.  The latter is computed here
over the *whole* input population:

  * the semantic arm must satisfy its absolute oracle contract;
  * semantic max ULP must be no worse than handwritten max ULP in every
    populated input class, against the same oracle and population;
  * the handwritten arm's absolute contract is reported, but is not an
    admission requirement.  That is precisely the useful semantic-uplift case.

Per-slice admission is intentionally not composed: max(candidate)<=max(hand)
is a population-level rule and ANDing the same comparison over shards is
strictly stronger than that rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import galaxy_combine
import ulp_admission


def parse_corr(path: Path) -> dict[str, str]:
    lines = path.read_text().strip().splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one correctness record")
    fields = lines[0].split(",")
    if not fields or fields[0] != "SFPU_CORRECTNESS":
        raise ValueError(f"{path}: not an SFPU_CORRECTNESS record")
    result: dict[str, str] = {}
    for field in fields[1:]:
        if "=" not in field:
            raise ValueError(f"{path}: malformed field {field!r}")
        key, value = field.split("=", 1)
        if not key or key in result:
            raise ValueError(f"{path}: duplicate or empty key {key!r}")
        result[key] = value
    return result


def _as_count(row: dict[str, str], key: str, path: Path) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid or missing {key}") from error
    if value < 0:
        raise ValueError(f"{path}: negative {key}")
    return value


def _band(row: dict[str, str], path: Path, op: str, leg: str) -> dict[str, Any]:
    if row.get("op") != op or row.get("leg") != leg:
        raise ValueError(
            f"{path}: identity mismatch, expected op={op},leg={leg}; "
            f"got op={row.get('op')},leg={row.get('leg')}"
        )
    count_key = "patterns" if "patterns" in row else "joints" if "joints" in row else ""
    if not count_key:
        raise ValueError(f"{path}: missing patterns/joints")
    count = _as_count(row, count_key, path)
    if count == 0:
        raise ValueError(f"{path}: empty correctness population")
    n_out = _as_count(row, "n_out_of_tol", path)
    if n_out > count:
        raise ValueError(f"{path}: n_out_of_tol exceeds population")
    classes = ulp_admission.parse_class_ulp(row.get("class_ulp", ""))
    known_classes = (
        ulp_admission.unary_class_names(True) | ulp_admission.BINARY_CLASS_NAMES
    )
    unknown_classes = set(classes) - known_classes
    if unknown_classes:
        raise ValueError(f"{path}: unknown input classes {sorted(unknown_classes)}")
    if sum(n for n, _ in classes.values()) != count:
        raise ValueError(f"{path}: class population does not cover the band")
    within = row.get("within_contract")
    if within not in ("True", "False") or (within == "True") != (n_out == 0):
        raise ValueError(f"{path}: inconsistent within_contract")
    if "status" in row:
        raise ValueError(f"{path}: unchecked correctness record")

    graded_present = "n_graded" in row or "n_out_graded" in row
    n_graded = n_out_graded = 0
    if graded_present:
        if "n_graded" not in row or "n_out_graded" not in row:
            raise ValueError(f"{path}: partial graded-contract counts")
        n_graded = _as_count(row, "n_graded", path)
        n_out_graded = _as_count(row, "n_out_graded", path)
        if n_graded > count or n_out_graded > n_graded:
            raise ValueError(f"{path}: invalid graded-contract counts")
    return {
        "count": count,
        "n_out": n_out,
        "classes": classes,
        "graded_present": graded_present,
        "n_graded": n_graded,
        "n_out_graded": n_out_graded,
    }


def _equivalence(root: Path, op: str) -> dict[str, Any]:
    path = root / f"{op}-VERDICT.txt"
    if not path.is_file():
        return {"status": "UNKNOWN", "covered": None, "full_space": None}
    tokens = galaxy_combine.verdict_tokens(path.read_text())
    verdict = tokens.get("VERDICT", "UNKNOWN")
    try:
        covered = int(tokens["covered"])
        full_space = int(tokens["full_space"])
    except (KeyError, ValueError):
        covered = full_space = None
    if verdict.startswith("BIT-EXACT"):
        status = "BIT_EXACT"
    elif verdict == "DIVERGENT":
        status = "DIVERGENT"
    elif verdict == "INCOMPLETE":
        status = "INCOMPLETE"
    else:
        status = "UNKNOWN"
    return {
        "status": status,
        "verdict": verdict,
        "covered": covered,
        "full_space": full_space,
    }


def aggregate(root: Path, op: str) -> dict[str, Any]:
    """Return one fail-closed admission record for an op output directory."""
    root = Path(root)
    equiv = _equivalence(root, op)
    sem_paths = sorted(root.glob("slice-*/bands/*-sem.txt.corr"))
    hand_paths = sorted(root.glob("slice-*/bands/*-hand.txt.corr"))
    if not sem_paths and not hand_paths:
        return {
            "op": op,
            "equivalence": equiv,
            "oracle": "MISSING",
            "semantic_absolute": "NOT_RUN",
            "hand_absolute": "NOT_RUN",
            "ulp_nonregression": "NOT_RUN",
            "numeric_admission": "NO_ORACLE",
            "reason": "no-correctness-sidecars",
        }

    sem_by_key = {p.relative_to(root).as_posix().removesuffix("-sem.txt.corr"): p for p in sem_paths}
    hand_by_key = {p.relative_to(root).as_posix().removesuffix("-hand.txt.corr"): p for p in hand_paths}
    if set(sem_by_key) != set(hand_by_key):
        missing_sem = sorted(set(hand_by_key) - set(sem_by_key))
        missing_hand = sorted(set(sem_by_key) - set(hand_by_key))
        return {
            "op": op,
            "equivalence": equiv,
            "oracle": "INCOMPLETE",
            "semantic_absolute": "NOT_RUN",
            "hand_absolute": "NOT_RUN",
            "ulp_nonregression": "NOT_RUN",
            "numeric_admission": "INCOMPLETE",
            "reason": f"unpaired-sidecars:missing-sem={missing_sem},missing-hand={missing_hand}",
        }

    legs = {
        "sem": {"patterns": 0, "n_out": 0, "n_graded": 0, "n_out_graded": 0, "graded": None, "classes": {}},
        "hand": {"patterns": 0, "n_out": 0, "n_graded": 0, "n_out_graded": 0, "graded": None, "classes": {}},
    }
    try:
        for key in sorted(sem_by_key):
            pair = {}
            for leg, paths in (("sem", sem_by_key), ("hand", hand_by_key)):
                path = paths[key]
                pair[leg] = _band(parse_corr(path), path, op, leg)
            if pair["sem"]["count"] != pair["hand"]["count"]:
                raise ValueError(f"{key}: semantic/hand population mismatch")
            if pair["sem"]["graded_present"] != pair["hand"]["graded_present"]:
                raise ValueError(f"{key}: semantic/hand graded schema mismatch")
            if pair["sem"]["n_graded"] != pair["hand"]["n_graded"]:
                raise ValueError(f"{key}: semantic/hand graded population mismatch")
            sem_classes = pair["sem"]["classes"]
            hand_classes = pair["hand"]["classes"]
            if set(sem_classes) != set(hand_classes) or any(
                sem_classes[name][0] != hand_classes[name][0]
                for name in sem_classes
            ):
                raise ValueError(f"{key}: semantic/hand class population mismatch")
            for leg in ("sem", "hand"):
                band = pair[leg]
                acc = legs[leg]
                acc["patterns"] += band["count"]
                acc["n_out"] += band["n_out"]
                acc["n_graded"] += band["n_graded"]
                acc["n_out_graded"] += band["n_out_graded"]
                acc["graded"] = band["graded_present"] if acc["graded"] is None else acc["graded"]
                if acc["graded"] != band["graded_present"]:
                    raise ValueError(f"{key}: mixed graded/non-graded sidecars")
                ulp_admission.fold_class_ulp(
                    acc["classes"], ulp_admission.format_class_ulp(band["classes"])
                )
    except ValueError as error:
        return {
            "op": op,
            "equivalence": equiv,
            "oracle": "INVALID",
            "semantic_absolute": "NOT_RUN",
            "hand_absolute": "NOT_RUN",
            "ulp_nonregression": "NOT_RUN",
            "numeric_admission": "INCOMPLETE",
            "reason": str(error),
        }

    covered = equiv.get("covered")
    full_space = equiv.get("full_space")
    complete = (
        equiv["status"] in ("BIT_EXACT", "DIVERGENT")
        and covered is not None
        and full_space is not None
        and covered == full_space
        and legs["sem"]["patterns"] == covered
        and legs["hand"]["patterns"] == covered
    )
    sem_in = (
        legs["sem"]["n_out_graded"] == 0
        if legs["sem"]["graded"] and legs["sem"]["n_graded"] > 0
        else legs["sem"]["n_out"] == 0
    )
    hand_in = (
        legs["hand"]["n_out_graded"] == 0
        if legs["hand"]["graded"] and legs["hand"]["n_graded"] > 0
        else legs["hand"]["n_out"] == 0
    )
    ulp_ok, ulp_reason = ulp_admission.candidate_not_worse(
        legs["sem"]["classes"], legs["hand"]["classes"]
    )
    admitted = complete and sem_in and ulp_ok
    if not complete:
        reason = "incomplete-full-space-or-oracle-coverage"
        status = "INCOMPLETE"
    elif not sem_in:
        reason = "semantic-arm-outside-absolute-contract"
        status = "FAIL"
    elif not ulp_ok:
        reason = ulp_reason
        status = "FAIL"
    else:
        reason = "semantic-absolute-pass-and-global-class-ulp-no-worse"
        status = "PASS"

    return {
        "op": op,
        "equivalence": equiv,
        "oracle": "AVAILABLE",
        "bands": len(sem_by_key),
        "patterns": legs["sem"]["patterns"],
        "semantic_absolute": "PASS" if sem_in else "FAIL",
        "hand_absolute": "PASS" if hand_in else "FAIL",
        "ulp_nonregression": "PASS" if ulp_ok else "FAIL",
        "ulp_reason": ulp_reason,
        "semantic_class_ulp": ulp_admission.format_class_ulp(legs["sem"]["classes"]),
        "hand_class_ulp": ulp_admission.format_class_ulp(legs["hand"]["classes"]),
        "semantic_n_out": legs["sem"]["n_out"],
        "hand_n_out": legs["hand"]["n_out"],
        "semantic_n_out_graded": legs["sem"]["n_out_graded"],
        "hand_n_out_graded": legs["hand"]["n_out_graded"],
        "numeric_admission": status,
        "reason": reason,
        "admitted": admitted,
    }


def write_record(record: dict[str, Any], prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    flat = {
        "op": record["op"],
        "equivalence": record["equivalence"]["status"],
        "oracle": record["oracle"],
        "semantic_absolute": record["semantic_absolute"],
        "hand_absolute": record["hand_absolute"],
        "ulp_nonregression": record["ulp_nonregression"],
        "numeric_admission": record["numeric_admission"],
        "patterns": record.get("patterns", ""),
        "reason": record["reason"],
    }
    with prefix.with_suffix(".tsv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat), delimiter="\t")
        writer.writeheader()
        writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="one op's galaxy_shard OUT directory")
    parser.add_argument("op")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        help="write PREFIX.json and PREFIX.tsv (default: print JSON only)",
    )
    args = parser.parse_args()
    record = aggregate(args.root, args.op)
    if args.out_prefix:
        write_record(record, args.out_prefix)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["numeric_admission"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
