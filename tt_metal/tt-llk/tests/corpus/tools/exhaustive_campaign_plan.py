#!/usr/bin/env python3
"""Plan exhaustive silicon coverage for a selected LLK/knob campaign.

This is a device-free planner.  It joins the current sweep corpus, search
selection and (optionally) an existing validation ledger, then emits explicit
run rosters.  It does not compile, submit jobs, or reinterpret a sampled run as
exhaustive coverage.

The current unary nodes omit their format from several pytest ids.  The
32-bit set below mirrors the format choices in test_sfpu_unary.py; every other
paired node in that file is Float16_b.  The census assertions make a newly
added/renamed row fail visibly instead of silently falling into the wrong
space.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


TWO16 = 1 << 16
TWO32 = 1 << 32

# Float32, Int32 and UInt32 pointwise unary rows in test_sfpu_unary.py.
UNARY_32 = frozenset(
    """absint32 add1 bitwisenot castfp32tofp16a cbrt-fresh clamp-fresh comp
    digamma-fresh eqz-fresh erf-fresh erfc-fresh erfinv-fresh expm1-fresh
    expm1cw-fresh fmod-fresh geluappx-fresh hardmish-fresh hardshrink-fresh
    hardtanh-fresh heaviside-fresh i1-fresh rdiv remainder-fresh rpow selu
    sigmoid-fresh sigmoidlut-fresh sign signbit softplus-fresh softshrink-fresh
    softsign-fresh tanhderivlut-fresh unarycomp-fresh unarypower-fresh
    unaryshift-fresh xielu-fresh""".split()
)


def read_corpus(path: Path) -> dict[str, dict[str, str]]:
    lines = (line for line in path.read_text().splitlines() if not line.startswith("#"))
    rows = {row["op"]: row for row in csv.DictReader(lines, delimiter="\t")}
    if not rows:
        raise ValueError(f"empty corpus: {path}")
    return rows


def read_search(path: Path) -> tuple[dict, str]:
    data = json.loads(path.read_text())
    operations = data.get("operations")
    if not isinstance(operations, dict) or not operations:
        raise ValueError(f"search has no operations object: {path}")
    baseline = data.get("settings", {}).get("baseline_flags")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError(f"search has no non-empty settings.baseline_flags: {path}")
    return operations, baseline


def read_validation(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    rows = data.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"validation results must be a list: {path}")
    return {row["op"]: row for row in rows}


def selected_flags(record: dict, op: str) -> str:
    try:
        flags = record["proposal"]["selection"]["flags"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{op}: missing proposal.selection.flags") from error
    if not isinstance(flags, str):
        raise ValueError(f"{op}: proposal.selection.flags is not a string")
    return flags


def classify(op: str, row: dict[str, str]) -> tuple[str, str, int, int | None, str]:
    """Return category, format, arity, full-space, reason."""
    node = row["sem_corr"]
    test_file = node.split("::", 1)[0]
    if test_file == "test_sfpu_unary.py":
        if op in UNARY_32:
            return "unary_32_exhaustive", "32-bit", 1, TWO32, "fp32 streamer"
        return "unary_bf16_exhaustive", "Float16_b", 1, TWO16, "fp32 streamer"
    if test_file == "test_sfpu_binary.py":
        if "formats:Float16_b->" in node and "bcast_dim:" not in node:
            return (
                "binary_bf16_pair_exhaustive",
                "Float16_b x Float16_b",
                2,
                TWO32,
                "binary streamer",
            )
        return (
            "binary_nonexhaustive",
            "32-bit pair or positional broadcast",
            2,
            None,
            "2^64 value space or position-coupled broadcast; class-stratify",
        )
    if test_file == "test_sfpu_ternary.py":
        return (
            "ternary_bf16_nonexhaustive",
            "Float16_b x3",
            3,
            None,
            "2^48 value space; class-stratify",
        )
    return (
        "structural_unhooked",
        "contract-specific",
        0,
        None,
        f"no pointwise exhaustive streamer for {test_file}",
    )


def plan(
    corpus: dict[str, dict[str, str]],
    search: dict,
    validation: dict[str, dict],
    baseline_flags: str,
) -> list[dict]:
    if not isinstance(baseline_flags, str) or not baseline_flags:
        raise ValueError("missing global baseline_flags")
    rows = []
    for op in sorted(search):
        if op not in corpus:
            raise ValueError(f"selected op absent from corpus: {op}")
        source = corpus[op]
        if source["kind"] != "full2x2":
            continue
        # NO_CANDIDATES rows are present in search.json for census completeness,
        # but they are not selected configurations and carry no flag profile.
        proposal = search[op].get("proposal")
        if not isinstance(proposal, dict) or not isinstance(proposal.get("selection"), dict):
            continue
        proposal_baseline = proposal.get("frozen_baseline_flags")
        if proposal_baseline != baseline_flags:
            raise ValueError(
                f"{op}: proposal.frozen_baseline_flags disagrees with "
                "settings.baseline_flags"
            )
        category, fmt, arity, full_space, reason = classify(op, source)
        prior = validation.get(op, {})
        covered = prior.get("covered")
        prior_space = prior.get("full_space")
        complete = full_space is not None and covered == full_space and prior_space == full_space
        rows.append(
            {
                "op": op,
                "category": category,
                "input_format": fmt,
                "arity": arity,
                "full_space": full_space,
                "covered": covered,
                "state": "COMPLETE" if complete else ("GAP" if full_space else "NONEXHAUSTIVE"),
                "reason": reason,
                "sem_node": source["sem_corr"],
                "hand_node": source["hand_corr"],
                "selected_flags": selected_flags(search[op], op),
                "baseline_flags": baseline_flags,
                # Tri-arm contract:
                # A vs B isolates compiler-knob semantics on identical C++.
                # B vs C evaluates the semantic uplift against handwritten code.
                "a_selected_sem_node": source["sem_corr"],
                "b_baseline_sem_node": source["sem_corr"],
                "c_baseline_hand_node": source["hand_corr"],
                "c_baseline_flags": baseline_flags,
            }
        )

    unary = [row for row in rows if row["category"].startswith("unary_")]
    unknown_32 = UNARY_32 - {row["op"] for row in unary}
    if unknown_32:
        raise ValueError(f"UNARY_32 names are not selected paired unary rows: {sorted(unknown_32)}")
    return rows


def write_outputs(out: Path, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "op", "category", "input_format", "arity", "full_space", "covered",
        "state", "reason", "sem_node", "hand_node",
    ]
    with (out / "plan.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fields, delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["category"] for row in rows)
    states = Counter(row["state"] for row in rows)
    (out / "plan.json").write_text(
        json.dumps({"schema_version": 1, "counts": dict(sorted(counts.items())),
                    "states": dict(sorted(states.items())), "rows": rows}, indent=2) + "\n"
    )

    run_categories = (
        "unary_bf16_exhaustive",
        "unary_32_exhaustive",
        "binary_bf16_pair_exhaustive",
    )
    for category in run_categories:
        selected = [row for row in rows if row["category"] == category and row["state"] == "GAP"]
        stem = category.replace("_exhaustive", "").replace("_", "-")
        (out / f"roster-{stem}.txt").write_text("".join(f"{row['op']}\n" for row in selected))
        # This is deliberately not the legacy op/sem/hand two-arm schema.  A
        # consumer must acknowledge all three profiles instead of accidentally
        # running selected-sem against selected-hand and calling that a compiler
        # correctness proof.
        with (out / f"tri-ops-{stem}.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["op", "a_selected_sem_node", "b_baseline_sem_node", "c_baseline_hand_node"]
            )
            writer.writerows(
                [
                    row["op"], row["a_selected_sem_node"],
                    row["b_baseline_sem_node"], row["c_baseline_hand_node"],
                ]
                for row in selected
            )

    feasible = [row for row in rows if row["full_space"] is not None and row["state"] == "GAP"]
    (out / "selected-flags.tsv").write_text(
        "".join(f"{row['op']}\t{row['selected_flags']}\n" for row in feasible)
    )
    (out / "baseline-flags.tsv").write_text(
        "".join(f"{row['op']}\t{row['baseline_flags']}\n" for row in feasible)
    )
    tri_fields = [
        "op", "category", "full_space", "state",
        "a_selected_sem_node", "selected_flags",
        "b_baseline_sem_node", "baseline_flags",
        "c_baseline_hand_node", "c_baseline_flags",
    ]
    with (out / "tri-profiles.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            tri_fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(feasible)
    class_rows = [row for row in rows if row["category"] in {"binary_nonexhaustive", "ternary_bf16_nonexhaustive"}]
    (out / "roster-class-stratified.txt").write_text("".join(f"{row['op']}\n" for row in class_rows))
    structural = [row for row in rows if row["category"] == "structural_unhooked"]
    (out / "roster-structural.txt").write_text("".join(f"{row['op']}\n" for row in structural))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory is not empty: {args.out}")
    search, baseline_flags = read_search(args.search)
    rows = plan(
        read_corpus(args.corpus), search, read_validation(args.validation), baseline_flags
    )
    write_outputs(args.out, rows)
    print(f"planned={len(rows)} " + " ".join(f"{k}={v}" for k, v in sorted(Counter(r['state'] for r in rows).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
