#!/usr/bin/env python3
"""Collect complete paired action sequences without making correctness judgments."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(source):
    raw = source.read_bytes()
    data = json.loads(raw)
    if data["status"] == "invalid":
        raise ValueError("Invalid source diagnostic summary")
    rows = []
    for case in data["cases"]:
        if not case["name"].startswith("action."):
            continue
        row = {k: case[k] for k in ("name", "task", "call", "case", "case_sha256")}
        row["status"] = "pending"
        if case.get("comparison_status") == "verified":
            branches = {}
            evidence = {}
            for mode in ("served", "full"):
                result = case[mode]
                if result["status"] != "complete":
                    raise ValueError("Verified comparison contains incomplete output")
                path = Path(result["path"])
                evidence[mode] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "runtime_config_id": result["runtime_config_id"],
                }
            for name in ("chosen", "correct"):
                pair = case["comparison"][name]
                count = pair["tokens"]
                if count <= 0 or any(case[m]["branches"][name]["tokens"] != count for m in ("served", "full")):
                    raise ValueError("Unaligned or empty branch")
                served = pair["served_sequence_log_probability"]
                full = pair["full_sequence_log_probability"]
                branches[name] = {
                    "tokens": count,
                    "served_full_top1_agreement": pair["top1_agreement"],
                    "served_sequence_log_probability": served,
                    "full_sequence_log_probability": full,
                    "full_minus_served_mean_token_log_probability": (full - served) / count,
                }
            row.update(status="paired", branches=branches, evidence=evidence)
        rows.append(row)
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_note": "Source summary is mutable; completed output hashes bind each paired row.",
        "scope": "Each branch is compared across precisions on identical forced tokens. Chosen and correct branches may differ in length, shell wrapper, and preceding reasoning consistency; their sequence sums or mean log probabilities do not establish task-level preference. These controls retain original reasoning and do not rule out precision effects on earlier decisions. No score, anomaly verdict, or precision acceptance is inferred.",
        "counts": dict(Counter(r["status"] for r in rows)),
        "unique_tasks": len({r["task"] for r in rows}),
        "actions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.summary)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({k: result[k] for k in ("counts", "unique_tasks")}))


if __name__ == "__main__":
    main()
