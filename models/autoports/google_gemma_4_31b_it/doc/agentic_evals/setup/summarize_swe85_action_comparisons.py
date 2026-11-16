#!/usr/bin/env python3
"""Extract local alternative-token evidence from the validated diagnostic summary."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(source):
    data = json.loads(source.read_text())
    if data["status"] == "invalid":
        raise ValueError("Cannot summarize invalid diagnostic evidence")
    rows = []
    for case in data["cases"]:
        modes = {}
        for mode in ("served", "full"):
            result = case[mode]
            local = result.get("action_divergence", {})
            if local.get("status") != "verified_common_prefix":
                continue
            chosen, correct = (local["targets"][k] for k in ("chosen", "correct"))
            modes[mode] = {
                "result": result["path"],
                "runtime_config_id": result["runtime_config_id"],
                "branch_position_zero_based": local["branch_position_zero_based"],
                "absolute_next_token_index_zero_based": local["absolute_next_token_index_zero_based"],
                "same_top1_across_branch_replays": local["same_top1_across_branch_replays"],
                "chosen": chosen,
                "correct": correct,
                "correct_minus_chosen_log_probability": correct["target_log_probability"]
                - chosen["target_log_probability"],
            }
        # Include not-yet-executed action jobs using the manifest name, so a
        # partial table cannot silently look like complete failed-task coverage.
        if not modes and not case["name"].startswith("action."):
            continue
        row = {
            "name": case["name"],
            "task": case["task"],
            "call": case["call"],
            "case": case["case"],
            "case_sha256": case["case_sha256"],
            "status": "pending",
            "modes": modes,
        }
        paired = case.get("action_divergence_comparison", {})
        if paired.get("status") == "verified_common_prefix_across_precisions" and len(modes) == 2:
            row["status"] = "paired"
            row["top1_agreement_by_branch"] = paired["served_full_top1_agreement_by_branch"]
            row["full_minus_served_local_log_odds"] = (
                modes["full"]["correct_minus_chosen_log_probability"]
                - modes["served"]["correct_minus_chosen_log_probability"]
            )
        rows.append(row)
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Descriptive local token evidence only. First token divergence may be formatting rather than the semantic error. It does not establish sequence preference, task correctness, native prompt equivalence, or a passing precision policy. Whole-reply agreement and semantic controls must be reviewed alongside it.",
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
    print(json.dumps({"counts": result["counts"], "unique_tasks": result["unique_tasks"]}))


if __name__ == "__main__":
    main()
