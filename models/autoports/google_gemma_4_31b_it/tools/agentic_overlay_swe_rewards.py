#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply verified offline grading evidence without changing raw collection files."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def overlay(raw, recovery_dir, job_dir, task_dir):
    result = copy.deepcopy(raw)
    summary = json.loads((recovery_dir / "recovery-summary.json").read_text())
    applied = []
    seen = set()
    for evidence in summary["rows"]:
        if evidence["status"] != "parsed":
            continue
        task, trial = evidence["task"], evidence["trial"]
        require((task, trial) not in seen, "Duplicate recovery identity")
        seen.add((task, trial))
        require(Path(task).name == task and Path(trial).name == trial, "Unsafe identity")
        source = (job_dir / trial / "result.json").resolve()
        require(Path(evidence["raw_result"]).resolve() == source, "Recovery belongs to a different job")
        native = json.loads(source.read_text())
        names = {native["task_name"], Path(native.get("config", {}).get("task", {}).get("path", "")).name}
        require(task in names or "swe-bench/swebench-verified__" + task in names, "Native task mismatch")
        require(native["trial_name"] == trial and native.get("finished_at"), "Incomplete or mismatched trial")
        require(native.get("verifier_result") == evidence["raw_reward"], "Raw reward changed")
        require(
            file_sha(source.parent / "verifier/test-stdout.txt") == evidence["raw_log_sha256"], "Raw log hash mismatch"
        )
        require(
            file_sha(task_dir / task / "tests/config.json") == evidence["config_sha256"], "Task config hash mismatch"
        )
        directory = recovery_dir / task
        require(
            json.loads((directory / "provenance.json").read_text()) == evidence, "Recovery summary/provenance mismatch"
        )
        require(file_sha(directory / "parser.py") == evidence["parser_sha256"], "Parser hash mismatch")
        require(
            file_sha(directory / "test-output.original.txt") == evidence["test_output_sha256"],
            "Test output hash mismatch",
        )
        report = json.loads((directory / "report.json").read_text())
        require(set(report) == {task}, "Recovered report identity mismatch")
        resolved = report[task].get("resolved")
        require(
            type(resolved) is bool and report[task].get("patch_successfully_applied") is True,
            "Invalid recovered status",
        )
        require(
            type(evidence["recovered_reward"]) is int and evidence["recovered_reward"] == int(resolved),
            "Recovered reward mismatch",
        )
        require(evidence["returncode"] == (0 if resolved else 1), "Parser exit/status mismatch")
        matches = [row for row in result["attempts"] if row.get("task") == task and row.get("trial_name") == trial]
        require(len(matches) == 1, "Recovery absent or ambiguous in collection")
        attempt = matches[0]
        require(Path(attempt["source_result"]).resolve() == source, "Collected run identity mismatch")
        require(
            attempt.get("rewards") == ((native.get("verifier_result") or {}).get("rewards") or {}),
            "Collected raw reward mismatch",
        )
        annotation = dict(
            raw_solved=attempt.get("solved"),
            raw_rewards=attempt.get("rewards"),
            resolved=resolved,
            evidence=str(directory / "provenance.json"),
            report_sha256=file_sha(directory / "report.json"),
            no_model_rerun=True,
            method="Generic offline verifier grading of preserved original test output",
        )
        for row in result["attempts"] + result["tasks"]:
            if row.get("task") == task and row.get("trial_name") == trial:
                row["solved"] = resolved
                row["verifier_recovery"] = annotation
        applied.append(dict(task=task, trial=trial, **annotation))
    score = result["score"]
    score["solved"] = sum(row.get("solved") is True for row in result["tasks"])
    score["complete"] = all(row.get("solved") is not None for row in result["tasks"])
    score["unresolved"] = [row["task"] for row in result["tasks"] if row.get("solved") is None]
    n, success, z = score["n"], score["solved"], 1.959963984540054
    center = (success / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(success / n * (1 - success / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    score["interval_95"] = [max(0, center - radius), min(1, center + radius)] if score["complete"] else None
    result["raw_score"] = raw["score"]
    result["verifier_recovery"] = dict(
        source=str(recovery_dir),
        source_sha256=file_sha(recovery_dir / "recovery-summary.json"),
        generic_fix=summary.get("generic_fix"),
        applied=applied,
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("collected", "verifier-recovery", "job-dir", "task-dir", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    require(args.out.resolve() != args.collected.resolve(), "Output must preserve raw collection")
    require(not args.out.exists(), "Refusing to overwrite existing overlay")
    result = overlay(json.loads(args.collected.read_text()), args.verifier_recovery, args.job_dir, args.task_dir)
    result["raw_collection"] = dict(path=str(args.collected), sha256=file_sha(args.collected))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["score"]))


if __name__ == "__main__":
    main()
