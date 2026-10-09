# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only pilot evidence plus an in-memory upstream tool reproduction.

Never rescore a task, edit its trajectory, call a model or open a TT device.
The output contains task IDs/counts/hashes, not full conversations.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

REVISION = "17e07b1da2bbc0cadfddeea36412686e0604127b"


def credit_limit_reproduction():
    from tau2.domains.banking_knowledge.data_model import DatabaseTable, TransactionalDB
    from tau2.domains.banking_knowledge.tools import KnowledgeTools

    # Synthetic, process-local state; no task database is loaded or changed.
    db = TransactionalDB(credit_card_accounts=DatabaseTable(data={"cc_repro": {"user_id": "u_repro"}}))
    tools = KnowledgeTools(db)
    first = tools.submit_credit_limit_increase_request_7392("cc_repro", "u_repro", 1000)
    denial = tools.deny_credit_limit_increase_5848("cc_repro", "u_repro", "high_utilization")
    retry = tools.submit_credit_limit_increase_request_7392("cc_repro", "u_repro", 1000)
    statuses = sorted(row["status"] for row in db.credit_limit_increase_requests.data.values())
    reproduced = (
        "submitted successfully" in first
        and "request denied" in denial
        and retry == "Error: A similar request may already exist."
        and statuses == ["DENIED", "PENDING"]
    )
    if not reproduced:
        raise ValueError("Pinned upstream tool behavior differs from the saved failure")
    return dict(
        reproduced=True,
        first_submission_succeeded=True,
        denial_succeeded=True,
        original_request_remains_pending=True,
        retry_error=retry,
        statuses=statuses,
        original_task_score_changed=False,
        state_scope="Synthetic in-memory database only",
    )


def audit(source, results):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if (
        revision != REVISION
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ):
        raise ValueError("Requires the unchanged pinned Tau3 source")
    summary = json.loads((results / "summary.json").read_text())
    protocol = json.loads((results / "protocol.json").read_text())
    if (
        summary["state"] != "completed"
        or summary["selected"] != 12
        or summary["attempted"] != 12
        or protocol["revision"] != REVISION
    ):
        raise ValueError("Requires the complete frozen twelve-task pilot")
    report = dict(
        revision=revision,
        score_unchanged=dict(correct=summary["passed"], total=summary["selected"]),
        matched_published_reference=False,
        hardware_opened=False,
        model_calls_made=0,
        full_manual_review_completed=False,
        trials=[],
        source_sha256={},
    )
    prefix = source / "src/tau2/domains/banking_knowledge"
    for name in ("tools.py", "db_query.py", "utils.py", "data_model.py"):
        report["source_sha256"][name] = hashlib.sha256((prefix / name).read_bytes()).hexdigest()
    for row in summary["trials"]:
        root = results / row["task_id"]
        record = dict(
            task_id=row["task_id"],
            passed=row["passed"],
            termination=row.get("termination"),
            process=row["process"],
            source_sha256={},
        )
        for name in ("raw-calls.jsonl", "completed-result.json"):
            path = root / name
            if path.exists():
                record["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = root / "raw-calls.jsonl"
        calls = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
        record.update(
            calls=len(calls),
            errors=Counter(call["error_type"] for call in calls if "error_type" in call),
            length_finishes=sum(
                choice.get("finish_reason") == "length"
                for call in calls
                for choice in call.get("response", {}).get("choices", [])
            ),
        )
        path = root / "completed-result.json"
        if path.exists():
            result = json.loads(path.read_text())
            simulations = result["simulations"]
            if len(simulations) != 1 or simulations[0]["task_id"] != row["task_id"]:
                raise ValueError("Trial identity mismatch")
            simulation = simulations[0]
            reward = simulation.get("reward_info") or {}
            messages = simulation["messages"]
            record.update(
                reward_basis=reward.get("reward_basis"),
                reward_breakdown=reward.get("reward_breakdown"),
                tool_messages=sum(m["role"] == "tool" for m in messages),
                tool_error_flags=sum(m.get("error") is True for m in messages if m["role"] == "tool"),
                tool_output_error_prefixes=[
                    i
                    for i, m in enumerate(messages)
                    if m["role"] == "tool"
                    and isinstance(m.get("content"), str)
                    and m["content"].lstrip().startswith("Error:")
                ],
                final_message_role=messages[-1]["role"] if messages else None,
            )
        report["trials"].append(record)
    report["credit_limit_tool_reproduction"] = credit_limit_reproduction()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source, args.results)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("score_unchanged", "credit_limit_tool_reproduction")}))
