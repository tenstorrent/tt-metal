#!/usr/bin/env python3
"""Render M4 issue/slack result JSON as markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_summary(data: dict[str, Any]) -> str:
    created = data.get("created", [])
    skipped = data.get("skipped", [])
    lines: list[str] = []
    lines.append("")
    lines.append("# M4 Issue + Slack Summary")
    lines.append("")
    lines.append(f"- Candidates total: {data.get('candidate_count', 0)}")
    lines.append(f"- Candidates processed: {data.get('processed_count', 0)}")
    lines.append(f"- Fresh agent calls: {data.get('fresh_agent_calls', 0)}")
    lines.append(f"- Fresh agent reviews: {data.get('fresh_agent_reviews', 0)}")
    lines.append(f"- Issues created: {len(created) if isinstance(created, list) else 0}")
    lines.append(f"- Skipped: {len(skipped) if isinstance(skipped, list) else 0}")
    if isinstance(created, list) and created:
        lines.append("")
        lines.append("## Created")
        for item in created:
            if not isinstance(item, dict):
                continue
            issue_url = item.get("issue_url", "(missing issue url)")
            workflow_name = item.get("workflow_name", "")
            job_name = item.get("job_name", "")
            lines.append(f"- {issue_url} | {workflow_name} / {job_name}")
            decision = item.get("agent_decision", {})
            if isinstance(decision, dict):
                signature = str(decision.get("signature", "")).strip()
                reason = str(decision.get("reason", "")).strip()
                if signature:
                    if len(signature) > 160:
                        signature = signature[:160] + "..."
                    lines.append(f"  - Why issue was created: deterministic signature `{signature}`")
                elif reason:
                    if len(reason) > 220:
                        reason = reason[:220] + "..."
                    lines.append(f"  - Why issue was created: {reason}")
            owner_selection = item.get("owner_selection", {})
            if isinstance(owner_selection, dict):
                selected = owner_selection.get("selected_owner_count", 0)
                candidates = owner_selection.get("candidate_owner_count", 0)
                source_counts = owner_selection.get("source_counts", {})
                if isinstance(source_counts, dict):
                    lines.append(
                        "  - Why people were pinged: "
                        f"selected {selected} of {candidates} candidates from "
                        f"codeowners={source_counts.get('codeowners', 0)}, "
                        f"commit_history={source_counts.get('commit_history', 0)}, "
                        f"auto_triage={source_counts.get('auto_triage', 0)}, "
                        f"job_owners={source_counts.get('job_owners', 0)}"
                    )
            unresolved = item.get("unresolved_owner_handles", [])
            if isinstance(unresolved, list) and unresolved:
                redacted = ", ".join(f"@{h}" for h in unresolved[:8])
                lines.append(f"  - Unresolved GitHub handles: {redacted}")
            auto_triage_selection = item.get("auto_triage_selection", {})
            if isinstance(auto_triage_selection, dict):
                if auto_triage_selection.get("used") and auto_triage_selection.get("permalinks"):
                    permalink = str(auto_triage_selection.get("permalinks", [""])[0]).strip()
                    if permalink:
                        lines.append(f"  - Auto-triage evidence used: {permalink}")
                else:
                    reason = str(auto_triage_selection.get("reason", "")).strip() or "no relevant matches"
                    lines.append(f"  - Auto-triage evidence not used: {reason}")
    if isinstance(skipped, list) and skipped:
        lines.append("")
        lines.append("## Top skip reasons")
        counts: dict[str, int] = {}
        for item in skipped:
            if isinstance(item, dict):
                reason = str(item.get("reason", "unknown"))
            else:
                reason = "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        for reason, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render M4 markdown summary from JSON output.")
    parser.add_argument("--input-json", required=True, help="Path to m4_issue_and_slack_result.json")
    args = parser.parse_args()
    path = Path(args.input_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Expected JSON object payload")
    print(render_summary(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
