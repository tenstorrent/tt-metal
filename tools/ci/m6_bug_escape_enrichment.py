#!/usr/bin/env python3
"""M6 bug-escape enrichment for CI triage issues in test repo."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ISSUE_REPO_TEST = "ebanerjeeTT/issue_dump"
PRIMARY_REPO = "tenstorrent/tt-metal"
LAYER_ORDER = {"llk": 1, "metalium": 2, "ttnn": 3, "models": 4, "infra": 5, "unknown": 99}


def now_utc() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=merged_env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def run_guarded_gh(tokens: list[str], *, github_token: str) -> subprocess.CompletedProcess[str]:
    cmd = " ".join(shlex.quote(tok) for tok in tokens)
    return run(
        [sys.executable, "tools/ci/guarded_gh.py", "--command", cmd],
        env={"GITHUB_TOKEN": github_token},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bug-escape enrichment records for closed CI triage issues.")
    p.add_argument("--state-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--summary-md", required=True)
    p.add_argument("--issue-repo", default=ISSUE_REPO_TEST)
    return p.parse_args()


def infer_layer_from_text(text: str) -> str:
    t = text.lower()
    if "llk" in t:
        return "llk"
    if "metalium" in t or "tt_metal" in t or "llrt" in t:
        return "metalium"
    if "ttnn" in t:
        return "ttnn"
    if "model" in t:
        return "models"
    if "infra" in t or "pipeline" in t:
        return "infra"
    return "unknown"


def infer_layer_from_paths(paths: list[str]) -> str:
    joined = "\n".join(paths)
    return infer_layer_from_text(joined)


def classify_escape_type(*, failure_layer: str, fix_layer: str) -> tuple[str, str]:
    if failure_layer == "unknown" or fix_layer == "unknown":
        return ("unknown", "low")
    if LAYER_ORDER.get(fix_layer, 99) < LAYER_ORDER.get(failure_layer, 99):
        return ("layer_escape_lower_to_higher", "high")
    if failure_layer != fix_layer:
        return ("gate_coverage_gap", "medium")
    return ("unknown", "medium")


def issue_numbers_from_state(path: Path) -> list[int]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    out: list[int] = []
    seen: set[int] = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        numbers = item.get("issue_numbers", [])
        if not isinstance(numbers, list):
            continue
        for n in numbers:
            try:
                num = int(n)
            except Exception:
                continue
            if num <= 0 or num in seen:
                continue
            seen.add(num)
            out.append(num)
    return out


def issue_view(repo: str, issue_number: int, token: str) -> dict[str, Any]:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "view",
            "--repo",
            repo,
            str(issue_number),
            "--json",
            "state,title,body,url,closedAt,labels,closedByPullRequestsReferences",
        ],
        github_token=token,
    )
    payload = json.loads(proc.stdout or "{}")
    return payload if isinstance(payload, dict) else {}


def pr_view(repo: str, pr_number: int, token: str) -> dict[str, Any]:
    proc = run_guarded_gh(
        [
            "gh",
            "pr",
            "view",
            "--repo",
            repo,
            str(pr_number),
            "--json",
            "number,url,mergeCommit,files",
        ],
        github_token=token,
    )
    payload = json.loads(proc.stdout or "{}")
    return payload if isinstance(payload, dict) else {}


def summarize(events: list[dict[str, Any]]) -> str:
    high = sum(1 for e in events if e.get("correlation_confidence") == "high")
    medium = sum(1 for e in events if e.get("correlation_confidence") == "medium")
    low = sum(1 for e in events if e.get("correlation_confidence") == "low")
    lines = [
        "## M6 Bug Escape Enrichment",
        "",
        f"- Events captured: {len(events)}",
        f"- High confidence: {high}",
        f"- Medium confidence: {medium}",
        f"- Low confidence: {low}",
        "",
        "## Sample",
    ]
    for event in events[:5]:
        lines.append(
            f"- issue #{event.get('issue_number')} -> PR {event.get('fix_pr_number')} "
            f"| {event.get('failure_layer')} -> {event.get('fix_layer')} "
            f"| {event.get('escape_type')} ({event.get('correlation_confidence')})"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    token = os.environ.get("ISSUE_WRITE_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("ISSUE_WRITE_TOKEN or GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    issues = issue_numbers_from_state(Path(args.state_json))
    events: list[dict[str, Any]] = []
    for issue_number in issues:
        issue = issue_view(args.issue_repo, issue_number, token)
        if str(issue.get("state", "")).upper() != "CLOSED":
            continue
        refs = issue.get("closedByPullRequestsReferences", [])
        if not isinstance(refs, list) or not refs:
            continue
        pr_ref = refs[0] if isinstance(refs[0], dict) else {}
        pr_number = int(pr_ref.get("number", 0) or 0)
        if pr_number <= 0:
            continue
        pr = pr_view(PRIMARY_REPO, pr_number, token)
        files = pr.get("files", []) if isinstance(pr, dict) else []
        paths: list[str] = []
        if isinstance(files, list):
            for f in files:
                if not isinstance(f, dict):
                    continue
                path = str(f.get("path", "")).strip()
                if path:
                    paths.append(path)
        failure_layer = infer_layer_from_text(f"{issue.get('title','')}\n{issue.get('body','')}")
        fix_layer = infer_layer_from_paths(paths)
        escape_type, confidence = classify_escape_type(failure_layer=failure_layer, fix_layer=fix_layer)
        merge_commit = pr.get("mergeCommit", {}) if isinstance(pr, dict) else {}
        fix_sha = str(merge_commit.get("oid", "")).strip() if isinstance(merge_commit, dict) else ""
        events.append(
            {
                "fingerprint": "",
                "issue_number": issue_number,
                "disable_pr_number": 0,
                "fix_pr_number": pr_number,
                "fix_commit_sha": fix_sha,
                "problem_surface": {
                    "layer": failure_layer,
                    "component": "",
                    "source": "issue_text",
                },
                "fix_surface": {
                    "layer": fix_layer,
                    "component": "",
                    "source": "changed_files",
                },
                "failure_layer": failure_layer,
                "fix_layer": fix_layer,
                "correlation_method": "path_overlap" if fix_layer != "unknown" else "semantic_match",
                "correlation_confidence": confidence,
                "escape_type": escape_type,
                "shift_left_target": "lower_layer_suite"
                if escape_type == "layer_escape_lower_to_higher"
                else "merge_gate",
                "explanation": f"Issue resolved by PR #{pr_number}; inferred {failure_layer} -> {fix_layer}.",
                "captured_at_utc": now_utc(),
                "issue_url": str(issue.get("url", "")).strip(),
                "fix_pr_url": str(pr.get("url", "")).strip(),
            }
        )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")
    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summarize(events), encoding="utf-8")
    print(json.dumps({"event_count": len(events)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
