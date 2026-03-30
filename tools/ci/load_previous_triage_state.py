#!/usr/bin/env python3
"""Restore triage state from the latest successful triage workflow artifact."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


GUARDED_GH = [sys.executable, "tools/ci/guarded_gh.py"]


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def run_guarded_gh(tokens: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command_str = " ".join(shlex.quote(tok) for tok in tokens)
    return run([*GUARDED_GH, "--command", command_str], check=check)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore triage state from prior successful workflow run.")
    parser.add_argument("--repo", default="tenstorrent/tt-metal")
    parser.add_argument("--workflow", default="triage-ci.yaml")
    parser.add_argument("--artifact-name", default="triage-state")
    parser.add_argument("--state-path", default="build_ci/triage_state/ci_triage_state.json")
    parser.add_argument("--max-runs", type=int, default=20)
    return parser.parse_args()


def first_state_file(root: Path) -> Path | None:
    matches = sorted(root.rglob("ci_triage_state.json"))
    return matches[0] if matches else None


def main() -> int:
    args = parse_args()
    current_run_id = os.environ.get("GITHUB_RUN_ID", "")

    try:
        runs = run_guarded_gh(
            [
                "gh",
                "run",
                "list",
                "--repo",
                args.repo,
                "--workflow",
                args.workflow,
                "--status",
                "completed",
                "--limit",
                str(args.max_runs),
                "--json",
                "databaseId,conclusion",
            ]
        )
    except subprocess.CalledProcessError:
        print(json.dumps({"restored": False, "reason": "run_list_failed"}))
        return 0

    payload = json.loads(runs.stdout or "[]")
    if not isinstance(payload, list):
        print(json.dumps({"restored": False, "reason": "invalid_run_list_payload"}))
        return 0
    candidate_ids: list[int] = []
    for run_info in payload:
        if not isinstance(run_info, dict):
            continue
        run_id = run_info.get("databaseId")
        conclusion = str(run_info.get("conclusion", ""))
        if not isinstance(run_id, int):
            continue
        if conclusion != "success":
            continue
        if current_run_id and str(run_id) == current_run_id:
            continue
        candidate_ids.append(run_id)

    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        print(json.dumps({"restored": False, "reason": "state_already_present", "path": str(state_path)}))
        return 0

    for run_id in candidate_ids:
        with tempfile.TemporaryDirectory(prefix="triage-state-") as tmpdir:
            download = run_guarded_gh(
                [
                    "gh",
                    "run",
                    "download",
                    "--repo",
                    args.repo,
                    str(run_id),
                    "--name",
                    args.artifact_name,
                    "--dir",
                    tmpdir,
                ],
                check=False,
            )
            if download.returncode != 0:
                continue
            state_file = first_state_file(Path(tmpdir))
            if state_file is None:
                continue
            shutil.copy2(state_file, state_path)
            print(json.dumps({"restored": True, "from_run_id": run_id, "path": str(state_path)}))
            return 0

    print(json.dumps({"restored": False, "reason": "no_prior_state_found", "path": str(state_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
