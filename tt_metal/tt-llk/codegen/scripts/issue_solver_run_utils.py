#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Small utilities for issue-solver runs.

These helpers stay intentionally boring: prompts can call them from Bash, and
tests can exercise the behavior without involving Claude.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def cmd_upsert_runs_jsonl(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    runs_jsonl = Path(args.runs_jsonl)
    run_path = log_dir / "run.json"
    if not run_path.exists():
        raise SystemExit(f"run.json not found: {run_path}")
    run = json.loads(run_path.read_text())
    run_id = run.get("run_id")
    if not run_id:
        raise SystemExit(f"run.json missing run_id: {run_path}")

    lock_path = runs_jsonl.parent / ".runs.jsonl.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            os.chmod(lock_path, 0o664)
        except OSError:
            pass
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_jsonl(runs_jsonl)
        replaced = False
        for i, row in enumerate(rows):
            if row.get("run_id") == run_id:
                rows[i] = run
                replaced = True
                break
        if not replaced:
            rows.append(run)

        payload = "".join(json.dumps(row) + "\n" for row in rows)
        _atomic_write(runs_jsonl, payload)
    action = "updated" if replaced else "appended"
    print(f"runs-jsonl-{action}: {run_id}")


def cmd_autodebug(args: argparse.Namespace) -> None:
    """Run the installed inspection-only launcher within the current worker retry."""
    log_dir = Path(args.log_dir).resolve()
    run = json.loads((log_dir / "run.json").read_text())
    package = (run.get("solver_plugins") or {}).get("tt-autodebug")
    if not package:
        raise SystemExit("tt-autodebug is not configured for this run")
    launcher = Path(package["path"]) / "skills/autodebug/scripts/autodebug.sh"
    worktree = Path(args.worktree).resolve()
    report = worktree / "AUTODEBUG.md"
    if report.exists() or report.is_symlink():
        raise SystemExit("refusing to overwrite pre-existing AUTODEBUG.md")
    directory = Path(tempfile.mkdtemp(prefix="autodebug-", dir=log_dir))
    # Pin child identity before launching so accounting survives a timeout.
    # The installed launcher remains authoritative for the investigation prompt;
    # this thin executable adds CLI identity/budget flags it does not expose.
    claude = shutil.which("claude")
    if not claude:
        raise SystemExit("claude executable not found")
    session_id = str(uuid.uuid4())
    state_path = log_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("RUN_ID") and state["RUN_ID"] != run["run_id"]:
        raise SystemExit("state belongs to another run")
    cap_raw = os.environ.get("CODEGEN_AUTODEBUG_BUDGET_USD")
    cap = float(cap_raw) if cap_raw is not None else None
    if cap is not None and (not 0 <= cap < float("inf")):
        raise SystemExit("CODEGEN_AUTODEBUG_BUDGET_USD must be finite and nonnegative")
    registry_path = log_dir / "session_registry.json"
    with (log_dir / ".session_registry.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = (
            json.loads(registry_path.read_text())
            if registry_path.exists()
            else {
                "schema": "issue-solver.session-registry",
                "version": 1,
                "run_id": run["run_id"],
                "sessions": [],
            }
        )
        if registry.get("run_id") != run["run_id"]:
            raise SystemExit("session registry belongs to another run")
        if (
            registry.get("schema") != "issue-solver.session-registry"
            or registry.get("version") != 1
            or not isinstance(registry.get("sessions"), list)
        ):
            raise SystemExit("invalid session registry schema")
        if any(not isinstance(entry, dict) for entry in registry["sessions"]):
            raise SystemExit("invalid session registry entry")
        ids = [entry.get("session_id") for entry in registry["sessions"]]
        if (
            any(not isinstance(value, str) for value in ids)
            or len(set(ids)) != len(ids)
            or session_id in ids
        ):
            raise SystemExit("duplicate or invalid session identity")
        try:
            allocations = [
                float(entry.get("allocated_budget_usd") or 0)
                for entry in registry["sessions"]
            ]
        except (ValueError, TypeError):
            raise SystemExit("invalid allocated budget")
        if any(not 0 <= value < float("inf") for value in allocations):
            raise SystemExit("invalid allocated budget")
        used = sum(allocations)
        remaining = max(0, cap - used) if cap is not None else None
        if remaining is not None and remaining < 0.01:
            raise SystemExit("AutoDebug budget is exhausted; use existing evidence")
        registry["sessions"].append(
            {
                "session_id": session_id,
                "project_cwd": str(worktree),
                "parent_session_id": state.get("SESSION_ID"),
                "kind": "autodebug",
                "allocated_budget_usd": remaining,
                "artifact_dir": str(directory),
            }
        )
        _atomic_write(registry_path, json.dumps(registry, indent=2) + "\n")
    shim_dir = directory / "bin"
    shim_dir.mkdir()
    argv = [claude, "--session-id", session_id]
    if remaining is not None:
        argv += ["--max-budget-usd", str(remaining)]
    shim = shim_dir / "claude"
    shim.write_text(
        "#!/bin/sh\nexport PATH="
        + shlex.quote(os.environ.get("PATH", ""))
        + "\nexec "
        + shlex.join(argv)
        + ' "$@"\n'
    )
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
    # A fresh Claude process must not inherit the parent's nesting sentinel.
    env.pop("CLAUDECODE", None)
    env.pop("AUTODEBUG_CLAUDE_MODEL", None)
    command = ["bash", str(launcher), "--agent", "claude", "--effort", "high"]
    if env.get("CODEGEN_MODEL") and not env.get("ANTHROPIC_BASE_URL"):
        command += ["--model", env["CODEGEN_MODEL"]]
    command += ["--", args.problem]
    with (directory / "launcher.log").open("w") as output:
        proc = subprocess.Popen(
            command,
            cwd=worktree,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=args.timeout)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if report.is_file() and not report.is_symlink():
                shutil.move(str(report), str(directory / "AUTODEBUG.md"))
            exporter = Path(__file__).with_name("extract_run_transcripts.py")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(exporter),
                        "--log-dir",
                        str(directory),
                        "--session-id",
                        session_id,
                        "--project-cwd",
                        str(worktree),
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                output.write(f"Transcript export unavailable: {type(exc).__name__}\n")
            print(directory)
    if code or not (directory / "AUTODEBUG.md").is_file():
        raise SystemExit(f"AutoDebug did not produce a successful report: {directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    upsert = sub.add_parser(
        "upsert-runs-jsonl", help="Insert or update a run in runs.jsonl"
    )
    upsert.add_argument("--log-dir", required=True)
    upsert.add_argument("--runs-jsonl", required=True)
    upsert.set_defaults(func=cmd_upsert_runs_jsonl)

    debug = sub.add_parser(
        "autodebug", help="Bounded installed AutoDebug investigation"
    )
    debug.add_argument("--log-dir", required=True)
    debug.add_argument("--worktree", required=True)
    debug.add_argument("--problem", required=True)
    debug.add_argument("--timeout", type=int, default=1800)
    debug.set_defaults(func=cmd_autodebug)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
