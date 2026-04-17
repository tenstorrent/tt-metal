# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Atomic writer for the Activity-Monitor-aware run.json.

Implements the live-update contract defined by:
  - /proj_sw/user_dev/llk_code_gen/dashboard/GEN_MONITOR_FIELDS.md
  - /proj_sw/user_dev/llk_code_gen/dashboard/RUN_JSON_SPEC.md

Every subcommand updates <LOG_DIR>/run.json by writing a temp file in the
same directory and atomically renaming it into place so the dashboard never
reads a half-written file.

Subcommands:
    init         Write the initial run.json at run start (status=running,
                 first step_history entry in_progress).
    advance      Transition from the current step to a new step. Closes out
                 the in-flight step_history entry and appends a new one.
    message      Mid-step update of current_step_message (and optionally the
                 current in-flight step_history entry's message).
    phase-start  Mark a per_phase[] entry as started (start_time, name).
    phase-test   Set per_phase[].test_result to "running" or "fixing" while
                 the simulator / debugger is live.
    phase-end    Finalize a per_phase[] entry (end_time, duration, test
                 result, compile_errors, test_details).
    failure      Append an entry to the top-level failures[] array.
    metric       Patch arbitrary top-level scalar fields (compilation_attempts,
                 debug_cycles, tests_total, tests_passed, etc.).
    finalize     Close out the last step_history entry, set end_time, flip
                 status to a terminal value, merge in any remaining summary
                 fields passed via --patch-json.

Every subcommand is idempotent in the sense that running it twice with the
same arguments produces the same final document (modulo timestamps the
caller passes explicitly).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Low-level IO
# --------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _duration_seconds(started: str, ended: str) -> int:
    return int((_parse_iso(ended) - _parse_iso(started)).total_seconds())


def _run_json_path(log_dir: Path) -> Path:
    return log_dir / "run.json"


def _load(log_dir: Path) -> dict[str, Any]:
    path = _run_json_path(log_dir)
    if not path.exists():
        raise SystemExit(f"run.json not found at {path} — call 'init' first")
    return json.loads(path.read_text())


def _atomic_write(log_dir: Path, doc: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = _run_json_path(log_dir)
    fd, tmp = tempfile.mkstemp(prefix=".run.json.", suffix=".tmp", dir=log_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        # mkstemp creates files with 0o600, which locks the dashboard (running as
        # a different user) out of reading run.json. Relax to 0o664 so the shared
        # group — and anything else — can read the live status.
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_arg(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


# --------------------------------------------------------------------------
# Subcommand: init
# --------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    start_time = args.start_time or _utcnow()

    doc: dict[str, Any] = {
        "run_id": args.run_id,
        "kernel": args.kernel,
        "kernel_type": args.kernel_type,
        "arch": args.arch,
        "reference_arch": args.reference_arch,
        "reference_file": args.reference_file,
        "generated_file": args.generated_file,
        "start_time": start_time,
        "end_time": None,
        "status": "running",
        "obstacle": None,
        "prompt": args.prompt,
        "batch_id": args.batch_id,
        "model": args.model,
        "run_type": args.run_type,
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "num_turns": 0,
        "solver_state": None,
        "cost_usd": 0,
        "duration_seconds": 0,
        "log_dir": args.log_dir,
        "phases_total": args.phases_total,
        "phases_completed": 0,
        "compilation_attempts": 0,
        "debug_cycles": 0,
        "tests_total": 0,
        "tests_passed": 0,
        "lines_generated": 0,
        "tests_generated": False,
        "prettified": False,
        "formatted": False,
        "optimized": False,
        "optimization_type": "none",
        "formats_tested": [],
        "formats_excluded": {},
        "failures": [],
        "per_phase": [],
        "tokens": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "total": 0,
            "cost_usd": 0,
        },
        "agents": [],
        # Activity Monitor live-state fields.
        "current_step": args.first_step,
        "current_step_started": start_time,
        "current_step_message": args.first_message,
        "steps_completed": [],
        "step_history": [
            {
                "step": args.first_step,
                "started": start_time,
                "ended": None,
                "duration_seconds": None,
                "result": "in_progress",
                "message": args.first_message,
            }
        ],
    }

    pipeline_steps = _json_arg(args.pipeline_steps, None)
    if pipeline_steps is not None:
        doc["pipeline_steps"] = pipeline_steps

    issue = _json_arg(args.issue, None)
    if issue is not None:
        doc["issue"] = issue

    # Apply any additional free-form patch.
    patch = _json_arg(args.patch_json, {})
    doc.update(patch)

    _atomic_write(log_dir, doc)
    print(f"init: wrote {_run_json_path(log_dir)}")


# --------------------------------------------------------------------------
# Subcommand: advance
# --------------------------------------------------------------------------


def cmd_advance(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    now = args.now or _utcnow()

    history = doc.setdefault("step_history", [])
    if history and history[-1].get("result") == "in_progress":
        last = history[-1]
        last["ended"] = now
        last["duration_seconds"] = _duration_seconds(last["started"], now)
        last["result"] = args.prev_result
        if args.prev_message:
            last["message"] = args.prev_message

        prev_step_id = last["step"]
        completed = doc.setdefault("steps_completed", [])
        if prev_step_id not in completed:
            completed.append(prev_step_id)

    history.append(
        {
            "step": args.new_step,
            "started": now,
            "ended": None,
            "duration_seconds": None,
            "result": "in_progress",
            "message": args.new_message,
        }
    )

    doc["current_step"] = args.new_step
    doc["current_step_started"] = now
    doc["current_step_message"] = args.new_message

    if args.agent:
        agents = doc.setdefault("agents", [])
        if args.agent not in agents:
            agents.append(args.agent)

    _atomic_write(log_dir, doc)
    print(f"advance: {args.new_step} ({args.prev_result} closed prior)")


# --------------------------------------------------------------------------
# Subcommand: message (mid-step progress)
# --------------------------------------------------------------------------


def cmd_message(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    doc["current_step_message"] = args.message

    history = doc.get("step_history") or []
    if history and history[-1].get("result") == "in_progress":
        history[-1]["message"] = args.message

    _atomic_write(log_dir, doc)
    print(f"message: {args.message[:60]}")


# --------------------------------------------------------------------------
# Per-phase helpers
# --------------------------------------------------------------------------


def _phase_entry(doc: dict[str, Any], phase_num: int) -> dict[str, Any]:
    per_phase = doc.setdefault("per_phase", [])
    for entry in per_phase:
        if entry.get("phase") == phase_num:
            return entry
    entry = {
        "phase": phase_num,
        "name": "",
        "compilation_attempts": 0,
        "debug_cycles": 0,
        "test_result": "pending",
        "compile_errors": [],
        "test_details": None,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
    }
    per_phase.append(entry)
    per_phase.sort(key=lambda e: e.get("phase", 0))
    return entry


def cmd_phase_start(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    now = args.now or _utcnow()
    entry = _phase_entry(doc, args.phase)
    if args.name:
        entry["name"] = args.name
    entry["start_time"] = now
    entry["test_result"] = "pending"
    entry["end_time"] = None
    entry["duration_seconds"] = None
    _atomic_write(log_dir, doc)
    print(f"phase-start: phase {args.phase} ({args.name or entry.get('name')})")


def cmd_phase_test(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    entry = _phase_entry(doc, args.phase)
    entry["test_result"] = args.state  # "running" | "fixing"
    if args.details is not None:
        entry["test_details"] = args.details
    _atomic_write(log_dir, doc)
    print(f"phase-test: phase {args.phase} -> {args.state}")


def cmd_phase_end(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    now = args.now or _utcnow()
    entry = _phase_entry(doc, args.phase)
    entry["end_time"] = now
    if entry.get("start_time"):
        entry["duration_seconds"] = _duration_seconds(entry["start_time"], now)
    entry["test_result"] = args.test_result  # passed | failed | skipped

    if args.compilation_attempts is not None:
        entry["compilation_attempts"] = args.compilation_attempts
    if args.debug_cycles is not None:
        entry["debug_cycles"] = args.debug_cycles
    if args.test_details is not None:
        entry["test_details"] = args.test_details
    compile_errors = _json_arg(args.compile_errors_json, None)
    if compile_errors is not None:
        entry["compile_errors"] = compile_errors

    if args.test_result == "passed":
        doc["phases_completed"] = (doc.get("phases_completed") or 0) + 1

    _atomic_write(log_dir, doc)
    print(f"phase-end: phase {args.phase} -> {args.test_result}")


# --------------------------------------------------------------------------
# Failures / metrics patching
# --------------------------------------------------------------------------


def cmd_failure(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    failures = doc.setdefault("failures", [])
    failures.append(
        {
            "step": args.step,
            "agent": args.agent,
            "type": args.type,
            "message": args.message,
            "resolved": args.resolved.lower() == "true",
        }
    )
    _atomic_write(log_dir, doc)
    print(f"failure: {args.type} @ {args.step}")


def cmd_metric(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    patch = _json_arg(args.patch_json, {})
    for k, v in patch.items():
        doc[k] = v
    _atomic_write(log_dir, doc)
    print(f"metric: patched {sorted(patch)}")


# --------------------------------------------------------------------------
# Subcommand: finalize
# --------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)
    now = args.end_time or _utcnow()

    history = doc.setdefault("step_history", [])
    if history and history[-1].get("result") == "in_progress":
        last = history[-1]
        last["ended"] = now
        last["duration_seconds"] = _duration_seconds(last["started"], now)
        last["result"] = args.final_result
        if args.final_message:
            last["message"] = args.final_message
        completed = doc.setdefault("steps_completed", [])
        if last["step"] not in completed:
            completed.append(last["step"])

    doc["end_time"] = now
    if doc.get("start_time"):
        doc["duration_seconds"] = _duration_seconds(doc["start_time"], now)
    doc["status"] = args.status  # success | compiled | failed | skipped
    doc["current_step_message"] = (
        args.final_message or doc.get("current_step_message") or ""
    )

    patch = _json_arg(args.patch_json, {})
    for k, v in patch.items():
        doc[k] = v

    # Apply typed --solver-state last so it cannot be silently overridden by
    # --patch-json (argparse choices are otherwise bypassed via that escape hatch).
    if args.solver_state is not None:
        doc["solver_state"] = args.solver_state

    _atomic_write(log_dir, doc)
    print(f"finalize: status={args.status}")


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-dir", required=True, help="Path to the run's LOG_DIR")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # init -----------------------------------------------------------------
    init = sub.add_parser("init", help="Create the initial run.json (status=running)")
    _add_common(init)
    init.add_argument("--run-id", required=True)
    init.add_argument("--kernel", required=True)
    init.add_argument("--kernel-type", default="")
    init.add_argument("--arch", required=True)
    init.add_argument("--reference-arch", default=None)
    init.add_argument("--reference-file", default=None)
    init.add_argument("--generated-file", default=None)
    init.add_argument("--start-time", default=None, help="ISO 8601; defaults to now")
    init.add_argument(
        "--first-step", required=True, help="Pipeline step ID of the first step"
    )
    init.add_argument("--first-message", required=True)
    init.add_argument("--prompt", default="")
    init.add_argument("--batch-id", default=None)
    init.add_argument("--model", default="")
    init.add_argument("--run-type", default="manual")
    init.add_argument("--git-commit", default="unknown")
    init.add_argument("--git-branch", default="")
    init.add_argument("--phases-total", type=int, default=0)
    init.add_argument(
        "--pipeline-steps", default=None, help="JSON array of {id,name,desc} objects"
    )
    init.add_argument("--issue", default=None, help="JSON object for issue-solver runs")
    init.add_argument(
        "--patch-json",
        default=None,
        help="JSON object merged into the doc after defaults",
    )
    init.set_defaults(func=cmd_init)

    # advance --------------------------------------------------------------
    adv = sub.add_parser("advance", help="Transition to a new pipeline step")
    _add_common(adv)
    adv.add_argument("--new-step", required=True)
    adv.add_argument("--new-message", required=True)
    adv.add_argument(
        "--prev-result",
        required=True,
        choices=["success", "compile_error", "test_failure"],
    )
    adv.add_argument("--prev-message", default=None)
    adv.add_argument(
        "--agent", default=None, help="Agent ID to append to agents[] if not present"
    )
    adv.add_argument(
        "--now", default=None, help="ISO 8601 timestamp to use (defaults to now)"
    )
    adv.set_defaults(func=cmd_advance)

    # message --------------------------------------------------------------
    msg = sub.add_parser("message", help="Update current_step_message mid-step")
    _add_common(msg)
    msg.add_argument("--message", required=True)
    msg.set_defaults(func=cmd_message)

    # phase-start ----------------------------------------------------------
    ps = sub.add_parser("phase-start", help="Mark a per_phase entry as started")
    _add_common(ps)
    ps.add_argument("--phase", type=int, required=True)
    ps.add_argument("--name", default="")
    ps.add_argument("--now", default=None)
    ps.set_defaults(func=cmd_phase_start)

    # phase-test -----------------------------------------------------------
    pt = sub.add_parser(
        "phase-test", help="Set per_phase.test_result to running|fixing"
    )
    _add_common(pt)
    pt.add_argument("--phase", type=int, required=True)
    pt.add_argument("--state", required=True, choices=["running", "fixing"])
    pt.add_argument("--details", default=None)
    pt.set_defaults(func=cmd_phase_test)

    # phase-end ------------------------------------------------------------
    pe = sub.add_parser("phase-end", help="Finalize a per_phase entry")
    _add_common(pe)
    pe.add_argument("--phase", type=int, required=True)
    pe.add_argument(
        "--test-result", required=True, choices=["passed", "failed", "skipped"]
    )
    pe.add_argument("--compilation-attempts", type=int, default=None)
    pe.add_argument("--debug-cycles", type=int, default=None)
    pe.add_argument("--test-details", default=None)
    pe.add_argument(
        "--compile-errors-json",
        default=None,
        help="JSON array of {attempt, error} objects",
    )
    pe.add_argument("--now", default=None)
    pe.set_defaults(func=cmd_phase_end)

    # failure --------------------------------------------------------------
    fl = sub.add_parser("failure", help="Append to failures[]")
    _add_common(fl)
    fl.add_argument("--step", required=True)
    fl.add_argument("--agent", required=True)
    fl.add_argument(
        "--type",
        required=True,
        choices=["compile_error", "test_failure", "agent_error", "infra_error"],
    )
    fl.add_argument("--message", required=True)
    fl.add_argument("--resolved", required=True, choices=["true", "false"])
    fl.set_defaults(func=cmd_failure)

    # metric ---------------------------------------------------------------
    mt = sub.add_parser("metric", help="Patch top-level scalar fields")
    _add_common(mt)
    mt.add_argument(
        "--patch-json", required=True, help="JSON object of key/value pairs to merge in"
    )
    mt.set_defaults(func=cmd_metric)

    # finalize -------------------------------------------------------------
    fz = sub.add_parser("finalize", help="Close out run.json at run end")
    _add_common(fz)
    fz.add_argument(
        "--status", required=True, choices=["success", "compiled", "failed", "skipped"]
    )
    fz.add_argument(
        "--end-time", default=None, help="ISO 8601 timestamp; defaults to now"
    )
    fz.add_argument(
        "--final-result",
        required=True,
        choices=["success", "compile_error", "test_failure"],
    )
    fz.add_argument("--final-message", default="")
    fz.add_argument(
        "--solver-state",
        default=None,
        choices=["not_working", "working", "draft_pr", "active_pr", "merged"],
        help="Optional issue-solver state for dashboard 5-state model",
    )
    fz.add_argument(
        "--patch-json",
        default=None,
        help="JSON object merged into the doc at finalize time",
    )
    fz.set_defaults(func=cmd_finalize)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"run_json_writer error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
