# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Atomic writer for the Activity-Monitor-aware run.json.

Implements the live-update contract defined by:
  - /proj_sw/user_dev/${USER}/llk_code_gen/dashboard/GEN_MONITOR_FIELDS.md
  - /proj_sw/user_dev/${USER}/llk_code_gen/dashboard/RUN_JSON_SPEC.md

Every subcommand updates <LOG_DIR>/run.json by writing a temp file in the
same directory and atomically renaming it into place so the dashboard never
reads a half-written file.

Subcommands:
    init            Write the initial run.json at run start (status=running,
                    first step_history entry in_progress).
    advance         Transition from the current step to a new step. Closes out
                    the in-flight step_history entry and appends a new one.
    message         Mid-step update of current_step_message (and optionally the
                    current in-flight step_history entry's message).
    phase-start     Mark a per_phase[] entry as started (start_time, name).
    phase-test      Set per_phase[].test_result to "running" or "fixing" while
                    the simulator / debugger is live.
    phase-end       Finalize a per_phase[] entry (end_time, duration, test
                    result, compile_errors, test_details).
    failure         Append an entry to the top-level failures[] array.
    metric          Patch arbitrary fields (compilation_attempts, debug_cycles,
                    tests_total, tests_passed, arch_results, etc.).
    link-siblings   Patch issue_run_id and sibling_runs on an existing run.json.
                    Kept for historical multi-arch issue-solver runs that used
                    one per-arch run.json under a shared issue_run_id.
    artifact-manifest
                    Seal a local attempt-owned LLK artifact set before execution.
    required-verification
                    Normalize and seal the run's immutable verification contract.
    candidate-patch-digest
                    Hash the complete base-to-worktree candidate without changing
                    the worktree's real Git index.
    verification-result
                    Write a strict v2 result from collection/JUnit/artifact evidence.
    reduce-verification
                    Reduce sealed v2 results into deterministic run evidence.
    finalize        Close out the last step_history entry, set end_time, flip
                    status to a terminal value, merge in any remaining summary
                    fields passed via --patch-json.

Idempotency:
    State-replacing subcommands are idempotent — running one twice with the
    same arguments yields the same final document (modulo timestamps the caller
    passes explicitly): init, message, phase-start, phase-test, metric,
    link-siblings, finalize.

    Append/counter subcommands are intentionally NOT idempotent, per
    RUN_JSON_SPEC.md: advance and finalize's closeout append one step_history
    entry per invocation (the timeline records retries), failure appends to the
    append-only failures[] log, and phase-end increments phases_completed.
    Callers must emit these exactly once per real event.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
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


@contextmanager
def _run_json_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".run.json.lock"
    with lock_path.open("a+") as lock:
        try:
            os.chmod(lock_path, 0o664)
        except OSError:
            pass
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _atomic_write(
    log_dir: Path,
    doc: dict[str, Any],
    *,
    destination: Path | None = None,
    lock_held: bool = False,
) -> None:
    path = destination or _run_json_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "run.json" and not lock_held:
        with _run_json_lock(path):
            _atomic_write(log_dir, doc, destination=destination, lock_held=True)
        return
    try:
        if path.name == "run.json":
            current_sequence = 0
            try:
                current = json.loads(path.read_text()) if path.exists() else {}
                current_sequence = current.get("progress_sequence", 0)
            except (OSError, ValueError, TypeError):
                current_sequence = 0
            sequence = doc.get("progress_sequence", 0)
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                sequence = 0
            if (
                isinstance(current_sequence, bool)
                or not isinstance(current_sequence, int)
                or current_sequence < 0
            ):
                current_sequence = 0
            doc["progress_sequence"] = max(sequence, current_sequence) + 1
            doc["last_heartbeat"] = _utcnow()
            doc.setdefault("supervisor_phase", "active_compute")
        fd, tmp = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        # mkstemp creates files with 0o600, which locks the dashboard (running as
        # a different user) out of reading run.json. Relax to 0o664 so the shared
        # group — and anything else — can read the live status.
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except Exception:
        if "tmp" in locals() and os.path.exists(tmp):
            os.unlink(tmp)
        raise


@contextmanager
def _run_json_transaction(log_dir: Path):
    """Hold the run lock across the complete read/modify/write transaction."""
    path = _run_json_path(log_dir)
    with _run_json_lock(path):
        doc = _load(log_dir)
        yield doc
        _atomic_write(log_dir, doc, lock_held=True)


def _json_arg(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Merge nested dictionaries without replacing sibling keys."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value


def _set_dotted(dst: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c`` as nested dictionaries.

    This is mostly a compatibility affordance for agent-written metric patches.
    Prefer passing a nested JSON object in new prompts.
    """
    parts = [p for p in dotted_key.split(".") if p]
    if len(parts) < 2:
        dst[dotted_key] = value
        return
    cur = dst
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


def _merge_patch(doc: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if "." in key:
            _set_dotted(doc, key, value)
        elif isinstance(value, dict) and isinstance(doc.get(key), dict):
            _deep_merge(doc[key], value)
        else:
            doc[key] = value


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
        # Semantic version of the pipeline that produced this run. Optional and
        # additive: callers that do not pass --version (e.g. the Quasar codegen
        # orchestrator) get null, exactly as before. The issue solver passes its
        # own issue-solver-local version. Rendered as a colored pill by the
        # dashboard's renderVersionTag(); null shows the grey "unversioned" badge.
        "version": args.version,
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
        # Audit-lane provenance is inherited from the dashboard runner. Reading
        # four environment variables during init adds no work to agent steps.
        "runner_pool": os.environ.get("CODEGEN_RUNNER_POOL") or "prod",
        "base_commit": os.environ.get("CODEGEN_BASE_COMMIT") or None,
        "campaign_id": os.environ.get("CODEGEN_CAMPAIGN_ID") or None,
        "attempt_id": os.environ.get("CODEGEN_ATTEMPT_ID") or None,
        "resumed_from_run_id": os.environ.get("CODEGEN_RESUME_RUN_ID") or None,
        "resumed_from_attempt_id": (
            os.environ.get("CODEGEN_RESUME_ATTEMPT_ID") or None
        ),
        "resume_checkpoint_digest": (
            os.environ.get("CODEGEN_RESUME_CHECKPOINT_DIGEST") or None
        ),
        "resume_patch_sha256": (os.environ.get("CODEGEN_RESUME_PATCH_SHA256") or None),
        "resume_verification": (
            {
                "outcome": os.environ.get("CODEGEN_RESUME_VERIFICATION_REUSE"),
                "reason_code": os.environ.get("CODEGEN_RESUME_INVALIDATION_REASON"),
            }
            if os.environ.get("CODEGEN_RESUME_RUN_ID")
            else None
        ),
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "description": args.description or None,
        "num_turns": 0,
        "solver_state": None,
        # cost_usd stays None until either (a) a batch runner drops a
        # cli_output.json into LOG_DIR and the dashboard backfills, or
        # (b) finalize sets it via --patch-json. Zero would render as
        # "$0.00" in the dashboard; None renders as "—" (not captured).
        "cost_usd": None,
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
        # Legacy multi-arch grouping (optional).
        #   issue_run_id    — shared ID across N per-arch runs for one issue.
        #                     None for single-arch runs (today's default).
        #   sibling_runs    — list of {arch, run_id} pointers to the other
        #                     per-arch runs in the same issue. Empty for
        #                     single-arch runs. New single-run multi-arch flows
        #                     use arch="multi", target_arches, and arch_results
        #                     via --patch-json instead.
        "issue_run_id": args.issue_run_id,
        "sibling_runs": _json_arg(args.sibling_runs, []),
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

    patch = _json_arg(args.patch_json, {})
    _merge_patch(doc, patch)

    _atomic_write(log_dir, doc)
    print(f"init: wrote {_run_json_path(log_dir)}")


# --------------------------------------------------------------------------
# Subcommand: advance
# --------------------------------------------------------------------------


def cmd_advance(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    now = args.now or _utcnow()
    with _run_json_transaction(log_dir) as doc:
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
    print(f"advance: {args.new_step} ({args.prev_result} closed prior)")


# --------------------------------------------------------------------------
# Subcommand: message (mid-step progress)
# --------------------------------------------------------------------------


def cmd_message(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    with _run_json_transaction(log_dir) as doc:
        doc["current_step_message"] = args.message

        history = doc.get("step_history") or []
        if history and history[-1].get("result") == "in_progress":
            history[-1]["message"] = args.message
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
    now = args.now or _utcnow()
    with _run_json_transaction(log_dir) as doc:
        entry = _phase_entry(doc, args.phase)
        if args.name:
            entry["name"] = args.name
        entry["start_time"] = now
        entry["test_result"] = "pending"
        entry["end_time"] = None
        entry["duration_seconds"] = None
    print(f"phase-start: phase {args.phase} ({args.name or entry.get('name')})")


def cmd_phase_test(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    with _run_json_transaction(log_dir) as doc:
        entry = _phase_entry(doc, args.phase)
        entry["test_result"] = args.state  # "running" | "fixing"
        if args.details is not None:
            entry["test_details"] = args.details
    print(f"phase-test: phase {args.phase} -> {args.state}")


def cmd_phase_end(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    now = args.now or _utcnow()
    with _run_json_transaction(log_dir) as doc:
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
    print(f"phase-end: phase {args.phase} -> {args.test_result}")


# --------------------------------------------------------------------------
# Failures / metrics patching
# --------------------------------------------------------------------------


def cmd_failure(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    with _run_json_transaction(log_dir) as doc:
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
    print(f"failure: {args.type} @ {args.step}")


def cmd_metric(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    patch = _json_arg(args.patch_json, {})
    with _run_json_transaction(log_dir) as doc:
        _merge_patch(doc, patch)
    print(f"metric: patched {sorted(patch)}")


# --------------------------------------------------------------------------
# Subcommand: finalize
# --------------------------------------------------------------------------


def _validate_audit_success(
    log_dir: Path, doc: dict[str, Any], args: argparse.Namespace
) -> None:
    # Audit runs are the initial fail-closed lane. A caller cannot promote an
    # audit attempt merely by patching run.json: success must be authorized by
    # the current sealed manifest, reduction, and exact packaged candidate.
    if args.status != "success" or doc.get("runner_pool") != "audit":
        return
    manifest_path = log_dir / "required_verification_manifest.json"
    reduction_path = log_dir / "verification_reduction.json"
    if not manifest_path.is_file():
        raise ValueError("audit success requires a required-verification manifest")
    if not reduction_path.is_file():
        raise ValueError("audit success requires a verification reduction")
    manifest = _load_required_manifest(manifest_path)
    reduction = _load_verification_reduction(reduction_path)
    if doc.get("run_id") != manifest["run_id"]:
        raise ValueError("audit success manifest does not belong to this run")
    for field in ("run_id", "attempt_id", "manifest_id", "expected_base_sha"):
        if reduction[field] != manifest[field]:
            raise ValueError(f"audit success reduction {field} mismatch")
    if (
        reduction["scope"] != "all"
        or reduction["classification"] != "success"
        or reduction["reason_codes"]
        or len(reduction["leaves"]) != len(manifest["requirements"])
        or any(
            leaf["classification"] != "success" or leaf["result_id"] is None
            for leaf in reduction["leaves"]
        )
        or reduction["patch_sha256"] is None
        or reduction["success_token"] is None
    ):
        raise ValueError("audit success reduction is not complete and successful")
    expected_token = _canonical_digest(
        {
            "schema": "tt.issue-solver.verification-success-token",
            "version": 1,
            "reduction_id": reduction["reduction_id"],
            "manifest_id": manifest["manifest_id"],
            "run_id": manifest["run_id"],
            "attempt_id": manifest["attempt_id"],
            "patch_sha256": reduction["patch_sha256"],
        }
    )
    if reduction["success_token"] != expected_token:
        raise ValueError("audit success token is invalid")
    if not args.worktree:
        raise ValueError("audit success requires the packaged worktree")
    worktree = Path(args.worktree).resolve()
    if (
        _candidate_patch_digest(worktree, manifest["expected_base_sha"])
        != reduction["patch_sha256"]
    ):
        raise ValueError("audit success candidate patch differs from verified patch")


def cmd_finalize(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    now = args.end_time or _utcnow()
    patch = _json_arg(args.patch_json, {})

    with _run_json_transaction(log_dir) as doc:
        _validate_audit_success(log_dir, doc, args)

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
        doc["final_result"] = args.final_result
        doc["final_message"] = args.final_message or None
        doc["current_step_message"] = (
            args.final_message or doc.get("current_step_message") or ""
        )

        _merge_patch(doc, patch)

        # Apply typed --solver-state last so it cannot be silently overridden by
        # --patch-json (argparse choices otherwise bypass it via that escape hatch).
        if args.solver_state is not None:
            doc["solver_state"] = args.solver_state

    print(f"finalize: status={args.status}")


# --------------------------------------------------------------------------
# Subcommand: link-siblings (multi-arch grouping, optional)
# --------------------------------------------------------------------------


def cmd_link_siblings(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    siblings = _json_arg(args.siblings, [])
    if not isinstance(siblings, list):
        raise SystemExit("--siblings must be a JSON array")
    with _run_json_transaction(log_dir) as doc:
        doc["sibling_runs"] = siblings
        if args.issue_run_id is not None:
            doc["issue_run_id"] = args.issue_run_id
    print(f"link-siblings: {len(siblings)} sibling(s) linked")


# --------------------------------------------------------------------------
# Immutable local artifacts and strict verification results
# --------------------------------------------------------------------------


_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INFRASTRUCTURE_MARKERS_V1 = (
    ("tt_fatal", "TT_FATAL"),
    ("gtest_suite_setup_failed", "SetUpTestSuite or TearDownTestSuite"),
    ("device_count_failed", "GetNumAvailableDevices"),
    ("runtime_root_unset", "Root Directory is not set"),
    ("firmware_initialization_failed", "failed to initialize FW"),
    ("device_initialization_failed", "failed to initialize device"),
    ("device_initialization_failed", "device initialization failed"),
    ("umd_initialization_failed", "UMD initialization failed"),
    ("no_device_available", "No devices available"),
    ("no_device_available", "No Tenstorrent device"),
    ("tensix_timed_out", "TENSIX TIMED OUT"),
    ("polling_timeout", "Polling brisc command timed out"),
)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_patch_digest(worktree: Path, base: str) -> str:
    """Hash all candidate content while preserving setup-owned index exclusions."""
    if not worktree.is_dir() or not _SHA40_RE.fullmatch(base):
        raise ValueError("candidate patch requires a worktree and exact base SHA")
    worktree = Path(
        subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    ).resolve()
    resolved = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "--verify", f"{base}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if resolved != base:
        raise ValueError("candidate patch base did not resolve to itself")
    index = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "index",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if not Path(index).is_file():
        raise ValueError("candidate worktree index is unavailable")

    temporary_index = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=Path(index).parent,
            prefix=".candidate-index-",
            delete=False,
        ) as temporary:
            temporary_index = temporary.name
            temporary.write(Path(index).read_bytes())
        env = {**os.environ, "GIT_INDEX_FILE": temporary_index}
        subprocess.run(
            ["git", "-C", str(worktree), "add", "-A", "--", "."],
            check=True,
            capture_output=True,
            env=env,
            timeout=120,
        )
        patch = subprocess.run(
            ["git", "-C", str(worktree), "diff", "--cached", "--binary", base, "--"],
            check=True,
            capture_output=True,
            env=env,
            timeout=120,
        ).stdout
    finally:
        if temporary_index:
            Path(temporary_index).unlink(missing_ok=True)
    return hashlib.sha256(patch).hexdigest()


def cmd_candidate_patch_digest(args: argparse.Namespace) -> None:
    print(_candidate_patch_digest(Path(args.worktree), args.expected_base_sha))


_VERIFICATION_ARCHES = {"blackhole", "wormhole", "quasar"}
_COVERED_STATES = {"existing", "added"}


def _markdown_section(text: str, heading: str) -> str:
    """Return one exact level-two Markdown section without parsing prose."""
    match = re.search(
        rf"(?ms)^##[ \t]+{re.escape(heading)}[ \t]*\n(.*?)(?=^##[ \t]+|\Z)",
        text,
    )
    return match.group(1) if match else ""


def _markdown_value(value: str) -> str:
    """Remove Markdown quoting and an unquoted inline comment from a scalar."""
    quote = None
    end = len(value)
    for index, char in enumerate(value):
        if char in "'\"`":
            quote = None if quote == char else (char if quote is None else quote)
        elif (
            char == "#" and quote is None and (index == 0 or value[index - 1].isspace())
        ):
            end = index
            break
    cleaned = value[:end].strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "'\"`":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _markdown_scalar(section: str, key: str) -> str:
    match = re.search(rf"(?mi)^[ \t]*{re.escape(key)}[ \t]*:[ \t]*([^\n]*)$", section)
    return _markdown_value(match.group(1)) if match else ""


def _markdown_items(section: str, group_names: set[str] | None = None) -> list[dict]:
    """Parse the small key/value list shape used by analysis and fix plans."""
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    group = ""
    for line in section.splitlines():
        group_match = re.match(r"^([A-Za-z_]+)[ \t]*:[ \t]*$", line)
        if group_match and group_names and group_match.group(1) in group_names:
            group = group_match.group(1)
            current = None
            continue
        item_match = re.match(r"^[ \t]*-[ \t]+([A-Za-z_]+)[ \t]*:[ \t]*(.*)$", line)
        if item_match:
            current = {"_group": group, "_raw": line}
            current[item_match.group(1)] = _markdown_value(item_match.group(2))
            items.append(current)
            continue
        field_match = re.match(r"^[ \t]+([A-Za-z_]+)[ \t]*:[ \t]*(.*)$", line)
        if current is not None and field_match:
            current[field_match.group(1)] = _markdown_value(field_match.group(2))
        if current is not None:
            current["_raw"] += "\n" + line
    return items


def _normalize_arch_scope(scope: str, requested: list[str]) -> list[str]:
    states: dict[str, str] = {}
    in_arch_scope = False
    for line in scope.splitlines():
        if re.match(r"^arch_scope[ \t]*:[ \t]*$", line):
            in_arch_scope = True
            continue
        if in_arch_scope and line and not line[0].isspace():
            break
        if in_arch_scope:
            match = re.match(
                r"^[ \t]+(blackhole|wormhole|quasar)[ \t]*:[ \t]*(in_scope|out_of_scope)[ \t]*$",
                line,
                re.IGNORECASE,
            )
            if match:
                states[match.group(1).lower()] = match.group(2).lower()
    if states:
        missing = set(requested) - states.keys()
        if missing:
            raise ValueError(
                "arch_scope omits requested architecture(s): "
                + ", ".join(sorted(missing))
            )
    return [arch for arch in requested if states.get(arch, "in_scope") == "in_scope"]


def _normalize_pytest_selector(
    raw: str, arch: str, worktree: Path
) -> dict[str, str | None]:
    """Normalize one exact run_test.sh selector and verify its test path exists."""
    value = _markdown_value(raw)
    try:
        tokens = shlex.split(value)
    except ValueError as exc:
        raise ValueError(f"invalid pytest selector {raw!r}: {exc}") from exc
    if len(tokens) == 3 and tokens[1] == "-k" and tokens[2]:
        target, k_filter = tokens[0], tokens[2]
    elif len(tokens) == 1:
        target, k_filter = tokens[0], None
    else:
        raise ValueError(
            f"pytest selector must be an exact path/id or 'path -k expression': {raw!r}"
        )
    if "::" in target:
        test_path, node = target.split("::", 1)
        if not node:
            raise ValueError(f"pytest selector has an empty node id: {raw!r}")
    else:
        test_path, node = target, None
    test_path = test_path.removeprefix("./").removeprefix("tt_metal/tt-llk/")
    tests_root = Path("tests/python_tests")
    arch_root = tests_root / "quasar" if arch == "quasar" else tests_root
    if test_path.startswith("tests/python_tests/"):
        relative = Path(test_path).relative_to(tests_root)
        if arch == "quasar":
            try:
                relative = relative.relative_to("quasar")
            except ValueError as exc:
                raise ValueError(
                    f"Quasar selector must be under tests/python_tests/quasar: {raw!r}"
                ) from exc
    elif "/" not in test_path or test_path.startswith("ai_gen/"):
        relative = Path(test_path)
    else:
        raise ValueError(f"pytest selector is outside the LLK suite: {raw!r}")
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
        raise ValueError(f"pytest selector has an invalid test path: {raw!r}")
    actual = worktree / "tt_metal" / "tt-llk" / arch_root / relative
    if not actual.is_file():
        raise ValueError(f"pytest selector names a missing file: {actual}")
    test = relative.as_posix()
    return {
        "test": test,
        "test_id": f"{test}::{node}" if node else None,
        "k": k_filter,
    }


def _is_performance_selector(raw: str) -> bool:
    value = _markdown_value(raw).split("::", 1)[0]
    return Path(value.split()[0]).name.startswith("perf_")


def _is_llk_pytest_selector(raw: str) -> bool:
    try:
        target = shlex.split(_markdown_value(raw))[0].split("::", 1)[0]
    except (IndexError, ValueError):
        return False
    target = target.removeprefix("./").removeprefix("tt_metal/tt-llk/")
    return target.endswith(".py") and (
        "/" not in target
        or target.startswith("tests/python_tests/")
        or target.startswith("ai_gen/")
    )


def _verification_backend(backend: str, arch: str) -> str:
    if backend == "ttsim":
        return "ttsim"
    if backend != "local":
        raise ValueError(f"unsupported verification backend: {backend!r}")
    return "quasar" if arch == "quasar" else "silicon"


def _requirement_count(item: dict, field: str, default: int = 1) -> int:
    raw = item.get(field, "")
    if not raw:
        return default
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{field} must be a positive integer (got {raw!r})")
    return int(raw)


def _required_measurements(item: dict) -> list[str]:
    raw = item.get("required_measurements", "")
    if not raw or raw == "[]":
        return []
    try:
        values = json.loads(raw)
    except ValueError as exc:
        raise ValueError("required_measurements must be a JSON string array") from exc
    allowed = {"cycle_comparison", "repeatability"}
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or value not in allowed for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            f"required_measurements must be a unique subset of {sorted(allowed)}"
        )
    return values


def _load_predeclared_waiver_policy(
    path: str, worktree: Path, expected_base_sha: str
) -> tuple[list[dict[str, Any]], str, str]:
    """Load waiver authority only from the exact checked-out base revision."""
    candidate = Path(path)
    try:
        relative = (
            candidate.relative_to(worktree) if candidate.is_absolute() else candidate
        )
    except ValueError as exc:
        raise ValueError(
            "waiver policy must be a tracked file inside the worktree"
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError("waiver policy must be a tracked file inside the worktree")
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "show",
            f"{expected_base_sha}:{relative.as_posix()}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError("waiver policy is not present in expected_base_sha")
    try:
        document = json.loads(proc.stdout)
    except ValueError as exc:
        raise ValueError("waiver policy is not valid JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "version", "policies"}
        or document.get("schema") != "tt.issue-solver.verification-waiver-policy"
        or document.get("version") != 1
        or not isinstance(document.get("policies"), list)
        or not document["policies"]
    ):
        raise ValueError("waiver policy schema is invalid")
    policy_fields = {
        "policy_id",
        "approver",
        "reason",
        "scope",
        "replacement",
        "allowed_outcomes",
    }
    scope_fields = {"architecture", "suite", "backend", "selector"}
    replacement_fields = {
        *scope_fields,
        "minimum_selected",
        "minimum_executed",
        "required_measurements",
    }
    identities = set()
    for item in document["policies"]:
        if not isinstance(item, dict) or set(item) != policy_fields:
            raise ValueError("waiver policy entry schema is invalid")
        if (
            not isinstance(item["policy_id"], str)
            or not item["policy_id"]
            or item["policy_id"] in identities
            or not isinstance(item["approver"], str)
            or not item["approver"].strip()
            or not isinstance(item["reason"], str)
            or not item["reason"].strip()
        ):
            raise ValueError("waiver policy identity or approval is invalid")
        identities.add(item["policy_id"])
        if (
            not isinstance(item["scope"], dict)
            or set(item["scope"]) != scope_fields
            or not isinstance(item["replacement"], dict)
            or set(item["replacement"]) != replacement_fields
        ):
            raise ValueError("waiver policy scope or replacement is invalid")
        for requirement in (item["scope"], item["replacement"]):
            selector = requirement["selector"]
            if (
                requirement["architecture"] not in _VERIFICATION_ARCHES
                or requirement["suite"] != "llk"
                or requirement["backend"] not in {"silicon", "ttsim", "quasar", "local"}
                or not isinstance(selector, dict)
                or set(selector) != {"test", "test_id", "k"}
                or not isinstance(selector["test"], str)
                or not selector["test"]
                or any(
                    selector[field] is not None and not isinstance(selector[field], str)
                    for field in ("test_id", "k")
                )
                or (selector["test_id"] is not None and selector["k"] is not None)
            ):
                raise ValueError("waiver policy requirement identity is invalid")
        replacement = item["replacement"]
        for field in ("minimum_selected", "minimum_executed"):
            value = replacement[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("waiver replacement counts must be positive")
        measurements = replacement["required_measurements"]
        if (
            not isinstance(measurements, list)
            or any(
                value not in {"cycle_comparison", "repeatability"}
                for value in measurements
            )
            or len(set(measurements)) != len(measurements)
        ):
            raise ValueError("waiver replacement measurements are invalid")
        outcomes = item["allowed_outcomes"]
        if (
            not isinstance(outcomes, list)
            or not outcomes
            or any(value not in {"skipped", "xfailed"} for value in outcomes)
            or len(set(outcomes)) != len(outcomes)
        ):
            raise ValueError("waiver allowed_outcomes are invalid")
    return document["policies"], _canonical_digest(document), relative.as_posix()


def _verify_manifest_waiver_policy(manifest: dict[str, Any], worktree: Path) -> None:
    """Reopen base-tracked policy and prove every sealed waiver is authorized."""
    requirements = {item["requirement_id"]: item for item in manifest["requirements"]}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for waiver in manifest["waivers"]:
        grouped.setdefault((waiver["policy_path"], waiver["policy_sha256"]), []).append(
            waiver
        )
    for (policy_path, recorded_digest), waivers in grouped.items():
        policies, actual_digest, actual_path = _load_predeclared_waiver_policy(
            policy_path, worktree, manifest["expected_base_sha"]
        )
        if actual_digest != recorded_digest or actual_path != policy_path:
            raise ValueError("required-verification waiver policy identity mismatch")
        policies_by_id = {item["policy_id"]: item for item in policies}
        for waiver in waivers:
            policy = policies_by_id.get(waiver["policy_id"])
            scope = requirements[waiver["scope_requirement_id"]]
            replacement = requirements[waiver["replacement_requirement_id"]]
            if (
                policy is None
                or waiver["approver"] != policy["approver"]
                or waiver["reason"] != policy["reason"]
                or waiver["allowed_outcomes"] != policy["allowed_outcomes"]
                or any(
                    scope[field] != policy["scope"][field]
                    for field in ("architecture", "suite", "backend", "selector")
                )
                or any(
                    replacement[field] != policy["replacement"][field]
                    for field in (
                        "architecture",
                        "suite",
                        "backend",
                        "selector",
                        "minimum_selected",
                        "minimum_executed",
                        "required_measurements",
                    )
                )
            ):
                raise ValueError(
                    "required-verification waiver is not policy-authorized"
                )


def _load_required_manifest(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema",
        "version",
        "manifest_id",
        "run_id",
        "attempt_id",
        "expected_base_sha",
        "revision",
        "parent_manifest_id",
        "supersedes_reason",
        "requirements",
        "waivers",
    }
    if (
        not isinstance(doc, dict)
        or set(doc) != fields
        or doc.get("schema") != "tt.issue-solver.required-verification"
        or doc.get("version") != 1
        or doc.get("manifest_id")
        != _canonical_digest(
            {key: value for key, value in doc.items() if key != "manifest_id"}
        )
    ):
        raise ValueError(f"invalid required-verification manifest: {path}")
    if not isinstance(doc["run_id"], str) or not doc["run_id"]:
        raise ValueError("required-verification run_id must be nonempty")
    if not re.fullmatch(r"attempt-\d{3,}", doc["attempt_id"] or ""):
        raise ValueError("required-verification attempt_id is invalid")
    if not _SHA40_RE.fullmatch(doc["expected_base_sha"] or ""):
        raise ValueError("required-verification expected_base_sha is invalid")
    if (
        isinstance(doc["revision"], bool)
        or not isinstance(doc["revision"], int)
        or doc["revision"] < 1
    ):
        raise ValueError("required-verification revision must be positive")
    if doc["revision"] == 1:
        if (
            doc["parent_manifest_id"] is not None
            or doc["supersedes_reason"] is not None
        ):
            raise ValueError("first required-verification revision cannot supersede")
    elif (
        not isinstance(doc["parent_manifest_id"], str)
        or not _SHA256_RE.fullmatch(doc["parent_manifest_id"])
        or not isinstance(doc["supersedes_reason"], str)
        or not doc["supersedes_reason"].strip()
    ):
        raise ValueError("superseding required-verification revision is unlinked")
    if not isinstance(doc["requirements"], list):
        raise ValueError("required-verification requirements must be an array")
    requirement_fields = {
        "requirement_id",
        "architecture",
        "suite",
        "backend",
        "selector",
        "minimum_selected",
        "minimum_executed",
        "required_measurements",
    }
    identities = set()
    for requirement in doc["requirements"]:
        if not isinstance(requirement, dict) or set(requirement) != requirement_fields:
            raise ValueError("required-verification requirement schema is invalid")
        identity = requirement["requirement_id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("required-verification requirement_id is invalid")
        identities.add(identity)
        if requirement["architecture"] not in _VERIFICATION_ARCHES:
            raise ValueError("required-verification architecture is invalid")
        if requirement["suite"] not in {"llk", "metal", "perf"}:
            raise ValueError("required-verification suite is invalid")
        if requirement["backend"] not in {"silicon", "ttsim", "quasar", "local"}:
            raise ValueError("required-verification backend is invalid")
        selector = requirement["selector"]
        if (
            not isinstance(selector, dict)
            or set(selector) != {"test", "test_id", "k"}
            or not isinstance(selector["test"], str)
            or not selector["test"]
            or any(
                selector[field] is not None and not isinstance(selector[field], str)
                for field in ("test_id", "k")
            )
            or (selector["test_id"] is not None and selector["k"] is not None)
        ):
            raise ValueError("required-verification selector is invalid")
        for field in ("minimum_selected", "minimum_executed"):
            value = requirement[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"required-verification {field} must be positive")
        measurements = requirement["required_measurements"]
        if (
            not isinstance(measurements, list)
            or any(
                value not in {"cycle_comparison", "repeatability"}
                for value in measurements
            )
            or len(set(measurements)) != len(measurements)
        ):
            raise ValueError("required-verification measurements are invalid")
    if not isinstance(doc["waivers"], list):
        raise ValueError("required-verification waivers must be an array")
    waiver_fields = {
        "waiver_id",
        "policy_id",
        "policy_sha256",
        "policy_path",
        "approver",
        "reason",
        "scope_requirement_id",
        "replacement_requirement_id",
        "allowed_outcomes",
    }
    waiver_ids = set()
    scope_ids = set()
    replacement_ids = set()
    for waiver in doc["waivers"]:
        if not isinstance(waiver, dict) or set(waiver) != waiver_fields:
            raise ValueError("required-verification waiver schema is invalid")
        if (
            not isinstance(waiver["waiver_id"], str)
            or not _SHA256_RE.fullmatch(waiver["waiver_id"])
            or waiver["waiver_id"] in waiver_ids
            or waiver["waiver_id"]
            != _canonical_digest(
                {key: value for key, value in waiver.items() if key != "waiver_id"}
            )
        ):
            raise ValueError("required-verification waiver_id is invalid")
        waiver_ids.add(waiver["waiver_id"])
        for field in ("policy_id", "approver", "reason"):
            if not isinstance(waiver[field], str) or not waiver[field].strip():
                raise ValueError(f"required-verification waiver {field} is invalid")
        if not _SHA256_RE.fullmatch(waiver["policy_sha256"] or ""):
            raise ValueError("required-verification waiver policy digest is invalid")
        if not isinstance(waiver["policy_path"], str) or not waiver["policy_path"]:
            raise ValueError("required-verification waiver policy path is invalid")
        policy_path = Path(waiver["policy_path"])
        if policy_path.is_absolute() or ".." in policy_path.parts:
            raise ValueError("required-verification waiver policy path is invalid")
        scope_id = waiver["scope_requirement_id"]
        replacement_id = waiver["replacement_requirement_id"]
        if (
            scope_id not in identities
            or replacement_id not in identities
            or scope_id == replacement_id
            or scope_id in scope_ids
        ):
            raise ValueError("required-verification waiver requirement link is invalid")
        scope_ids.add(scope_id)
        replacement_ids.add(replacement_id)
        outcomes = waiver["allowed_outcomes"]
        if (
            not isinstance(outcomes, list)
            or not outcomes
            or any(value not in {"skipped", "xfailed"} for value in outcomes)
            or len(set(outcomes)) != len(outcomes)
        ):
            raise ValueError("required-verification waiver outcomes are invalid")
    if scope_ids & replacement_ids:
        raise ValueError("required-verification chained waivers are unsupported")
    return doc


def cmd_required_verification(args: argparse.Namespace) -> None:
    """Normalize the current analysis/plan into one immutable manifest revision."""
    worktree = Path(args.worktree).resolve(strict=True)
    analysis = Path(args.analysis).read_text(encoding="utf-8")
    plan = Path(args.plan).read_text(encoding="utf-8")
    if not _SHA40_RE.fullmatch(args.expected_base_sha):
        raise ValueError("expected_base_sha must be 40 lowercase hex")
    requested = json.loads(args.architectures_json)
    if (
        not isinstance(requested, list)
        or not requested
        or any(arch not in _VERIFICATION_ARCHES for arch in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("architectures_json must be a unique nonempty supported list")

    scope = _markdown_section(analysis, "Scope")
    arches = _normalize_arch_scope(scope, requested)
    if not arches:
        raise ValueError("required-verification manifest has no in-scope architecture")
    verification = _markdown_section(analysis, "Verification")
    candidates = _markdown_items(_markdown_section(analysis, "Test Candidates"))
    strategy = _markdown_section(plan, "Test Strategy")
    plan_tests = _markdown_items(strategy, {"reproduction_tests", "regression_tests"})

    verify_required = _markdown_scalar(verification, "verification_required")
    verifiable = _markdown_scalar(verification, "verifiable_in_llk_suite")
    llk_coverage = _markdown_scalar(verification, "llk_coverage")
    metal_match = re.search(
        r"(?ms)^metal_verification[ \t]*:[^\n]*\n(.*?)(?=^[^ \t\n]|\Z)",
        verification,
    )
    metal = metal_match.group(1) if metal_match else ""
    metal_target = _markdown_scalar(metal, "target")
    metal_coverage = _markdown_scalar(metal, "coverage")
    metal_test_file = _markdown_scalar(metal, "test_file")
    metal_filter = _markdown_scalar(metal, "gtest_filter")
    metal_dispatch = _markdown_scalar(metal, "dispatch")

    applicable_candidates = [
        item
        for item in candidates
        if item.get("arch", "all") in {*arches, "all"}
        and item.get("test")
        and _is_llk_pytest_selector(item["test"])
        and not _is_performance_selector(item["test"])
    ]
    # Old analysis revisions had only verifiable_in_llk_suite plus Test Candidates.
    # Preserve their applicable LLK route instead of treating the absent Metal block
    # or later-added coverage keys as permission to skip verification.
    llk_applicable = verify_required != "no" and verifiable in {"yes", "partial"}
    if not verifiable and verify_required != "no" and applicable_candidates:
        llk_applicable = True
    if llk_applicable and not llk_coverage:
        candidate_states = {item.get("coverage", "") for item in applicable_candidates}
        if "add_required" in candidate_states:
            llk_coverage = "add_required"
        elif applicable_candidates and candidate_states <= {"", *_COVERED_STATES}:
            llk_coverage = "existing"

    functional_plan = [
        item
        for item in plan_tests
        if item.get("test") and not _is_performance_selector(item["test"])
    ]
    performance_plan = [
        item
        for item in plan_tests
        if item.get("test") and _is_performance_selector(item["test"])
    ]
    if not llk_applicable and verify_required != "no" and functional_plan:
        llk_applicable = True
    if llk_applicable and not llk_coverage:
        plan_states = {item.get("coverage", "") for item in functional_plan}
        if "add_required" in plan_states:
            llk_coverage = "add_required"
        elif functional_plan and plan_states <= {"", *_COVERED_STATES}:
            llk_coverage = "existing"
    if args.performance_only:
        llk_applicable = False
    if llk_applicable and llk_coverage not in _COVERED_STATES:
        raise ValueError(
            f"LLK coverage must be existing|added before execution (got {llk_coverage or 'missing'})"
        )
    if not args.performance_only and any(
        item.get("coverage") == "add_required" for item in plan_tests
    ):
        raise ValueError("required test coverage remains add_required")

    requirements: list[dict[str, Any]] = []
    counters: dict[tuple[str, str], int] = {}

    def add_requirement(
        arch: str,
        suite: str,
        backend: str,
        selector: dict[str, str | None],
        minimum_selected: int = 1,
        minimum_executed: int = 1,
        measurements: list[str] | None = None,
    ) -> dict[str, Any]:
        key = (arch, suite)
        if any(
            item["architecture"] == arch
            and item["suite"] == suite
            and item["selector"] == selector
            for item in requirements
        ):
            raise ValueError(f"duplicate {suite} selector for {arch}: {selector!r}")
        counters[key] = counters.get(key, 0) + 1
        requirement = {
            "requirement_id": f"{arch}:{suite}:{counters[key]}",
            "architecture": arch,
            "suite": suite,
            "backend": backend,
            "selector": selector,
            "minimum_selected": minimum_selected,
            "minimum_executed": minimum_executed,
            "required_measurements": measurements or [],
        }
        requirements.append(requirement)
        return requirement

    if llk_applicable:
        source_items = functional_plan or applicable_candidates
        for arch in arches:
            selected = [
                item
                for item in source_items
                if item.get("arch", "all") in {arch, "all"}
            ]
            if not selected:
                raise ValueError(f"missing exact LLK selector for {arch}")
            for item in selected:
                add_requirement(
                    arch,
                    "llk",
                    _verification_backend(args.backend, arch),
                    _normalize_pytest_selector(item["test"], arch, worktree),
                    _requirement_count(item, "minimum_selected"),
                    _requirement_count(item, "minimum_executed"),
                    _required_measurements(item),
                )

    metal_applicable = not args.performance_only and metal_target not in {"", "none"}
    if metal_applicable:
        if metal_target != "unit_tests_llk" or metal_coverage not in _COVERED_STATES:
            raise ValueError("Metal verification target/coverage is not executable")
        if not metal_filter or metal_filter in {"*", "'*'", '"*"'}:
            raise ValueError("Metal verification requires a tight gtest_filter")
        if metal_dispatch not in {"slow", "fast"}:
            raise ValueError("Metal verification dispatch must be slow|fast")
        metal_path = metal_test_file.removeprefix("./")
        if not metal_path or metal_path == "none":
            raise ValueError("Metal verification requires a test_file")
        if not (worktree / metal_path).is_file():
            raise ValueError(
                f"Metal selector names a missing file: {worktree / metal_path}"
            )
        for arch in arches:
            add_requirement(
                arch,
                "metal",
                _verification_backend(args.backend, arch),
                {"test": metal_filter, "test_id": None, "k": None},
            )
    elif (
        not args.performance_only
        and verifiable in {"no", "partial"}
        and verify_required == "yes"
    ):
        if not llk_applicable:
            raise ValueError(
                "required Metal route has no executable verification section"
            )
        if verifiable == "partial":
            raise ValueError("partial verification route is missing its Metal section")

    for arch in arches:
        selected = [
            item
            for item in performance_plan
            if item.get("arch", "all") in {arch, "all"}
        ]
        for item in selected:
            backend = _verification_backend(args.backend, arch)
            if backend != "silicon":
                raise ValueError(
                    f"performance requirement for {arch} requires silicon, got {backend}"
                )
            repetitions = [
                int(value)
                for value in re.findall(r"(?i)\bN\s*(?:>=|≥)\s*(\d+)\b", item["_raw"])
            ]
            lowered = item["_raw"].casefold()
            deterministic = "determinism" in lowered or "deterministic" in lowered
            measurements = _required_measurements(item) or ["cycle_comparison"]
            if deterministic and "repeatability" not in measurements:
                measurements.append("repeatability")
            add_requirement(
                arch,
                "perf",
                backend,
                _normalize_pytest_selector(item["test"], arch, worktree),
                _requirement_count(item, "minimum_selected"),
                max([*repetitions, _requirement_count(item, "minimum_executed")]),
                measurements,
            )

    if verify_required == "yes" and not requirements:
        raise ValueError(
            "verification is required but no executable requirement exists"
        )
    if not requirements and verify_required != "no":
        raise ValueError("unsupported or missing verification route")

    output = Path(args.output)
    revisions = output.parent / "required_verification_manifests"
    prior_paths = list(revisions.glob("revision-*.json")) if revisions.exists() else []
    numbered = []
    for path in prior_paths:
        match = re.fullmatch(r"revision-(\d+)\.json", path.name)
        if not match:
            raise ValueError(f"invalid manifest revision filename: {path}")
        numbered.append((int(match.group(1)), path))
    numbered.sort()
    previous = None
    for expected_revision, (revision, path) in enumerate(numbered, 1):
        if revision != expected_revision:
            raise ValueError(
                "required-verification manifest revisions are not contiguous"
            )
        current = _load_required_manifest(path)
        if current.get("revision") != revision:
            raise ValueError(f"manifest revision number contradicts filename: {path}")
        expected_parent = previous["manifest_id"] if previous else None
        if current.get("parent_manifest_id") != expected_parent:
            raise ValueError(f"manifest revision chain is broken at {path}")
        previous = current
    if previous:
        if previous["run_id"] != args.run_id:
            raise ValueError("cannot supersede a manifest from another run")
        if previous["expected_base_sha"] != args.expected_base_sha:
            raise ValueError(
                "cannot change expected_base_sha across manifest revisions"
            )
        if not args.supersedes_reason:
            raise ValueError("a superseding manifest requires --supersedes-reason")

    def apply_waiver_policies(
        policies: list[dict[str, Any]], policy_sha256: str, policy_path: str
    ) -> list[dict[str, Any]]:
        waivers = []
        for policy in policies:
            matches = [
                requirement
                for requirement in requirements
                if all(
                    requirement[field] == policy["scope"][field]
                    for field in ("architecture", "suite", "backend", "selector")
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"waiver policy {policy['policy_id']} must match exactly one requirement"
                )
            scope = matches[0]
            replacement_spec = policy["replacement"]
            selector = replacement_spec["selector"]
            selector_text = selector["test_id"] or selector["test"]
            if selector["k"] is not None:
                selector_text = f"{selector['test']} -k {shlex.quote(selector['k'])}"
            normalized = _normalize_pytest_selector(
                selector_text, replacement_spec["architecture"], worktree
            )
            if normalized != selector:
                raise ValueError(
                    f"waiver policy {policy['policy_id']} replacement is not normalized"
                )
            replacement_matches = [
                requirement
                for requirement in requirements
                if all(
                    requirement[field] == replacement_spec[field]
                    for field in (
                        "architecture",
                        "suite",
                        "backend",
                        "selector",
                        "minimum_selected",
                        "minimum_executed",
                        "required_measurements",
                    )
                )
            ]
            if len(replacement_matches) > 1:
                raise ValueError("waiver replacement requirement is ambiguous")
            replacement = (
                replacement_matches[0]
                if replacement_matches
                else add_requirement(
                    replacement_spec["architecture"],
                    replacement_spec["suite"],
                    replacement_spec["backend"],
                    replacement_spec["selector"],
                    replacement_spec["minimum_selected"],
                    replacement_spec["minimum_executed"],
                    replacement_spec["required_measurements"],
                )
            )
            waiver = {
                "waiver_id": "0" * 64,
                "policy_id": policy["policy_id"],
                "policy_sha256": policy_sha256,
                "policy_path": policy_path,
                "approver": policy["approver"],
                "reason": policy["reason"],
                "scope_requirement_id": scope["requirement_id"],
                "replacement_requirement_id": replacement["requirement_id"],
                "allowed_outcomes": policy["allowed_outcomes"],
            }
            waiver["waiver_id"] = _canonical_digest(
                {key: value for key, value in waiver.items() if key != "waiver_id"}
            )
            waivers.append(waiver)
        return waivers

    waivers: list[dict[str, Any]] = []
    if args.waiver_policy:
        if previous and not previous["waivers"]:
            raise ValueError("cannot introduce a verification waiver after revision 1")
        policies, policy_sha256, policy_path = _load_predeclared_waiver_policy(
            args.waiver_policy, worktree, args.expected_base_sha
        )
        waivers = apply_waiver_policies(policies, policy_sha256, policy_path)
        if previous and {
            (item["policy_id"], item["policy_sha256"], item["policy_path"])
            for item in waivers
        } != {
            (item["policy_id"], item["policy_sha256"], item["policy_path"])
            for item in previous["waivers"]
        }:
            raise ValueError(
                "cannot change verification waiver policy across revisions"
            )
    elif previous and previous["waivers"]:
        previous_requirements = {
            item["requirement_id"]: item for item in previous["requirements"]
        }
        carried_policies = []
        for prior in previous["waivers"]:
            scope = previous_requirements[prior["scope_requirement_id"]]
            replacement = previous_requirements[prior["replacement_requirement_id"]]
            carried_policies.append(
                {
                    "policy_id": prior["policy_id"],
                    "approver": prior["approver"],
                    "reason": prior["reason"],
                    "scope": {
                        field: scope[field]
                        for field in ("architecture", "suite", "backend", "selector")
                    },
                    "replacement": {
                        field: replacement[field]
                        for field in (
                            "architecture",
                            "suite",
                            "backend",
                            "selector",
                            "minimum_selected",
                            "minimum_executed",
                            "required_measurements",
                        )
                    },
                    "allowed_outcomes": prior["allowed_outcomes"],
                }
            )
        waivers = apply_waiver_policies(
            carried_policies,
            previous["waivers"][0]["policy_sha256"],
            previous["waivers"][0]["policy_path"],
        )
    revision = int(previous["revision"]) + 1 if previous else 1
    doc = {
        "schema": "tt.issue-solver.required-verification",
        "version": 1,
        "manifest_id": "0" * 64,
        "run_id": args.run_id,
        "attempt_id": f"attempt-{revision:03d}",
        "expected_base_sha": args.expected_base_sha,
        "revision": revision,
        "parent_manifest_id": previous["manifest_id"] if previous else None,
        "supersedes_reason": args.supersedes_reason or None,
        "requirements": requirements,
        "waivers": waivers,
    }
    doc["manifest_id"] = _canonical_digest(
        {key: value for key, value in doc.items() if key != "manifest_id"}
    )
    revision_path = revisions / f"revision-{revision:03d}.json"
    if revision_path.exists():
        raise ValueError(f"manifest revision already exists: {revision_path}")
    _atomic_write(revisions, doc, destination=revision_path)
    _atomic_write(output.parent, doc, destination=output)
    suites = {item["suite"] for item in requirements}
    functional = suites & {"llk", "metal"}
    route = "both" if functional == {"llk", "metal"} else next(iter(functional), "none")
    print(
        f"required-verification: route={route} revision={revision} "
        f"attempt_id={doc['attempt_id']} manifest_id={doc['manifest_id']} "
        f"requirements={len(requirements)} output={output}"
    )


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink():
        raise ValueError(f"artifact root cannot be a symlink: {root}")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"artifact root is not a real directory: {root}")
    records = []
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise ValueError(
                    f"artifact root contains a symlink: {current_path / name}"
                )
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"artifact root contains a nonregular file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    if not records:
        raise ValueError("artifact root contains no files")
    return sorted(records, key=lambda record: record["path"])


def cmd_artifact_manifest(args: argparse.Namespace) -> None:
    records = _artifact_records(Path(args.artifact_root))
    manifest = {
        "schema": "tt.issue-solver.local-artifact-manifest",
        "version": 1,
        "manifest_id": "0" * 64,
        "owner_id": args.owner_id,
        "build_input_digest": args.build_input_digest,
        "source_tree_sha256": args.source_tree_sha256,
        "compiler_sha256": args.compiler_sha256,
        "artifact_set_sha256": _canonical_digest(records),
        "artifacts": records,
    }
    for field in (
        "build_input_digest",
        "source_tree_sha256",
        "compiler_sha256",
        "artifact_set_sha256",
    ):
        if not _SHA256_RE.fullmatch(manifest[field]):
            raise ValueError(
                f"local artifact manifest {field} must be 64 lowercase hex"
            )
    manifest["manifest_id"] = _canonical_digest(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    )
    output = Path(args.output)
    _atomic_write(output.parent, manifest, destination=output)
    print(f"artifact-manifest: wrote {output}")


def _parse_junit(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"invalid JUnit report: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if root.tag not in ("testsuite", "testsuites") or not suites:
        raise ValueError("JUnit report contains no direct test suites")
    totals = {field: 0 for field in ("tests", "failures", "errors", "skipped")}
    try:
        for suite in suites:
            for field in totals:
                value = int(suite.attrib.get(field, "0"))
                if value < 0:
                    raise ValueError
                totals[field] += value
    except (TypeError, ValueError) as exc:
        raise ValueError("JUnit report contains malformed counts") from exc

    xfailed = xpassed = xpassed_as_skip = 0
    for testcase in root.iter("testcase"):
        properties = {
            item.attrib.get("name"): item.attrib.get("value")
            for item in testcase.findall("./properties/property")
        }
        outcome = properties.get("codegen_outcome")
        skipped = testcase.find("skipped")
        if outcome == "xfailed" or (
            skipped is not None and skipped.attrib.get("type") == "pytest.xfail"
        ):
            xfailed += 1
        elif outcome == "xpassed" or (
            skipped is not None
            and skipped.attrib.get("message") == "xfail-marked test passes unexpectedly"
        ):
            xpassed += 1
            xpassed_as_skip += int(skipped is not None)
    failed = totals["failures"] + totals["errors"]
    passed = totals["tests"] - failed - totals["skipped"] - (xpassed - xpassed_as_skip)
    skipped = totals["skipped"] - xfailed - xpassed_as_skip
    if min(passed, skipped) < 0:
        raise ValueError("JUnit outcome counts contradict suite totals")
    return {
        "executed": passed + failed + xpassed,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
    }


def _load_local_manifest(path: Path, artifact_root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "version",
        "manifest_id",
        "owner_id",
        "build_input_digest",
        "source_tree_sha256",
        "compiler_sha256",
        "artifact_set_sha256",
        "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("local artifact manifest does not match the exact schema")
    if (
        manifest["schema"] != "tt.issue-solver.local-artifact-manifest"
        or manifest["version"] != 1
    ):
        raise ValueError("unsupported local artifact manifest")
    expected_id = _canonical_digest(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    )
    if manifest["manifest_id"] != expected_id:
        raise ValueError("local artifact manifest checksum mismatch")
    current_records = _artifact_records(artifact_root)
    current_digest = _canonical_digest(current_records)
    manifest["executed_artifact_sha256"] = current_digest
    manifest["artifact_mutated"] = (
        current_records != manifest["artifacts"]
        or current_digest != manifest["artifact_set_sha256"]
    )
    return manifest


def _classify_verification(
    collection: dict[str, int], execution: dict[str, Any]
) -> tuple[str, list[str]]:
    markers = execution["infrastructure_markers"]
    collection_nonzero = collection["returncode"] != 0 and not (
        collection["returncode"] == 5 and collection["selected"] == 0
    )
    if execution["timed_out"]:
        return "timed_out", ["execution_timed_out"]
    if collection_nonzero or collection["errors"] or markers:
        reasons = []
        if collection_nonzero:
            reasons.append("collection_nonzero_exit")
        if collection["errors"]:
            reasons.append("collection_error")
        reasons.extend(markers)
        return "infra_error", list(dict.fromkeys(reasons))
    if collection["selected"] == 0:
        return "coverage_error", ["zero_selected"]
    if execution["executed"] == 0:
        return "coverage_error", ["zero_executed"]
    if (
        execution["returncode"] == 0
        and execution["failed"] == 0
        and execution["xpassed"] == 0
        and execution["passed"] == execution["executed"]
    ):
        return "success", []
    if execution["returncode"] == 1 and execution["failed"] > 0:
        return "candidate_failure", ["test_failure"]
    if execution["returncode"] == 0:
        return "candidate_failure", ["outcome_count_mismatch"]
    if execution["signal"] is not None:
        return "infra_error", ["execution_signalled"]
    return "infra_error", ["execution_nonzero_exit"]


def cmd_verification_result(args: argparse.Namespace) -> int:
    collection = json.loads(Path(args.collection_json).read_text(encoding="utf-8"))
    expected_collection = {
        "schema",
        "version",
        "selected",
        "collected",
        "errors",
        "returncode",
    }
    if not isinstance(collection, dict) or set(collection) != expected_collection:
        raise ValueError("collection result does not match the exact schema")
    if (
        collection["schema"] != "tt.issue-solver.pytest-collection"
        or collection["version"] != 1
    ):
        raise ValueError("unsupported collection-result schema")
    for field in ("selected", "collected", "errors", "returncode"):
        value = collection[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (field != "returncode" and value < 0)
        ):
            raise ValueError(f"collection {field} is invalid")
    if collection["collected"] < collection["selected"]:
        raise ValueError("collection selected count exceeds collected count")

    marker_codes = list(args.infrastructure_code or [])
    output = ""
    if args.output_log:
        try:
            output = Path(args.output_log).read_text(encoding="utf-8", errors="replace")
        except OSError:
            marker_codes.append("execution_log_missing")
    marker_codes.extend(
        code
        for code, marker in _INFRASTRUCTURE_MARKERS_V1
        if marker.casefold() in output.casefold()
    )
    try:
        counts = _parse_junit(Path(args.junit))
    except ValueError:
        counts = {
            "executed": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
        }
        marker_codes.append("result_report_missing_or_invalid")

    manifest = _load_local_manifest(
        Path(args.artifact_manifest), Path(args.artifact_root)
    )
    if manifest["artifact_mutated"]:
        marker_codes.append("artifact_mutated_during_execution")
    marker_codes = list(dict.fromkeys(marker_codes))
    signal_number = args.signal
    if signal_number is None and 129 <= args.returncode <= 255:
        signal_number = args.returncode - 128
    execution = {
        "ran": counts["executed"] > 0,
        **counts,
        "returncode": args.returncode,
        "signal": signal_number,
        "timed_out": bool(args.timed_out),
        "infrastructure_markers": marker_codes,
    }
    normalized_collection = {
        "selected": collection["selected"],
        "collected": collection["collected"],
        "errors": collection["errors"],
        "returncode": collection["returncode"],
    }
    if (
        execution["executed"] + execution["skipped"] + execution["xfailed"]
        > normalized_collection["selected"]
    ):
        marker_codes.append("execution_count_exceeds_selection")
        execution["infrastructure_markers"] = list(dict.fromkeys(marker_codes))
        marker_codes = execution["infrastructure_markers"]
    if signal_number is not None and (
        signal_number < 1
        or args.returncode not in (-signal_number, 128 + signal_number)
    ):
        raise ValueError("signal number does not match the execution return code")

    classification, reasons = _classify_verification(normalized_collection, execution)

    for field, pattern in (
        ("expected_base_sha", _SHA40_RE),
        ("actual_base_sha", _SHA40_RE),
        ("patch_sha256", _SHA256_RE),
    ):
        if not pattern.fullmatch(getattr(args, field)):
            raise ValueError(f"{field} does not match the verification contract")
    for field in (
        "requirement_id",
        "run_id",
        "attempt_id",
        "job_id",
        "architecture",
        "suite",
        "backend",
        "test",
    ):
        if not isinstance(getattr(args, field), str) or not getattr(args, field):
            raise ValueError(f"{field} must be non-empty")
    result = {
        "schema": "tt.issue-solver.verification-result",
        "version": 2,
        "result_id": "0" * 64,
        "requirement_id": args.requirement_id,
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "job_id": args.job_id,
        "architecture": args.architecture,
        "suite": args.suite,
        "backend": args.backend,
        "selector": {"test": args.test, "test_id": args.test_id, "k": args.k},
        "provenance": {
            "expected_base_sha": args.expected_base_sha,
            "actual_base_sha": args.actual_base_sha,
            "patch_sha256": args.patch_sha256,
            "manifest_id": manifest["manifest_id"],
            "artifact_set_sha256": manifest["artifact_set_sha256"],
            "executed_artifact_sha256": manifest["executed_artifact_sha256"],
        },
        "collection": normalized_collection,
        "execution": execution,
        "classification": classification,
        "reason_codes": reasons,
    }
    result["result_id"] = _canonical_digest(
        {key: value for key, value in result.items() if key != "result_id"}
    )
    output_path = Path(args.output)
    _atomic_write(output_path.parent, result, destination=output_path)
    print(f"verification-result: {classification} -> {output_path}")
    return {
        "success": 0,
        "candidate_failure": 1,
        "coverage_error": 1,
        "infra_error": 3,
        "timed_out": 5,
    }[classification]


def _load_verification_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema",
        "version",
        "result_id",
        "requirement_id",
        "run_id",
        "attempt_id",
        "job_id",
        "architecture",
        "suite",
        "backend",
        "selector",
        "provenance",
        "collection",
        "execution",
        "classification",
        "reason_codes",
    }
    if (
        not isinstance(result, dict)
        or set(result) != fields
        or result["schema"] != "tt.issue-solver.verification-result"
        or result["version"] != 2
        or result["result_id"]
        != _canonical_digest(
            {key: value for key, value in result.items() if key != "result_id"}
        )
    ):
        raise ValueError("verification result does not match the exact v2 schema")
    for field in (
        "result_id",
        "requirement_id",
        "run_id",
        "attempt_id",
        "job_id",
        "architecture",
        "suite",
        "backend",
    ):
        if not isinstance(result[field], str) or not result[field]:
            raise ValueError(f"verification result {field} must be nonempty")
    if not _SHA256_RE.fullmatch(result["result_id"]):
        raise ValueError("verification result result_id is invalid")
    selector = result["selector"]
    if (
        not isinstance(selector, dict)
        or set(selector) != {"test", "test_id", "k"}
        or not isinstance(selector["test"], str)
        or not selector["test"]
        or any(
            selector[field] is not None and not isinstance(selector[field], str)
            for field in ("test_id", "k")
        )
        or (selector["test_id"] is not None and selector["k"] is not None)
    ):
        raise ValueError("verification result selector is invalid")
    provenance = result["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "expected_base_sha",
        "actual_base_sha",
        "patch_sha256",
        "manifest_id",
        "artifact_set_sha256",
        "executed_artifact_sha256",
    }:
        raise ValueError("verification result provenance is invalid")
    for field in ("expected_base_sha", "actual_base_sha"):
        if not isinstance(provenance[field], str) or not _SHA40_RE.fullmatch(
            provenance[field]
        ):
            raise ValueError(f"verification result provenance.{field} is invalid")
    for field in (
        "patch_sha256",
        "manifest_id",
        "artifact_set_sha256",
        "executed_artifact_sha256",
    ):
        if not isinstance(provenance[field], str) or not _SHA256_RE.fullmatch(
            provenance[field]
        ):
            raise ValueError(f"verification result provenance.{field} is invalid")
    collection = result["collection"]
    if not isinstance(collection, dict) or set(collection) != {
        "selected",
        "collected",
        "errors",
        "returncode",
    }:
        raise ValueError("verification result collection is invalid")
    for field in ("selected", "collected", "errors", "returncode"):
        value = collection[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (field != "returncode" and value < 0)
        ):
            raise ValueError(f"verification result collection.{field} is invalid")
    if collection["selected"] > collection["collected"]:
        raise ValueError("verification result selected exceeds collected")
    execution = result["execution"]
    count_fields = {
        "executed",
        "passed",
        "failed",
        "skipped",
        "xfailed",
        "xpassed",
    }
    if not isinstance(execution, dict) or set(execution) != {
        "ran",
        *count_fields,
        "returncode",
        "signal",
        "timed_out",
        "infrastructure_markers",
    }:
        raise ValueError("verification result execution is invalid")
    for field in count_fields:
        value = execution[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"verification result execution.{field} is invalid")
    if isinstance(execution["returncode"], bool) or not isinstance(
        execution["returncode"], int
    ):
        raise ValueError("verification result execution.returncode is invalid")
    if not isinstance(execution["ran"], bool) or not isinstance(
        execution["timed_out"], bool
    ):
        raise ValueError("verification result execution flags are invalid")
    signal_number = execution["signal"]
    if signal_number is not None and (
        isinstance(signal_number, bool)
        or not isinstance(signal_number, int)
        or signal_number < 1
        or execution["returncode"] not in (-signal_number, 128 + signal_number)
    ):
        raise ValueError("verification result execution.signal is invalid")
    markers = execution["infrastructure_markers"]
    if (
        not isinstance(markers, list)
        or any(not isinstance(value, str) or not value for value in markers)
        or len(set(markers)) != len(markers)
    ):
        raise ValueError("verification result infrastructure markers are invalid")
    if execution["ran"] != (execution["executed"] > 0):
        raise ValueError("verification result ran flag contradicts executed count")
    if execution["executed"] != (
        execution["passed"] + execution["failed"] + execution["xpassed"]
    ):
        raise ValueError("verification result executed count identity failed")
    if (
        execution["executed"] + execution["skipped"] + execution["xfailed"]
        > collection["selected"]
    ):
        raise ValueError("verification result outcomes exceed selected count")
    classification, reasons = _classify_verification(collection, execution)
    if result["classification"] != classification or result["reason_codes"] != reasons:
        raise ValueError("verification result classification contradicts evidence")
    return result


_REDUCTION_PRIORITY = {
    "success": 0,
    "partial": 1,
    "candidate_failure": 2,
    "coverage_error": 3,
    "infra_error": 4,
}


def _reduction_verdict(classification: str) -> str:
    if classification == "success":
        return "SUCCESS"
    if classification in {"coverage_error", "candidate_failure"}:
        return "TESTS_FAILED"
    return "ENV_ERROR"


def _load_perf_measurements(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _load_verification_reduction(path: Path) -> dict[str, Any]:
    reduction = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema",
        "version",
        "reduction_id",
        "scope",
        "manifest_id",
        "run_id",
        "attempt_id",
        "expected_base_sha",
        "patch_sha256",
        "classification",
        "reason_codes",
        "leaves",
        "excluded_results",
        "architecture_results",
        "tests_total",
        "tests_passed",
        "success_token",
    }
    if (
        not isinstance(reduction, dict)
        or set(reduction) != fields
        or reduction["schema"] != "tt.issue-solver.verification-reduction"
        or reduction["version"] != 1
        or reduction["reduction_id"]
        != _canonical_digest(
            {
                key: value
                for key, value in reduction.items()
                if key not in {"reduction_id", "success_token"}
            }
        )
    ):
        raise ValueError("verification reduction does not match the exact v1 schema")
    if reduction["scope"] not in {"functional", "all"}:
        raise ValueError("verification reduction scope is invalid")
    if reduction["classification"] not in _REDUCTION_PRIORITY:
        raise ValueError("verification reduction classification is invalid")
    for field, pattern in (
        ("reduction_id", _SHA256_RE),
        ("manifest_id", _SHA256_RE),
        ("expected_base_sha", _SHA40_RE),
    ):
        if not isinstance(reduction[field], str) or not pattern.fullmatch(
            reduction[field]
        ):
            raise ValueError(f"verification reduction {field} is invalid")
    for field in ("run_id", "attempt_id"):
        if not isinstance(reduction[field], str) or not reduction[field]:
            raise ValueError(f"verification reduction {field} is invalid")
    for field in ("patch_sha256", "success_token"):
        value = reduction[field]
        if value is not None and (
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
        ):
            raise ValueError(f"verification reduction {field} is invalid")
    if (
        not isinstance(reduction["reason_codes"], list)
        or any(
            not isinstance(value, str) or not value
            for value in reduction["reason_codes"]
        )
        or len(set(reduction["reason_codes"])) != len(reduction["reason_codes"])
        or not isinstance(reduction["leaves"], list)
        or not isinstance(reduction["excluded_results"], list)
        or not isinstance(reduction["architecture_results"], dict)
    ):
        raise ValueError("verification reduction aggregate fields are invalid")
    for field in ("tests_total", "tests_passed"):
        value = reduction[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"verification reduction {field} is invalid")
    if reduction["tests_passed"] > reduction["tests_total"]:
        raise ValueError("verification reduction passed count exceeds total")
    leaf_fields = {
        "requirement_id",
        "architecture",
        "suite",
        "result_id",
        "classification",
        "reason_codes",
        "selected",
        "executed",
        "passed",
        "failed",
        "skipped",
        "xfailed",
        "xpassed",
        "result_classification",
        "execution_returncode",
        "waived",
        "waiver_id",
    }
    requirement_ids = set()
    for leaf in reduction["leaves"]:
        if not isinstance(leaf, dict) or set(leaf) != leaf_fields:
            raise ValueError("verification reduction leaf schema is invalid")
        identity = leaf["requirement_id"]
        if not isinstance(identity, str) or not identity or identity in requirement_ids:
            raise ValueError("verification reduction leaf identity is invalid")
        requirement_ids.add(identity)
        if (
            not isinstance(leaf["architecture"], str)
            or not leaf["architecture"]
            or leaf["suite"] not in {"llk", "metal", "perf"}
            or leaf["classification"] not in _REDUCTION_PRIORITY
        ):
            raise ValueError("verification reduction leaf contract is invalid")
        if leaf["result_id"] is not None and (
            not isinstance(leaf["result_id"], str)
            or not _SHA256_RE.fullmatch(leaf["result_id"])
        ):
            raise ValueError("verification reduction leaf result_id is invalid")
        if (
            not isinstance(leaf["reason_codes"], list)
            or any(
                not isinstance(value, str) or not value
                for value in leaf["reason_codes"]
            )
            or len(set(leaf["reason_codes"])) != len(leaf["reason_codes"])
        ):
            raise ValueError("verification reduction leaf reasons are invalid")
        for field in (
            "selected",
            "executed",
            "passed",
            "failed",
            "skipped",
            "xfailed",
            "xpassed",
        ):
            value = leaf[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"verification reduction leaf {field} is invalid")
        outcome_count_incomplete = (
            leaf["classification"] == "coverage_error"
            and "execution_outcome_count_incomplete" in leaf["reason_codes"]
        )
        selection_count_incomplete = (
            leaf["selected"] != leaf["executed"] + leaf["skipped"] + leaf["xfailed"]
        )
        if (
            leaf["executed"] != leaf["passed"] + leaf["failed"] + leaf["xpassed"]
            or selection_count_incomplete != outcome_count_incomplete
        ):
            raise ValueError("verification reduction leaf counts are inconsistent")
        if leaf["result_classification"] is not None and (
            not isinstance(leaf["result_classification"], str)
            or leaf["result_classification"] not in {*_REDUCTION_PRIORITY, "timed_out"}
        ):
            raise ValueError("verification reduction source classification is invalid")
        if leaf["execution_returncode"] is not None and (
            isinstance(leaf["execution_returncode"], bool)
            or not isinstance(leaf["execution_returncode"], int)
        ):
            raise ValueError("verification reduction execution returncode is invalid")
        if not isinstance(leaf["waived"], bool):
            raise ValueError("verification reduction waived flag is invalid")
        if leaf["waived"]:
            if (
                not isinstance(leaf["waiver_id"], str)
                or not _SHA256_RE.fullmatch(leaf["waiver_id"])
                or leaf["classification"] != "success"
            ):
                raise ValueError("verification reduction waiver identity is invalid")
        elif leaf["waiver_id"] is not None:
            raise ValueError("verification reduction has an inactive waiver identity")
    if reduction["tests_total"] != sum(
        leaf["selected"] for leaf in reduction["leaves"]
    ) or reduction["tests_passed"] != sum(
        leaf["passed"] for leaf in reduction["leaves"]
    ):
        raise ValueError("verification reduction aggregate counts are inconsistent")
    for excluded in reduction["excluded_results"]:
        if (
            not isinstance(excluded, dict)
            or set(excluded) != {"result_id", "path", "reason"}
            or not isinstance(excluded["result_id"], str)
            or not _SHA256_RE.fullmatch(excluded["result_id"])
            or not isinstance(excluded["path"], str)
            or not excluded["path"]
            or not isinstance(excluded["reason"], str)
            or not excluded["reason"]
        ):
            raise ValueError("verification reduction excluded result is invalid")
    if reduction["classification"] == "success" and (
        not reduction["leaves"]
        or reduction["reason_codes"]
        or any(
            leaf["classification"] != "success" or leaf["result_id"] is None
            for leaf in reduction["leaves"]
        )
    ):
        raise ValueError("successful verification reduction is internally inconsistent")
    if reduction["success_token"] is not None and (
        reduction["scope"] != "all"
        or reduction["classification"] != "success"
        or reduction["patch_sha256"] is None
    ):
        raise ValueError("verification reduction success token is unauthorized")
    return reduction


def cmd_reduce_verification(args: argparse.Namespace) -> int:
    """Reduce sealed leaves into deterministic suite/architecture/final state."""
    log_dir = Path(args.log_dir)
    manifest = _load_required_manifest(Path(args.manifest))
    if manifest["waivers"]:
        if not args.worktree:
            raise ValueError("waived verification reduction requires --worktree")
        _verify_manifest_waiver_policy(
            manifest, Path(args.worktree).resolve(strict=True)
        )
    requirements = [
        requirement
        for requirement in manifest["requirements"]
        if args.scope == "all" or requirement["suite"] in {"llk", "metal"}
    ]
    all_required_by_id = {
        requirement["requirement_id"]: requirement
        for requirement in manifest["requirements"]
    }
    required_by_id = {
        requirement["requirement_id"]: requirement for requirement in requirements
    }
    results_dir = Path(args.results_dir)
    parsed: dict[str, dict[str, Any]] = {}
    excluded = []
    global_reasons = []
    if results_dir.exists():
        for path in sorted(results_dir.rglob("*.json")):
            relative_path = path.relative_to(results_dir).as_posix()
            try:
                result = _load_verification_result(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                global_reasons.append(f"malformed_result:{relative_path}:{exc}")
                continue
            result_id = result["result_id"]
            if result_id in parsed:
                excluded.append(
                    {
                        "result_id": result_id,
                        "path": relative_path,
                        "reason": "duplicate_result_copy",
                    }
                )
                continue
            parsed[result_id] = {"result": result, "path": relative_path}

    current: dict[str, list[dict[str, Any]]] = {
        identity: [] for identity in required_by_id
    }
    for result_id, entry in parsed.items():
        result = entry["result"]
        if (
            result["run_id"] != manifest["run_id"]
            or result["attempt_id"] != manifest["attempt_id"]
        ):
            excluded.append(
                {
                    "result_id": result_id,
                    "path": str(entry["path"]),
                    "reason": "superseded_or_foreign_attempt",
                }
            )
            continue
        identity = result["requirement_id"]
        if identity not in required_by_id:
            if identity in all_required_by_id:
                excluded.append(
                    {
                        "result_id": result_id,
                        "path": str(entry["path"]),
                        "reason": "out_of_scope_for_reduction",
                    }
                )
            else:
                global_reasons.append(f"unknown_requirement:{identity}")
            continue
        current[identity].append(entry)

    latest_perf = _load_perf_measurements(
        Path(args.perf_result) if args.perf_result else None
    )
    perf_by_arch: dict[str, dict[str, Any]] = {}
    if _run_json_path(log_dir).is_file():
        run_evidence = _load(log_dir)
        top_level_perf = run_evidence.get("perf")
        if isinstance(top_level_perf, dict) and isinstance(
            top_level_perf.get("arch"), str
        ):
            perf_by_arch[top_level_perf["arch"]] = top_level_perf
        for architecture, arch_result in (
            run_evidence.get("arch_results") or {}
        ).items():
            if isinstance(arch_result, dict) and isinstance(
                arch_result.get("perf"), dict
            ):
                perf_by_arch[architecture] = arch_result["perf"]
    if isinstance(latest_perf.get("arch"), str):
        perf_by_arch[latest_perf["arch"]] = latest_perf
    leaves = []
    patch_digests = set()
    for requirement in requirements:
        identity = requirement["requirement_id"]
        entries = current[identity]
        classification = "partial"
        reasons = []
        selected_result = None
        selected = executed = passed = failed = skipped = xfailed = xpassed = 0
        result_classification = None
        execution_returncode = None
        if not entries:
            reasons.append("result_missing")
        elif len(entries) > 1:
            classification = "infra_error"
            reasons.append("duplicate_current_result")
        else:
            selected_result = entries[0]["result"]
            result = selected_result
            collection = result["collection"]
            execution = result["execution"]
            selected = collection["selected"]
            executed = execution["executed"]
            passed = execution["passed"]
            failed = execution["failed"]
            skipped = execution["skipped"]
            xfailed = execution["xfailed"]
            xpassed = execution["xpassed"]
            result_classification = result["classification"]
            execution_returncode = execution["returncode"]
            mismatch_fields = []
            for field in (
                "architecture",
                "suite",
                "backend",
                "selector",
            ):
                if result[field] != requirement[field]:
                    mismatch_fields.append(field)
            if (
                result["provenance"]["expected_base_sha"]
                != manifest["expected_base_sha"]
            ):
                mismatch_fields.append("expected_base_sha")
            if result["provenance"]["actual_base_sha"] != manifest["expected_base_sha"]:
                mismatch_fields.append("actual_base_sha")
            if (
                result["provenance"]["artifact_set_sha256"]
                != result["provenance"]["executed_artifact_sha256"]
            ):
                mismatch_fields.append("executed_artifact_sha256")
            if mismatch_fields:
                classification = "infra_error"
                reasons.extend(
                    f"identity_mismatch:{field}" for field in mismatch_fields
                )
            elif result["classification"] == "timed_out":
                classification = "infra_error"
                reasons.append("execution_timed_out")
            elif result["classification"] == "infra_error":
                classification = "infra_error"
                reasons.extend(result["reason_codes"])
            elif selected == 0:
                classification = "coverage_error"
                reasons.append("zero_selected")
            elif executed == 0:
                classification = "coverage_error"
                reasons.append("zero_executed")
            elif selected < requirement["minimum_selected"]:
                classification = "coverage_error"
                reasons.append("minimum_selected_not_met")
            elif executed < requirement["minimum_executed"]:
                classification = "coverage_error"
                reasons.append("minimum_executed_not_met")
            elif (
                execution["executed"] + execution["skipped"] + execution["xfailed"]
                != selected
            ):
                classification = "coverage_error"
                reasons.append("execution_outcome_count_incomplete")
            elif execution["skipped"] or execution["xfailed"]:
                classification = "coverage_error"
                reasons.append("required_case_not_executed")
            else:
                classification = result["classification"]
                reasons.extend(result["reason_codes"])
            patch_digests.add(result["provenance"]["patch_sha256"])

        if classification == "success" and requirement["required_measurements"]:
            perf = perf_by_arch.get(requirement["architecture"], {})
            if (
                perf.get("outcome") != "PERF_OK"
                or perf.get("measured") is not True
                or perf.get("arch") != requirement["architecture"]
                or perf.get("test") != requirement["selector"]["test"]
                or perf.get("base_commit") != manifest["expected_base_sha"]
                or perf.get("run_id") != manifest["run_id"]
                or perf.get("attempt_id") != manifest["attempt_id"]
                or perf.get("requirement_id") != identity
                or perf.get("patch_sha256")
                != selected_result["provenance"]["patch_sha256"]
            ):
                classification = "coverage_error"
                reasons.append("required_measurement_result_missing_or_invalid")
            measurements = perf.get("measurements") or {}
            for measurement in requirement["required_measurements"]:
                evidence = measurements.get(measurement)
                if (
                    not isinstance(evidence, dict)
                    or evidence.get("measured") is not True
                ):
                    classification = "coverage_error"
                    reasons.append(f"required_measurement_missing:{measurement}")
                    continue
                if measurement == "repeatability" and (
                    isinstance(evidence.get("executions"), bool)
                    or not isinstance(evidence.get("executions"), int)
                    or evidence["executions"] < requirement["minimum_executed"]
                ):
                    classification = "coverage_error"
                    reasons.append("repeatability_execution_count_not_met")

        leaves.append(
            {
                "requirement_id": identity,
                "architecture": requirement["architecture"],
                "suite": requirement["suite"],
                "result_id": selected_result["result_id"] if selected_result else None,
                "classification": classification,
                "reason_codes": list(dict.fromkeys(reasons)),
                "selected": selected,
                "executed": executed,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "xfailed": xfailed,
                "xpassed": xpassed,
                "result_classification": result_classification,
                "execution_returncode": execution_returncode,
                "waived": False,
                "waiver_id": None,
            }
        )

    leaves_by_id = {leaf["requirement_id"]: leaf for leaf in leaves}
    for waiver in manifest["waivers"]:
        scope = leaves_by_id[waiver["scope_requirement_id"]]
        replacement = leaves_by_id[waiver["replacement_requirement_id"]]
        requirement = required_by_id[scope["requirement_id"]]
        observed_outcomes = {
            outcome
            for outcome, count in (
                ("skipped", scope["skipped"]),
                ("xfailed", scope["xfailed"]),
            )
            if count
        }
        if (
            replacement["classification"] == "success"
            and scope["classification"] == "coverage_error"
            and set(scope["reason_codes"])
            <= {
                "zero_executed",
                "minimum_executed_not_met",
                "required_case_not_executed",
            }
            and scope["result_id"] is not None
            and scope["result_classification"]
            not in {"infra_error", "timed_out", "candidate_failure"}
            and scope["execution_returncode"] == 0
            and scope["failed"] == 0
            and scope["selected"] >= requirement["minimum_selected"]
            and scope["executed"] + scope["skipped"] + scope["xfailed"]
            == scope["selected"]
            and observed_outcomes
            and observed_outcomes <= set(waiver["allowed_outcomes"])
        ):
            scope["classification"] = "success"
            # Keep the established success contract (no obstacle reason codes).
            # Waiver use remains explicit in the sealed leaf identity below.
            scope["reason_codes"] = []
            scope["waived"] = True
            scope["waiver_id"] = waiver["waiver_id"]

    if len(patch_digests) > 1:
        global_reasons.append("patch_digest_mismatch")
    if not leaves:
        classification = "partial"
        global_reasons.append("no_requirements_in_scope")
    elif global_reasons:
        classification = "infra_error"
    elif all(leaf["classification"] == "success" for leaf in leaves):
        classification = "success"
    else:
        classification = max(
            (leaf["classification"] for leaf in leaves),
            key=lambda value: _REDUCTION_PRIORITY[value],
        )
    reason_codes = list(dict.fromkeys(global_reasons))
    reason_codes.extend(
        f"{leaf['requirement_id']}:{reason}"
        for leaf in leaves
        for reason in leaf["reason_codes"]
    )
    reason_codes = list(dict.fromkeys(reason_codes))

    architecture_results: dict[str, Any] = {}
    for architecture in sorted({item["architecture"] for item in requirements}):
        arch_leaves = [item for item in leaves if item["architecture"] == architecture]
        suites = {}
        for suite in sorted({item["suite"] for item in arch_leaves}):
            suite_leaves = [item for item in arch_leaves if item["suite"] == suite]
            suite_classification = max(
                (item["classification"] for item in suite_leaves),
                key=lambda value: _REDUCTION_PRIORITY[value],
            )
            suite_reasons = [
                f"{item['requirement_id']}:{reason}"
                for item in suite_leaves
                for reason in item["reason_codes"]
            ]
            suites[suite] = {
                "status": (
                    "done"
                    if all(item["result_id"] for item in suite_leaves)
                    else "missing"
                ),
                "verdict": _reduction_verdict(suite_classification),
                "classification": suite_classification,
                "tests_total": sum(item["selected"] for item in suite_leaves),
                "tests_passed": sum(item["passed"] for item in suite_leaves),
                "result_ids": [
                    item["result_id"] for item in suite_leaves if item["result_id"]
                ],
                "obstacle": "; ".join(suite_reasons) or None,
            }
        arch_classification = max(
            (item["classification"] for item in arch_leaves),
            key=lambda value: _REDUCTION_PRIORITY[value],
        )
        architecture_results[architecture] = {
            "status": (
                "done" if all(item["result_id"] for item in arch_leaves) else "missing"
            ),
            "verdict": _reduction_verdict(arch_classification),
            "classification": arch_classification,
            "tests_total": sum(item["selected"] for item in arch_leaves),
            "tests_passed": sum(item["passed"] for item in arch_leaves),
            "suite_results": suites,
            "obstacle": "; ".join(
                f"{item['requirement_id']}:{reason}"
                for item in arch_leaves
                for reason in item["reason_codes"]
            )
            or None,
        }

    if global_reasons:
        global_obstacle = "; ".join(global_reasons)
        for architecture in architecture_results.values():
            architecture["verdict"] = "ENV_ERROR"
            architecture["classification"] = "infra_error"
            architecture["obstacle"] = "; ".join(
                value for value in (global_obstacle, architecture["obstacle"]) if value
            )

    reduction = {
        "schema": "tt.issue-solver.verification-reduction",
        "version": 1,
        "reduction_id": "0" * 64,
        "scope": args.scope,
        "manifest_id": manifest["manifest_id"],
        "run_id": manifest["run_id"],
        "attempt_id": manifest["attempt_id"],
        "expected_base_sha": manifest["expected_base_sha"],
        "patch_sha256": next(iter(patch_digests)) if len(patch_digests) == 1 else None,
        "classification": classification,
        "reason_codes": reason_codes,
        "leaves": leaves,
        "excluded_results": excluded,
        "architecture_results": architecture_results,
        "tests_total": sum(item["selected"] for item in leaves),
        "tests_passed": sum(item["passed"] for item in leaves),
        "success_token": None,
    }
    reduction["reduction_id"] = _canonical_digest(
        {
            key: value
            for key, value in reduction.items()
            if key not in {"reduction_id", "success_token"}
        }
    )
    if args.scope == "all" and classification == "success" and patch_digests:
        reduction["success_token"] = _canonical_digest(
            {
                "schema": "tt.issue-solver.verification-success-token",
                "version": 1,
                "reduction_id": reduction["reduction_id"],
                "manifest_id": manifest["manifest_id"],
                "run_id": manifest["run_id"],
                "attempt_id": manifest["attempt_id"],
                "patch_sha256": reduction["patch_sha256"],
            }
        )

    output = (
        Path(args.output) if args.output else log_dir / "verification_reduction.json"
    )
    retained = log_dir / "verification_reductions" / f"{reduction['reduction_id']}.json"
    if not retained.exists():
        _atomic_write(retained.parent, reduction, destination=retained)
    _atomic_write(output.parent, reduction, destination=output)
    if _run_json_path(log_dir).is_file():
        patch = {
            "arch_results": architecture_results,
            "tests_total": reduction["tests_total"],
            "tests_passed": reduction["tests_passed"],
            "verification_reduction": {
                "reduction_id": reduction["reduction_id"],
                "manifest_id": reduction["manifest_id"],
                "scope": reduction["scope"],
                "classification": reduction["classification"],
                "reason_codes": reduction["reason_codes"],
                "success_token": reduction["success_token"],
            },
        }
        with _run_json_transaction(log_dir) as run:
            _deep_merge(run, patch)
    print(
        f"verification-reduction: {args.scope} {classification} "
        f"({len(leaves)} leaves) -> {output}"
    )
    # A completed reduction is successful command execution. Test failures
    # remain data in the reduction and are handled by the orchestrator/finalizer.
    return 0


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
    init.add_argument(
        "--version",
        default=None,
        help=(
            "Semantic version (major.minor.patch) of the pipeline producing this "
            "run. Optional; defaults to null."
        ),
    )
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
    init.add_argument(
        "--description",
        default=None,
        help="Short one-line description (e.g. issue title) shown in dashboard rows",
    )
    init.add_argument("--phases-total", type=int, default=0)
    init.add_argument(
        "--pipeline-steps", default=None, help="JSON array of {id,name,desc} objects"
    )
    init.add_argument("--issue", default=None, help="JSON object for issue-solver runs")
    init.add_argument(
        "--issue-run-id",
        default=None,
        help="Shared ID across N per-arch runs of one multi-arch issue (optional)",
    )
    init.add_argument(
        "--sibling-runs",
        default=None,
        help=(
            "JSON array of {arch, run_id} pointers to other per-arch runs in "
            "the same issue (optional; defaults to [])"
        ),
    )
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

    # link-siblings --------------------------------------------------------
    ls = sub.add_parser(
        "link-siblings",
        help="Patch issue_run_id and sibling_runs on an existing run.json",
    )
    _add_common(ls)
    ls.add_argument(
        "--siblings",
        required=True,
        help="JSON array of {arch, run_id} objects (may be empty)",
    )
    ls.add_argument(
        "--issue-run-id",
        default=None,
        help="Shared ID across N per-arch runs (optional; unchanged if omitted)",
    )
    ls.set_defaults(func=cmd_link_siblings)

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
    fz.add_argument(
        "--worktree",
        default=None,
        help="Packaged worktree used to validate an audit success patch",
    )
    fz.set_defaults(func=cmd_finalize)

    manifest = sub.add_parser(
        "artifact-manifest",
        help="Seal a local LLK artifact directory before execution",
    )
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--artifact-root", required=True)
    manifest.add_argument("--owner-id", required=True)
    manifest.add_argument("--build-input-digest", required=True)
    manifest.add_argument("--source-tree-sha256", required=True)
    manifest.add_argument("--compiler-sha256", required=True)
    manifest.set_defaults(func=cmd_artifact_manifest)

    candidate = sub.add_parser(
        "candidate-patch-digest",
        help="Hash the complete base-to-worktree candidate using a temporary index",
    )
    candidate.add_argument("--worktree", required=True)
    candidate.add_argument("--expected-base-sha", required=True)
    candidate.set_defaults(func=cmd_candidate_patch_digest)

    required = sub.add_parser(
        "required-verification",
        help="Normalize analysis and fix-plan selectors into an immutable manifest",
    )
    _add_common(required)
    required.add_argument("--output", required=True)
    required.add_argument("--analysis", required=True)
    required.add_argument("--plan", required=True)
    required.add_argument("--worktree", required=True)
    required.add_argument("--run-id", required=True)
    required.add_argument("--expected-base-sha", required=True)
    required.add_argument("--architectures-json", required=True)
    required.add_argument("--backend", required=True, choices=["local", "ttsim"])
    required.add_argument("--supersedes-reason", default=None)
    required.add_argument(
        "--waiver-policy",
        default=None,
        help=(
            "optional policy JSON tracked in expected_base_sha; accepted only on "
            "the first manifest revision and carried unchanged across retries"
        ),
    )
    required.add_argument(
        "--performance-only",
        action="store_true",
        help="seal only explicit perf leaves when a hypothesis was refuted",
    )
    required.set_defaults(func=cmd_required_verification)

    result = sub.add_parser(
        "verification-result",
        help="Write a strict result from structured collection and JUnit evidence",
    )
    result.add_argument("--output", required=True)
    result.add_argument("--collection-json", required=True)
    result.add_argument("--junit", required=True)
    result.add_argument("--output-log", default=None)
    result.add_argument("--artifact-manifest", required=True)
    result.add_argument("--artifact-root", required=True)
    result.add_argument("--requirement-id", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--attempt-id", required=True)
    result.add_argument("--job-id", required=True)
    result.add_argument("--architecture", required=True)
    result.add_argument("--suite", required=True)
    result.add_argument(
        "--backend", required=True, choices=["silicon", "ttsim", "quasar", "local"]
    )
    result.add_argument("--test", required=True)
    result.add_argument("--test-id", default=None)
    result.add_argument("--k", default=None)
    result.add_argument("--expected-base-sha", required=True)
    result.add_argument("--actual-base-sha", required=True)
    result.add_argument("--patch-sha256", required=True)
    result.add_argument("--returncode", required=True, type=int)
    result.add_argument("--signal", default=None, type=int)
    result.add_argument("--timed-out", action="store_true")
    result.add_argument("--infrastructure-code", action="append", default=[])
    result.set_defaults(func=cmd_verification_result)

    reduce_result = sub.add_parser(
        "reduce-verification",
        help="Reduce sealed result leaves into deterministic run evidence",
    )
    _add_common(reduce_result)
    reduce_result.add_argument("--manifest", required=True)
    reduce_result.add_argument("--results-dir", required=True)
    reduce_result.add_argument("--scope", required=True, choices=["functional", "all"])
    reduce_result.add_argument("--perf-result", default=None)
    reduce_result.add_argument("--output", default=None)
    reduce_result.add_argument(
        "--worktree",
        default=None,
        help="candidate worktree used to reopen base-tracked waiver policy",
    )
    reduce_result.set_defaults(func=cmd_reduce_verification)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"run_json_writer error: {exc}", file=sys.stderr)
        return 1
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
