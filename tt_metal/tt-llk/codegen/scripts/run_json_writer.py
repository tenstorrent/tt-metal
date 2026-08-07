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
    verification-result
                    Write a strict v2 result from collection/JUnit/artifact evidence.
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
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import xml.etree.ElementTree as ET
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


def _atomic_write(
    log_dir: Path, doc: dict[str, Any], *, destination: Path | None = None
) -> None:
    path = destination or _run_json_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    _merge_patch(doc, patch)
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
    doc["final_result"] = args.final_result
    doc["final_message"] = args.final_message or None
    doc["current_step_message"] = (
        args.final_message or doc.get("current_step_message") or ""
    )

    patch = _json_arg(args.patch_json, {})
    _merge_patch(doc, patch)

    # Apply typed --solver-state last so it cannot be silently overridden by
    # --patch-json (argparse choices are otherwise bypassed via that escape hatch).
    if args.solver_state is not None:
        doc["solver_state"] = args.solver_state

    _atomic_write(log_dir, doc)
    print(f"finalize: status={args.status}")


# --------------------------------------------------------------------------
# Subcommand: link-siblings (multi-arch grouping, optional)
# --------------------------------------------------------------------------


def cmd_link_siblings(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    doc = _load(log_dir)

    siblings = _json_arg(args.siblings, [])
    if not isinstance(siblings, list):
        raise SystemExit("--siblings must be a JSON array")

    doc["sibling_runs"] = siblings
    if args.issue_run_id is not None:
        doc["issue_run_id"] = args.issue_run_id

    _atomic_write(log_dir, doc)
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


def _load_required_manifest(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != "tt.issue-solver.required-verification"
        or doc.get("version") != 1
        or doc.get("manifest_id")
        != _canonical_digest(
            {key: value for key, value in doc.items() if key != "manifest_id"}
        )
    ):
        raise ValueError(f"invalid required-verification manifest: {path}")
    if doc.get("waivers") != []:
        raise ValueError("agent-created verification waivers are unsupported")
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
    ) -> None:
        key = (arch, suite)
        if any(
            item["architecture"] == arch
            and item["suite"] == suite
            and item["selector"] == selector
            for item in requirements
        ):
            raise ValueError(f"duplicate {suite} selector for {arch}: {selector!r}")
        counters[key] = counters.get(key, 0) + 1
        requirements.append(
            {
                "requirement_id": f"{arch}:{suite}:{counters[key]}",
                "architecture": arch,
                "suite": suite,
                "backend": backend,
                "selector": selector,
                "minimum_selected": minimum_selected,
                "minimum_executed": minimum_executed,
                "required_measurements": measurements or [],
            }
        )

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
        "waivers": [],
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

    collection_nonzero = normalized_collection["returncode"] != 0 and not (
        normalized_collection["returncode"] == 5
        and normalized_collection["selected"] == 0
    )
    if execution["timed_out"]:
        classification, reasons = "timed_out", ["execution_timed_out"]
    elif collection_nonzero or normalized_collection["errors"] or marker_codes:
        reasons = []
        if collection_nonzero:
            reasons.append("collection_nonzero_exit")
        if normalized_collection["errors"]:
            reasons.append("collection_error")
        reasons.extend(marker_codes)
        classification, reasons = "infra_error", list(dict.fromkeys(reasons))
    elif normalized_collection["selected"] == 0:
        classification, reasons = "coverage_error", ["zero_selected"]
    elif execution["executed"] == 0:
        classification, reasons = "coverage_error", ["zero_executed"]
    elif (
        execution["returncode"] == 0
        and execution["failed"] == 0
        and execution["xpassed"] == 0
        and execution["passed"] == execution["executed"]
    ):
        classification, reasons = "success", []
    elif execution["returncode"] == 1 and execution["failed"] > 0:
        classification, reasons = "candidate_failure", ["test_failure"]
    elif execution["returncode"] == 0:
        classification, reasons = "candidate_failure", ["outcome_count_mismatch"]
    elif signal_number is not None:
        classification, reasons = "infra_error", ["execution_signalled"]
    else:
        classification, reasons = "infra_error", ["execution_nonzero_exit"]

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
