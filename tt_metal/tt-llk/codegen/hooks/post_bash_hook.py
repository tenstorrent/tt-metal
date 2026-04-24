#!/usr/bin/env python3
"""
PostToolUse hook — captures Bash tool outcomes for compiler.py and run_llk_tests.sh.

Receives a JSON payload on stdin (Claude Code PostToolUse format):
  tool_name       "Bash"
  tool_input      {"command": "...", "description": "..."}
  tool_response   combined output string (may contain "Exit code: N" for failures)
  cwd             current working directory of the Claude session

Behaviour:
  compiler.py success      → increment compile_successes in run.json
                           → write compile_success_{N}.txt artifact
  compiler.py failure      → increment compile_failures in run.json
                           → write compile_failure_{N}.txt artifact
  run_llk_tests.sh compile success   → compile_successes++, compile_success_{N}.txt
  run_llk_tests.sh simulate success  → run_successes++, run_success_{N}/test_run.txt
  run_llk_tests.sh run success       → both compile_successes++ and run_successes++
  run_llk_tests.sh exit 2  → compile_failures++, compile_failure_{N}.txt
  run_llk_tests.sh exit 1/3→ run_failures++, failed_attempt_{N}/test_run.txt

LOG_DIR is read from /tmp/codegen_run_state.sh (written by the orchestrator Step 0).
If the state file is absent or LOG_DIR does not exist, the hook exits silently.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Run-state helpers
# ---------------------------------------------------------------------------


def _load_run_state() -> dict[str, str]:
    state_file = Path("/tmp/codegen_run_state.sh")
    if not state_file.exists():
        return {}
    state: dict[str, str] = {}
    for line in state_file.read_text().splitlines():
        m = re.match(r'^export\s+(\w+)="([^"]*)"', line)
        if m:
            state[m.group(1)] = m.group(2)
    return state


def _log_dir() -> Path | None:
    state = _load_run_state()
    raw = state.get("LOG_DIR", "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _tool_output(payload: dict) -> str:
    resp = payload.get("tool_response", "")
    if isinstance(resp, dict):
        return resp.get("stdout", "") + "\n" + resp.get("stderr", "")
    return str(resp)


def _exit_code(output: str) -> int | None:
    """Extract exit code from 'Exit code: N' that Claude Code appends on non-zero exits."""
    m = re.search(r"Exit code:\s*(\d+)", output)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# run.json — counter updates (compile_failures, compile_successes, run_successes, run_failures)
# ---------------------------------------------------------------------------


def _increment_counter(log_dir: Path, field: str) -> int:
    run_json = log_dir / "run.json"
    if not run_json.exists():
        return 0
    try:
        doc = json.loads(run_json.read_text())
        new_val = doc.get(field, 0) + 1
        doc[field] = new_val
        fd, tmp = tempfile.mkstemp(prefix=".run.json.", suffix=".tmp", dir=log_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f, indent=2)
                f.write("\n")
            os.chmod(tmp, 0o664)
            os.replace(tmp, run_json)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return new_val
    except Exception as exc:
        print(f"[hook] {field} update failed: {exc}", file=sys.stderr)
        return 0


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _next_n(log_dir: Path, prefix: str) -> int:
    return len(list(log_dir.glob(f"{prefix}*"))) + 1


def _write_compile_failure(log_dir: Path, command: str, output: str) -> Path:
    n = _next_n(log_dir, "compile_failure_")
    artifact = log_dir / f"compile_failure_{n}.txt"
    artifact.write_text(f"Command:\n{command}\n\nOutput:\n{output}\n")
    os.chmod(artifact, 0o664)
    return artifact


def _write_compile_success(log_dir: Path, command: str, output: str) -> Path:
    n = _next_n(log_dir, "compile_success_")
    artifact = log_dir / f"compile_success_{n}.txt"
    artifact.write_text(f"Command:\n{command}\n\nOutput:\n{output}\n")
    os.chmod(artifact, 0o664)
    return artifact


def _write_failed_attempt(
    log_dir: Path, command: str, output: str, reason: str
) -> Path:
    n = _next_n(log_dir, "failed_attempt_")
    attempt_dir = log_dir / f"failed_attempt_{n}"
    attempt_dir.mkdir(exist_ok=True)
    run_file = attempt_dir / "test_run.txt"
    run_file.write_text(
        f"Reason: {reason}\n\nCommand:\n{command}\n\nOutput:\n{output}\n"
    )
    os.chmod(run_file, 0o664)
    return attempt_dir


def _write_run_success(log_dir: Path, command: str, output: str) -> Path:
    n = _next_n(log_dir, "run_success_")
    attempt_dir = log_dir / f"run_success_{n}"
    attempt_dir.mkdir(exist_ok=True)
    run_file = attempt_dir / "test_run.txt"
    run_file.write_text(f"Command:\n{command}\n\nOutput:\n{output}\n")
    os.chmod(run_file, 0o664)
    return attempt_dir


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------


def _compiler_py_failed(output: str, exit_code: int | None) -> bool:
    if exit_code is not None:
        return exit_code != 0
    # Fallback: look for error patterns when exit code wasn't captured
    if re.search(r"\berror\b", output, re.IGNORECASE):
        # Exclude lines that indicate success despite containing "error"
        if not re.search(
            r"(0 error|Environment OK|compilation successful)", output, re.IGNORECASE
        ):
            return True
    return False


def _test_runner_outcome(output: str, exit_code: int | None) -> str | None:
    """Return 'compile_error', 'test_failure', 'timeout', or None (pass)."""
    # Explicit exit codes from run_llk_tests.sh:
    #   0 = pass, 1 = test failure, 2 = compile error, 3 = env/timeout error
    if exit_code == 2:
        return "compile_error"
    if exit_code == 3:
        return "timeout"
    if exit_code == 1:
        # Distinguish timeout from test failure by output content
        if re.search(
            r"(timed out|timeout|Could not acquire simulator lock)",
            output,
            re.IGNORECASE,
        ):
            return "timeout"
        return "test_failure"
    if exit_code == 0:
        return None
    # exit_code unknown — use content heuristics
    if re.search(r"ERROR: compile step failed", output):
        return "compile_error"
    if re.search(r"Could not acquire simulator lock", output, re.IGNORECASE):
        return "timeout"
    if re.search(r"\bFAILED\b", output):
        return "test_failure"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    if payload.get("tool_name") != "Bash":
        return

    command: str = payload.get("tool_input", {}).get("command", "")
    output = _tool_output(payload)
    exit_code = _exit_code(output)

    log_dir = _log_dir()
    if log_dir is None:
        return  # not inside a codegen run

    # ── compiler.py ──────────────────────────────────────────────────────────
    if re.search(r"scripts/compiler\.py", command):
        if _compiler_py_failed(output, exit_code):
            n = _increment_counter(log_dir, "compile_failures")
            artifact = _write_compile_failure(log_dir, command, output)
            print(f"[hook] compiler.py failure #{n} → {artifact.name}")
        else:
            n = _increment_counter(log_dir, "compile_successes")
            artifact = _write_compile_success(log_dir, command, output)
            print(f"[hook] compiler.py success #{n} → {artifact.name}")

    # ── run_llk_tests.sh ─────────────────────────────────────────────────────
    elif re.search(r"run_llk_tests\.sh", command):
        outcome = _test_runner_outcome(output, exit_code)

        # Detect the subcommand (compile | simulate | run | count | …)
        m = re.search(r"run_llk_tests\.sh\s+(\w+)", command)
        subcommand = m.group(1) if m else "run"

        if outcome == "compile_error":
            n = _increment_counter(log_dir, "compile_failures")
            artifact = _write_compile_failure(log_dir, command, output)
            print(f"[hook] run_llk_tests compile failure #{n} → {artifact.name}")
        elif outcome in ("test_failure", "timeout"):
            n = _increment_counter(log_dir, "run_failures")
            attempt_dir = _write_failed_attempt(log_dir, command, output, outcome)
            print(f"[hook] {outcome} → {attempt_dir.name}/")
        else:  # outcome is None → success
            # compile or run subcommand succeeded → track compile_successes
            if subcommand in ("compile", "run"):
                n = _increment_counter(log_dir, "compile_successes")
                artifact = _write_compile_success(log_dir, command, output)
                print(f"[hook] compile success #{n} → {artifact.name}")
            # simulate or run subcommand succeeded → track run_successes
            if subcommand in ("simulate", "run"):
                n = _increment_counter(log_dir, "run_successes")
                attempt_dir = _write_run_success(log_dir, command, output)
                print(f"[hook] run pass #{n} → {attempt_dir.name}/")


if __name__ == "__main__":
    main()
