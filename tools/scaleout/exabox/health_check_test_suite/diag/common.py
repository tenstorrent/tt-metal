# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the phases: paths, subprocesses and console output."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .model import IP_ORDER

# tools/scaleout/exabox/health_check_test_suite
SUITE_DIR = Path(__file__).resolve().parents[1]


def default_tt_metal_path() -> Path:
    """Repo root: $TT_METAL_HOME, else four levels up from the suite directory
    (tools/scaleout/exabox/health_check_test_suite -> repo root)."""
    return Path(os.environ.get("TT_METAL_HOME") or SUITE_DIR.parents[3])


# ─────────────────────────────────────────────────────────────────────────────
# Subprocesses
# ─────────────────────────────────────────────────────────────────────────────


def kill_process_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM then SIGKILL the child's process group."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


def run(cmd: list[str], dry_run: bool = False, **kw) -> subprocess.CompletedProcess:
    """Run a command. In dry_run, print and return a stubbed CompletedProcess."""
    log(f"$ {' '.join(cmd)}")
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, **kw)


HOOK_TIMEOUT_S = 1800


def call_hook(
    hook: str | None, event: str, step: str, status: str, outdir: Path, dry_run: bool, cwd: Path | None = None
) -> int:
    """Run --hook as SCRIPT <event> <step> <status> <outdir> from cwd: its rc (0 with no hook or in a dry run)."""
    if not hook:
        return 0
    if dry_run:
        log(f"[dry-run] {hook} {event} {step} {status} {outdir}")
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    # Paths are made absolute here, before cwd moves; a bare name is still looked up on PATH.
    cmd = [os.path.abspath(hook) if "/" in hook else hook, event, step, status, str(outdir.absolute())]
    try:  # a site power cycle and boot fit in the bound; a hung hook is rc 124
        return subprocess.run(cmd, check=False, timeout=HOOK_TIMEOUT_S, cwd=cwd).returncode
    except subprocess.TimeoutExpired:
        log(f"hook {hook} {event} {step} timed out after {HOOK_TIMEOUT_S} s")
        return 124
    except OSError as e:
        log(f"hook {hook} did not run: {e}")
        return 127


def as_text(v) -> str:
    """TimeoutExpired carries whatever was buffered; normalize bytes-or-str."""
    if v is None:
        return ""
    return v.decode(errors="replace") if isinstance(v, bytes) else str(v)


def run_with_timeout(cmd: list[str], timeout_s: float, **kw) -> tuple[int, str, str, bool]:
    """Run an external tool under a wall-clock bound.

    Returns ``(rc, stdout, stderr, timed_out)``. A timeout is rc 124 with
    whatever the tool had written by then. OSError (the tool could not be
    started) propagates, since each phase reports that case in its own words.
    """
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, **kw)
    except subprocess.TimeoutExpired as e:
        return 124, as_text(e.stdout), as_text(e.stderr), True
    return cp.returncode, cp.stdout or "", cp.stderr or "", False


def write_command_log(path: Path, cmdline: str, rc: int, duration_s: float, stdout: str, stderr: str) -> bool:
    """Write one command's full output to ``path``. False if it could not be
    written: a lost log is reported and costs nothing else."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"$ {cmdline}\nrc={rc}  duration={duration_s:.1f}s\n"
            f"\n--- stdout ---\n{stdout or '(empty)'}"
            f"\n--- stderr ---\n{stderr or '(empty)'}\n"
        )
    except OSError as e:
        log(f"  could not write {path}: {e!r}")
        return False
    return True


def resolve_optional_tool(override: str | None, binary: str, flag: str, absent: str) -> tuple[str | None, str]:
    """An optional tool's binary, or None plus the reason its phase can't run.

    A reason rather than an exception: the tool is optional, so taking a whole
    diagnostic run down over a package that isn't installed would trade real
    coverage for none. An override that doesn't resolve is reported as its own
    case, since an operator who passed a path meant it and should not have it
    silently ignored.
    """
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate), ""
        return None, f"{flag}={override} is not an executable file"
    found = shutil.which(binary)
    if found:
        return found, ""
    return None, absent


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(f"[diag] {msg}", file=sys.stderr, flush=True)


_USE_INPLACE_PROGRESS = sys.stdout.isatty()


def emit_running(name: str) -> None:
    """Print 'running...' line. If stdout is a TTY we'll overwrite it with \\r;
    otherwise just print on its own line so the captured log stays readable."""
    if _USE_INPLACE_PROGRESS:
        sys.stdout.write(f"  {name:30} running...")
    else:
        sys.stdout.write(f"  {name:30} running...\n")
    sys.stdout.flush()


def emit_result(name: str, status: str, suffix: str = "") -> None:
    """Print final result. On TTY, \\r overwrites the running... line."""
    prefix = "\r" if _USE_INPLACE_PROGRESS else ""
    extra = f"  {suffix}" if suffix else ""
    sys.stdout.write(f"{prefix}  {name:30} {status:5}{extra}\n")
    sys.stdout.flush()


def print_phase_summary(phase_name: str, phase_dict: dict) -> None:
    """Print one phase's grouped per-IP summary to stdout (flushed).

    Multi-line check details (containing '\\n') render with continuation lines
    indented under the first detail line — used by bdf_topology to print its
    per-UBB mapping table without breaking the columnar layout.
    """
    lines = [f"  {phase_name:14} {phase_dict['status']:5} ({phase_dict['duration_s']:.1f}s)"]
    visible = [c for c in phase_dict["checks"] if c.get("console_visible", True)]
    by_ip: dict[str, list] = defaultdict(list)
    for c in visible:
        by_ip[c.get("ip", "other")].append(c)
    for ip in IP_ORDER:
        checks = by_ip.get(ip, [])
        if not checks:
            continue
        for i, c in enumerate(checks):
            head = f"{ip}:" if i == 0 else ""
            detail_lines = (c["details"] or "").split("\n")
            lines.append(f"    {head:7} {c['name']:32} {c['status']:5}  {detail_lines[0]}")
            for extra in detail_lines[1:]:
                lines.append(f"      {extra}")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def print_phase_line(phase_dict: dict) -> None:
    """The phase's rollup on one line, for phases that print their own
    per-step lines as they run."""
    print(f"  {phase_dict['name']:14} {phase_dict['status']:5} ({phase_dict['duration_s']:.1f}s)", flush=True)
