# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host event captures: the BMC SEL and the kernel log since the run began. Reads only, and evidence rather than
findings: the phase never votes; the verdict cites these under the faults they explain."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from .common import run, write_command_log
from .galaxy import name_chips
from .model import PASS, SKIP, WARN, Check, Phase

# SEL records that bear on a lost chip, tray or host: boot failures (FRB), power, voltage, AC loss, watchdog.
SEL_POWER = "AC lost|Power Supply|Power Unit"
SEL_POWER_RE = re.compile(SEL_POWER, re.IGNORECASE)
SEL_EVENT_RE = re.compile(rf"FRB|{SEL_POWER}|Voltage|Critical|Watchdog", re.IGNORECASE)

# Kernel lines for a misbehaving PCIe link or driver.
KERNEL_EVENT_RE = re.compile(r"AER|pciehp|Link Down|tenstorrent.*(error|fail|timeout)", re.IGNORECASE)

# `ipmitool sel elist` dates: "   7 | 11/20/2026 | 10:00:01 | ...".
SEL_DATE_RE = re.compile(r"^\s*\w+\s*\|\s*(\d\d/\d\d/\d{4})\s*\|")


def _read(cmd: list[str], dry_run: bool) -> tuple[str | None, str]:
    """A privileged read's stdout, or None plus one line saying why not."""
    if os.geteuid() != 0:
        # -n: fail at once rather than hang on a password prompt.
        cmd = ["sudo", "-n"] + cmd
    try:
        cp = run(cmd, dry_run=dry_run, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{shlex.join(cmd)}: {e!r}"
    if dry_run:
        return None, f"(dry) {shlex.join(cmd)}"
    if cp.returncode != 0:
        first = ((cp.stderr or cp.stdout or "").strip().splitlines() or ["no output"])[0]
        return None, f"{shlex.join(cmd)} rc={cp.returncode}: {first}"
    if "No journal files" in (cp.stderr or ""):
        return None, f"{shlex.join(cmd)}: no journal on this host"  # journalctl exits 0 here
    return cp.stdout, ""


def _sel_since(line: str, since: datetime) -> bool:
    """False for a SEL record dated before the run (a day's slack: the BMC clock's zone is unknown)."""
    m = SEL_DATE_RE.match(line)
    if not m:
        return True
    try:
        return datetime.strptime(m.group(1), "%m/%d/%Y").date() >= (since - timedelta(days=1)).date()
    except ValueError:
        return True


def capture_events(phase: Phase, dry_run: bool, logs_dir: Path, since: datetime) -> None:
    """Add ``bmc_sel_events`` and ``kernel_events``: SKIP when the read cannot run, WARN on lines of note, else PASS."""
    reads = (
        ("bmc_sel_events", ["ipmitool", "sel", "elist"], SEL_EVENT_RE, "sel.log"),
        (
            "kernel_events",
            ["journalctl", "-k", "--no-pager", "-o", "short-iso", "--since", f"@{int(since.timestamp())}"],
            KERNEL_EVENT_RE,
            "kernel_events.log",
        ),
    )
    for name, cmd, pattern, log_name in reads:
        t0 = time.time()
        out, why_not = _read(cmd, dry_run)
        if out is None:
            phase.add(Check(name=name, status=SKIP, details=why_not, ip="other"))
            continue
        log_path = logs_dir / log_name
        if not write_command_log(log_path, shlex.join(cmd), 0, time.time() - t0, out, ""):
            log_path = "(log not written)"
        lines = [
            ln.strip()
            for ln in out.splitlines()
            if pattern.search(ln) and (name != "bmc_sel_events" or _sel_since(ln, since))
        ]
        details = f"{len(lines)} line(s) of note; full output in {log_path}"
        if lines:
            details = f"{lines[-1][:160]} ({details})"
        phase.add(
            Check(
                name=name,
                status=WARN if lines else PASS,
                details=details,
                data={"lines": lines[-40:], "log": str(log_path)},
                ip="other",
                parts=name_chips(lines),
            )
        )
