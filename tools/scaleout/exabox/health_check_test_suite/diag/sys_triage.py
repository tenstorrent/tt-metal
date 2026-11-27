# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 7: the syseng sys-triage bundle (system and per-tray triage dumps)."""
from __future__ import annotations

import os
import shlex
import shutil
import time
from pathlib import Path

from .common import emit_result, emit_running, log, resolve_optional_tool, run_with_timeout, write_command_log
from .model import PASS, SKIP, WARN, Check, Phase

# Ships in the same syseng .deb as the QSFP collector, and is probed for the
# same way. It writes a directory of text files for a person to read — the
# Quanta FW stack, FRU, CPLD DIAG, lspci, kernel logs, per-chip vuart — rather
# than findings for this report to judge, so the phase records whether the
# bundle was collected and what it could not collect, and never votes.
SYS_TRIAGE_BIN = "sys-triage"

# sys-triage refuses to start without all three, and reads each one from the
# environment when its flag is absent. They reach it through the environment
# only: the command line is logged and stored in the report, and argv is
# readable by every user on the host.
SYS_TRIAGE_BMC_ENV = {
    "ip": "BMC_IP",
    "user": "BMC_USER",
    "password": "BMC_PASSWORD",
}

# A backstop, not the expected duration. CPLD DIAG self-bounds at 180 s, the
# 32 tt-fw-terminal drains run 8 at a time, and the rest is ipmitool and
# Redfish reads of a few seconds each.
SYS_TRIAGE_TIMEOUT_S = 900

# The bundle's own index file. Written last from what the collectors left
# behind, so its presence is what "the bundle was collected" means here.
SYS_TRIAGE_INDEX = "triage.txt"


def resolve_sys_triage(override: str | None) -> tuple[str | None, str]:
    """The sys-triage binary, or None plus the reason this phase can't run.

    Same contract as resolve_qsfp_tool; see resolve_optional_tool.
    """
    return resolve_optional_tool(
        override,
        SYS_TRIAGE_BIN,
        "--sys-triage-path",
        f"{SYS_TRIAGE_BIN} not on PATH — it ships in the syseng .deb, "
        f"so a host without that package collects no triage bundle",
    )


def resolve_bmc_env(ip: str | None, user: str | None, password: str | None) -> tuple[dict[str, str], list[str]]:
    """The BMC credentials as sys-triage's environment, and what is missing.

    Each value comes from its flag, else from the variable sys-triage itself
    would read, so a fleet launcher that already exports them needs no flags.
    """
    given = {"ip": ip, "user": user, "password": password}
    env: dict[str, str] = {}
    missing: list[str] = []
    for key, var in SYS_TRIAGE_BMC_ENV.items():
        value = (given[key] or os.environ.get(var) or "").strip()
        if value:
            env[var] = value
        else:
            missing.append(f"--bmc-{key}/${var}")
    return env, missing


def run_sys_triage(
    phase: Phase,
    dry_run: bool,
    logs_dir: Path,
    *,
    binary_override: str | None = None,
    bmc_ip: str | None = None,
    bmc_user: str | None = None,
    bmc_password: str | None = None,
) -> None:
    """Collect the sys-triage bundle into ``<logs_dir>/sys_triage/``.

    * **SKIP** — no bundle. The package isn't installed, the BMC credentials
      are incomplete, the run timed out, or it exited without writing its
      index. Always with the reason attached.
    * **WARN** — a bundle was written, but sys-triage reported sections it
      could not collect; its ``warning:`` lines are in the details and data.
    * **PASS** — a bundle was written with nothing reported missing.

    The phase never gates: a missing section is lost forensics, not a
    hardware finding, and the bundle's contents are for a person to read.
    """
    binary, unavailable = resolve_sys_triage(binary_override)
    if binary is None:
        phase.add(Check(name="sys_triage", status=SKIP, details=unavailable, ip="other"))
        return

    bmc_env, missing = resolve_bmc_env(bmc_ip, bmc_user, bmc_password)
    if missing:
        phase.add(
            Check(
                name="sys_triage",
                status=SKIP,
                details=(
                    f"BMC credentials incomplete, which sys-triage refuses to run without; "
                    f"missing: {', '.join(missing)}"
                ),
                ip="other",
            )
        )
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir = logs_dir / "sys_triage"
    log_path = logs_dir / "sys_triage.log"
    cmd = [binary, str(out_dir)]

    log("--- sys-triage ---")
    log(f"  cmd: {shlex.join(cmd)}  (BMC {bmc_env['BMC_USER']}@{bmc_env['BMC_IP']} via environment)")
    log(f"  out: {out_dir}")

    if dry_run:
        print(f"  {'sys_triage':30} (dry-run)")
        phase.add(Check(name="sys_triage", status=SKIP, details=f"(dry) {shlex.join(cmd)}", ip="other"))
        return

    # A bundle left by an earlier run would be read as this run's state.
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)

    emit_running("sys_triage")
    t0 = time.time()
    try:
        rc, out, err, timed_out = run_with_timeout(cmd, SYS_TRIAGE_TIMEOUT_S, env={**os.environ, **bmc_env})
    except OSError as e:
        dt = time.time() - t0
        emit_result("sys_triage", SKIP, suffix=f"({dt:.1f}s)")
        phase.add(Check(name="sys_triage", status=SKIP, details=f"could not run {binary}: {e!r}", ip="other"))
        return
    dt = time.time() - t0
    # sys-triage hands the password on to cpld-dump.sh as an argument, so an
    # error from that script could carry it. Nothing captured here may.
    secret = bmc_env["BMC_PASSWORD"]
    out, err = out.replace(secret, "***"), err.replace(secret, "***")

    write_command_log(log_path, shlex.join(cmd), rc, dt, out, err)

    # sys-triage carries on past a section it cannot collect and says so on
    # stderr, so its warnings are the only record of what the bundle lacks.
    warnings = [ln.strip()[len("warning:") :].strip() for ln in err.splitlines() if ln.strip().startswith("warning:")]
    files = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()) if out_dir.is_dir() else []
    collected = (out_dir / SYS_TRIAGE_INDEX).is_file()

    if timed_out:
        status = SKIP
        details = f"timed out after {SYS_TRIAGE_TIMEOUT_S}s; {len(files)} partial file(s) kept"
    elif not collected:
        status = SKIP
        first = next((ln.strip() for ln in err.splitlines() if ln.strip()), "")
        details = f"rc={rc} but no {SYS_TRIAGE_INDEX} written"
        if first:
            details += f": {first[:200]}"
    else:
        status = WARN if warnings else PASS
        details = f"rc={rc} {len(files)} file(s)"
        if warnings:
            details += f"; {len(warnings)} section(s) incomplete: {'; '.join(w[:120] for w in warnings[:5])}"
    details += f" dur={dt:.1f}s dir={out_dir} log={log_path}"

    emit_result("sys_triage", status, suffix=f"({dt:.1f}s)")
    phase.add(
        Check(
            name="sys_triage",
            status=status,
            details=details,
            data={
                "command": shlex.join(cmd),
                "bmc_ip": bmc_env["BMC_IP"],
                "bmc_user": bmc_env["BMC_USER"],
                "rc": rc,
                "duration_s": dt,
                "timed_out": timed_out,
                "out_dir": str(out_dir),
                "files": files,
                "warnings": warnings,
                "stdout_tail": out[-2000:],
                "stderr_tail": err[-2000:],
            },
            ip="other",
        )
    )
