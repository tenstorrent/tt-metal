# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 5: the kmd_triage first-step tools.

TRIAGE_TOOLS defines the steps a tier can list under ``triage`` in tiers.py.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

from .common import default_tt_metal_path, emit_result, emit_running, log, run_with_timeout
from .external import normalize_external_check
from .model import PASS, SKIP, WARN, Check, Phase

# First-step triage tools, run after the tier's post-test reset. They live in
# tools/scaleout/kmd_triage: host_side.sh reads host, PCIe and driver state out
# of sysfs, device_side.sh drives the kmd_triage binary over every chip.
#
# Their interface, which is all this phase depends on:
#   <script> --json <path> -o <path>
# writes a JSON file {"checks": [{"name", "status", "details", "ip", "data"}]}
# and a human-readable text report. Status is one of PASS/WARN/FAIL/SKIP; `ip`
# is one of the IP_ORDER groups (an unrecognised value lands in the dashboard's
# "other" category rather than breaking). The Phase and its rollup are built
# here, not there, so FAIL > WARN > PASS stays computed in one place.
#
# host_side needs root for the kernel-log scan; device_side does not, and
# opening the chips as root would be a needless escalation.
#
# Timeouts bound a wedged run. host_side is passive bash (32 lspci dumps plus a
# kernel-log scan); device_side opens each chip and self-bounds every probe at
# 60 s, so its budget has to cover 5 probes x 32 devices with slack.
TRIAGE_TOOLS = {
    "host_side": {"script": "host_side.sh", "root": True, "timeout_s": 600},
    "device_side": {"script": "device_side.sh", "root": False, "timeout_s": 2400},
}


# Triage checks whose WARN is recorded as PASS, by bare (unprefixed) name. Both
# count host state accumulated over a boot — correctable AER, kernel-log fault
# lines — so they WARN on units everything else calls healthy, and one phase
# WARN is a run WARN. FAIL continues to follow the normal triage gating policy.
TRIAGE_ADVISORY_WARN = frozenset({"hostside_pcie_aer", "hostside_kernel_log"})

# The scripts' home, relative to the repo root.
TRIAGE_SUBDIR = "tools/scaleout/kmd_triage"

# Override for the above, so a working copy of the scripts can be run against a
# deployed checkout without editing anything.
TRIAGE_DIR_ENV = "HC_TRIAGE_DIR"


def default_triage_dir() -> Path:
    """Where the triage scripts live: in this repo, unless overridden.

    They used to be cloned in at run time, so an unset env var meant the phase
    could only SKIP. Now that they ship here, the default is the checkout this
    file is part of and the phase runs by default.
    """
    raw = os.environ.get(TRIAGE_DIR_ENV, "").strip()
    return Path(raw) if raw else default_tt_metal_path() / TRIAGE_SUBDIR


def normalize_triage_check(payload: dict, gating: bool) -> Check:
    """One triage-script finding as a Check. See normalize_external_check.

    TRIAGE_ADVISORY_WARN checks record their WARN as PASS. Keyed on the status
    the tool wrote, so an unparseable one still lands on WARN; independent of
    ``--triage-gating``, which decides what a FAIL does and never gated a WARN.
    """
    name = str(payload.get("name") or "")
    advisory_warn = (
        str(payload.get("status") or "").upper() == WARN and name.removeprefix("triage_") in TRIAGE_ADVISORY_WARN
    )
    return normalize_external_check(
        payload,
        prefix="triage_",
        hold_fail_at=None if gating else WARN,
        hold_warn_at=PASS if advisory_warn else None,
        gating_flag="--triage-gating",
    )


def run_triage(
    triage_dir: Path | None,
    tools: Sequence[str],
    phase: Phase,
    dry_run: bool,
    logs_dir: Path,
    gating: bool,
    tt_metal: Path | None = None,
) -> None:
    """Run the named TRIAGE_TOOLS and fold their findings in.

    Every way of not having them — directory or script missing — is a SKIP
    carrying its reason, never a failure. A SKIP with
    a reason is the point: a check that silently vanishes reads as coverage we
    had, and this suite has been bitten by that before (see the zero-match gtest
    filter note in HEALTH_CHECK.md).

    Tooling breakage (timeout, no JSON, the tool reporting its own failure) is a
    WARN, not a FAIL: it says nothing about the hardware, only that we didn't get
    a reading. Findings about the hardware come from the tools' JSON.
    """
    if triage_dir is None:
        phase.add(Check(name="triage", status=SKIP, details="no triage directory given", ip="other"))
        return
    if not triage_dir.is_dir():
        phase.add(Check(name="triage", status=SKIP, details=f"triage scripts absent: {triage_dir}", ip="other"))
        return

    # device_side.sh locates the kmd_triage binary under $TT_METAL_HOME. Pass
    # the path the run was told to use rather than trusting the ambient value,
    # which is what --tt-metal-path is for.
    env = os.environ.copy()
    if tt_metal is not None:
        env["TT_METAL_HOME"] = str(tt_metal)

    logs_dir.mkdir(parents=True, exist_ok=True)

    for tool in tools:
        spec = TRIAGE_TOOLS[tool]
        run_name = f"triage_{tool}_run"
        script = triage_dir / spec["script"]
        if not script.is_file():
            phase.add(Check(name=run_name, status=SKIP, details=f"not present: {script}", ip="other"))
            continue

        json_path = logs_dir / f"triage_{tool}.json"
        text_path = logs_dir / f"triage_{tool}.txt"
        # A stale file from an earlier run would be read as this run's findings.
        json_path.unlink(missing_ok=True)

        cmd = ["bash", str(script), "--json", str(json_path), "-o", str(text_path)]
        if spec["root"] and os.geteuid() != 0:
            # -n: fail immediately rather than hang on a password prompt.
            cmd = ["sudo", "-n"] + cmd

        log(f"--- triage '{tool}' ---")
        log(f"  cmd:  {' '.join(cmd)}")
        log(f"  log:  {text_path}")

        if dry_run:
            print(f"  {run_name:30} (dry-run)")
            phase.add(Check(name=run_name, status=SKIP, details=f"(dry) {' '.join(cmd)}", ip="other"))
            continue

        emit_running(run_name)
        t0 = time.time()
        try:
            rc, out, err, timed_out = run_with_timeout(cmd, spec["timeout_s"], cwd=str(triage_dir), env=env)
        except OSError as e:
            dt = time.time() - t0
            emit_result(run_name, SKIP, suffix=f"({dt:.1f}s)")
            phase.add(Check(name=run_name, status=SKIP, details=f"could not run {script}: {e!r}", ip="other"))
            continue
        dt = time.time() - t0

        checks: list[Check] = []
        parse_error = ""
        if not json_path.is_file():
            parse_error = f"tool wrote no {json_path.name}"
        else:
            try:
                payload = json.loads(json_path.read_text())
                raw = payload.get("checks") if isinstance(payload, dict) else None
                if isinstance(raw, list):
                    checks = [normalize_triage_check(c, gating) for c in raw if isinstance(c, dict)]
                    if not checks:
                        parse_error = "JSON 'checks' list contains no usable checks"
                else:
                    parse_error = "JSON has no 'checks' list"
            except (OSError, ValueError) as e:
                parse_error = repr(e)

        if timed_out:
            status = WARN
            details = f"timed out after {spec['timeout_s']}s; no findings collected"
        elif parse_error:
            status = WARN
            # Why it produced nothing is the only useful thing left to report,
            # and it is almost always the tool's first line of stderr (a missing
            # binary, a permission problem). Without it the console shows only
            # "no usable findings" and the reason sits unread in the JSON.
            reason = next((ln.strip() for ln in err.splitlines() if ln.strip()), "")
            details = f"rc={rc} but no usable findings ({parse_error})"
            if reason:
                details += f": {reason[:200]}"
        elif rc == 3:
            # The tools reserve 3 for "the script itself failed", as distinct
            # from their 1/2 which are findings about the hardware.
            status = WARN
            details = f"tool reported its own failure (rc=3); {len(checks)} check(s) collected"
        else:
            status = PASS
            details = f"rc={rc} {len(checks)} check(s)"
        details += f" dur={dt:.1f}s log={text_path}"

        emit_result(run_name, status, suffix=f"({dt:.1f}s)")
        phase.add(
            Check(
                name=run_name,
                status=status,
                details=details,
                data={
                    "command": " ".join(cmd),
                    "rc": rc,
                    "duration_s": dt,
                    "timed_out": timed_out,
                    "checks_collected": len(checks),
                    "report_file": str(text_path),
                    "stdout_tail": out[-2000:],
                    "stderr_tail": err[-2000:],
                },
                ip="other",
            )
        )
        for c in checks:
            phase.add(c)
