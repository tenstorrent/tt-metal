# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 2: the ethchk Ethernet link check (ethchk-upstream-tests.sh)."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import time
from pathlib import Path

from .common import SUITE_DIR, emit_result, emit_running, log, run_with_timeout, write_command_log
from .model import FAIL, PASS, SKIP, WARN, Check, Phase

# A sibling of the suite rather than part of it: it is as useful run by hand on
# a host as it is here. It reads every eth link twice, 30 s apart, with tt-metal
# triage's check_eth_status and prints one report ending in a verdict. Triage
# runs in the public upstream-tests image, started with `docker run`, so the
# check needs Docker on the host. Inside a container (run_diag_docker.sh, the
# fleet image) there is none, and the phase reports SKIP.
ETH_CHECK_SCRIPT = SUITE_DIR.parent / "ethchk-upstream-tests.sh"

# The script's exit code is its verdict: 0 healthy, 2 problem, 3 warning. 4 is
# a chip that dropped off PCIe, which the script reports without running triage
# so the chips are left as they were found; that is a fault in its own right.
ETH_CHECK_VERDICTS = {0: PASS, 2: FAIL, 3: WARN, 4: FAIL}
# Docker, or its image, is not usable: no reading was taken, as with a missing package.
ETH_CHECK_NO_DOCKER_RC = 5

# A backstop, not the expected duration: the two reads and the 30 s between
# them take a minute or two, but the first run on a host also downloads the
# image once, about 9 GB.
ETH_CHECK_TIMEOUT_S = 1200

# What the report prints that this phase records. The SUMMARY line is the one
# meant for a machine; the rest is for whoever reads the check.
VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(.+?)\s*$")
LINKS_RE = re.compile(r"^\s*LINKS:\s*(.+?)\s*$")
IMAGE_RE = re.compile(r"^\s*image\s+(\S+)\s*$")
VERSION_RE = re.compile(r"^\s*ethchk\s+(version\s.+?)\s*$")
RAW_DIR_RE = re.compile(r"^\s*raw triage output:\s*(\S+)")
SUMMARY_RE = re.compile(r"^SUMMARY\s+(.*)$")
SUMMARY_FIELD_RE = re.compile(r"(\w+)=(\S*)")

# The findings are indented under the verdict; LINKS continues on lines indented to its text.
FINDING_INDENT = " " * 4
LINKS_INDENT = " " * 11


def parse_eth_check_report(text: str) -> dict:
    """The parts of an ethchk report this phase records. Anything not found stays empty.

    ``findings`` are the lines printed under the verdict, in order and as
    printed: the faults (at most 8, then "... and N more"), "also worth a look:"
    and the warnings after it, and any closing note. ``summary`` is the closing
    SUMMARY line's key=value fields.
    """
    report: dict = {
        "verdict": "",
        "findings": [],
        "links": "",
        "image": "",
        "version": "",
        "raw_dir": "",
        "summary": {},
    }
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if m := VERDICT_RE.match(line):
            report["verdict"] = m.group(1)
            while i < len(lines) and lines[i].startswith(FINDING_INDENT) and lines[i].strip():
                report["findings"].append(lines[i].strip())
                i += 1
        elif m := LINKS_RE.match(line):
            parts = [m.group(1)]
            while i < len(lines) and lines[i].startswith(LINKS_INDENT) and lines[i].strip():
                parts.append(lines[i].strip())
                i += 1
            report["links"] = " ".join(parts)
        elif m := IMAGE_RE.match(line):
            report["image"] = m.group(1)
        elif m := VERSION_RE.match(line):
            report["version"] = m.group(1)
        elif m := RAW_DIR_RE.match(line):
            report["raw_dir"] = m.group(1)
        elif m := SUMMARY_RE.match(line):
            report["summary"] = dict(SUMMARY_FIELD_RE.findall(m.group(1)))
    return report


def no_verdict_reason(out: str, err: str) -> str:
    """The script's own account of why it gave no verdict: its ERROR line, the
    last line it quotes under it (the triage log's, when triage failed), and the
    hint it closes on (run it as root, log out of ghcr.io, triage failed)."""
    lines = [ln.strip() for ln in f"{out}\n{err}".splitlines() if ln.strip()]
    first = next((i for i, ln in enumerate(lines) if "ERROR:" in ln), None)
    picked = lines[-1:] if first is None else [lines[first], *lines[first + 1 :][-2:]]
    return "; ".join(ln[:200] for ln in picked)


def run_eth_check(phase: Phase, dry_run: bool, logs_dir: Path, script_override: str | None = None) -> None:
    """Run ethchk-upstream-tests.sh and record its verdict.

    * ``ethchk_run`` **SKIP** — no reading was taken: the script is absent, or
      Docker is not on PATH, not usable, or could not get the image. Always with
      the reason attached.
    * ``ethchk_run`` **WARN** — it ran but gave no verdict: it timed out, or
      triage itself failed. Lost coverage, not a statement about the hardware.
    * ``ethchk_verdict`` — the script's own verdict: HEALTHY is PASS, WARNING is
      WARN, PROBLEM is FAIL, and so is a chip that dropped off PCIe. The lines it
      printed under the verdict are the details, and its SUMMARY fields the data.

    The script's settings (IMG, SAMPLE_SECS, ETHCHK_FORCE, the error-rate
    thresholds) reach it through the environment, as they do by hand.
    """
    script = Path(script_override) if script_override else ETH_CHECK_SCRIPT
    if not script.is_file():
        reason = f"--eth-check-script={script_override} is not a file" if script_override else f"not present: {script}"
        phase.add(Check(name="ethchk_run", status=SKIP, details=reason, ip="other"))
        return
    if shutil.which("docker") is None:
        phase.add(
            Check(
                name="ethchk_run",
                status=SKIP,
                details=(
                    "docker not on PATH — ethchk runs tt-metal triage in its own image, so it runs where the "
                    "suite runs on the host itself (run_diag.sh), not inside a container"
                ),
                ip="other",
            )
        )
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    # The script compares each run against the earlier ones it finds beside it,
    # so a caller who keeps them elsewhere keeps that history. By default they
    # land in logs/, where collect_run_artifacts() picks them up for the ticket.
    runs_dir = os.environ.get("ETHCHK_DIR") or str(logs_dir)
    log_path = logs_dir / "ethchk.log"
    cmd = ["bash", str(script)]
    cmdline = f"ETHCHK_DIR={shlex.quote(runs_dir)} {shlex.join(cmd)}"

    log("--- eth check ---")
    log(f"  cmd:  {cmdline}")
    log(f"  log:  {log_path}")

    if dry_run:
        print(f"  {'ethchk_run':30} (dry-run)")
        phase.add(Check(name="ethchk_run", status=SKIP, details=f"(dry) {cmdline}", ip="other"))
        return

    emit_running("ethchk_run")
    t0 = time.time()
    try:
        rc, out, err, timed_out = run_with_timeout(
            cmd, ETH_CHECK_TIMEOUT_S, env={**os.environ, "ETHCHK_DIR": runs_dir}
        )
    except OSError as e:
        dt = time.time() - t0
        emit_result("ethchk_run", SKIP, suffix=f"({dt:.1f}s)")
        phase.add(Check(name="ethchk_run", status=SKIP, details=f"could not run {script}: {e!r}", ip="other"))
        return
    dt = time.time() - t0

    write_command_log(log_path, cmdline, rc, dt, out, err)
    report = parse_eth_check_report(out)
    verdict = ETH_CHECK_VERDICTS.get(rc) if report["verdict"] and not timed_out else None

    if timed_out:
        status = WARN
        details = f"timed out after {ETH_CHECK_TIMEOUT_S}s; no verdict"
    elif verdict is None:
        status = SKIP if rc == ETH_CHECK_NO_DOCKER_RC else WARN
        details = f"rc={rc} " + (
            "Docker or its image is not usable" if rc == ETH_CHECK_NO_DOCKER_RC else "but no verdict"
        )
        reason = no_verdict_reason(out, err)
        if reason:
            details += f": {reason}"
    else:
        status = PASS
        details = f"rc={rc}"
        if report["raw_dir"]:
            details += f" raw={report['raw_dir']}"
    details += f" dur={dt:.1f}s log={log_path}"

    emit_result("ethchk_run", status, suffix=f"({dt:.1f}s)")
    phase.add(
        Check(
            name="ethchk_run",
            status=status,
            details=details,
            data={
                "command": cmdline,
                "rc": rc,
                "duration_s": dt,
                "timed_out": timed_out,
                "image": report["image"],
                "ethchk_version": report["version"],
                "run_dir": report["raw_dir"],
                "log_file": str(log_path),
                "stdout_tail": out[-2000:],
                "stderr_tail": err[-2000:],
            },
            ip="other",
        )
    )
    if verdict is None:
        return
    lines = [report["verdict"], *report["findings"]]
    if report["links"]:
        lines.append(f"LINKS: {report['links']}")
    phase.add(
        Check(
            name="ethchk_verdict",
            status=verdict,
            details="\n".join(lines),
            data={
                "verdict": report["verdict"],
                "findings": report["findings"],
                "links": report["links"],
                "summary": report["summary"],
            },
            ip="eth",
        )
    )
