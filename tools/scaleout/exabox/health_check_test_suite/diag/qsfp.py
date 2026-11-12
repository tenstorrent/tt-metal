# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 6: the QSFP tests (ETH link, cabling and module state).

Collects a ``tt-bh-glx-cluster-debug`` dump; qsfp_ingest.py turns it into checks.
"""
from __future__ import annotations

import re
import shlex
import socket
import time
from pathlib import Path

from .common import emit_result, emit_running, log, resolve_optional_tool, run_with_timeout, write_command_log
from .external import normalize_external_check
from .model import FAIL, PASS, SKIP, Check, Phase

# `tt-bh-glx-cluster-debug` ships as a .deb built with PyInstaller, so it is one
# binary on PATH with no repo, no venv and no import path to arrange — which is
# why this phase probes for a command rather than for a checkout the way the
# triage phase probes for its scripts. A host that has not had the package
# installed simply does not have it, and that is a SKIP: an optional tool is
# lost coverage when it is absent, never a failure.
QSFP_TOOL_BIN = "tt-bh-glx-cluster-debug"

# `collect` reads one galaxy into a JSONL snapshot: per-ASIC telemetry, the
# per-port ERISC blobs for all 448 ETH ports, and the QSFP cages over I2C.
# --parallelize reads the four UBBs' cages at once, about twice as fast, and the
# cage sweep is where most of a run's time goes.
QSFP_COLLECT_ARGS = ["collect", "--parallelize"]

# A backstop, not the expected duration: the cage sweep self-bounds at the
# tool's own 600 s --qsfp-budget and a full run measures ~7 min. It has to stay
# well inside run_health_check.py's whole-run --timeout-minutes (30 by default),
# because that one kills the process group and takes the report with it, while
# overrunning this one only costs this phase.
QSFP_TIMEOUT_S = 1200


def import_qsfp_ingest():
    """The ingest module beside this one, or None if it didn't ship with this copy.

    The conversion from the dump's records to checks is kept in its own module
    because it is pure data shaping with no hardware in it: a stored dump is all
    it needs, so it can be iterated on and tested at a desk.

    Imported here rather than at module scope because a missing ingest module is
    lost coverage for one phase, not a broken suite — an unguarded
    top-level import would turn a partial deployment into a run that produces no
    report at all.
    """
    try:
        from . import qsfp_ingest
    except ImportError:
        return None
    return qsfp_ingest


def resolve_qsfp_tool(override: str | None) -> tuple[str | None, str]:
    """The QSFP collector binary, or None plus the reason this phase can't run.

    Unlike resolve_tt_smi this never raises; see resolve_optional_tool.
    """
    return resolve_optional_tool(
        override,
        QSFP_TOOL_BIN,
        "--qsfp-tool-path",
        f"{QSFP_TOOL_BIN} not on PATH — it ships in the syseng cluster-debug .deb, "
        f"so a host without that package collects no ETH dump",
    )


# tt-smi's board_id prefix and the collector's board_id bit field are two reads
# of one 64-bit register down two independent paths, so they agree or one of
# them is wrong. Worth stating because the revision selects the collector's
# internal topology table, and the wrong table resolves 24 confident wrong
# inter-UBB partners per galaxy — which would surface above as miscabling on a
# correctly cabled machine.
QSFP_REV_EQUIVALENT = {
    "RevA/B": "BH_GALAXY_REV_AB",
    "RevC": "BH_GALAXY_REV_C",
}

# The ingest check this cross-check reads its half of the comparison from, after
# normalize_external_check has applied the phase prefix.
QSFP_REV_CHECK = "qsfp_board_rev"


def normalize_qsfp_check(payload: dict, gating: bool) -> Check:
    """One QSFP-test finding as a Check. See normalize_external_check.

    Until ``--qsfp-gating`` says otherwise the phase reports only PASS and SKIP:
    the tool is still being validated against the fleet, and a finding it is not
    yet trusted to have got right should not be the thing an operator's eye is
    drawn to. Everything it found is still on the record — the details line is
    the tool's own, the ``data`` is untouched, and the annotation says what was
    held — so the fleet data needed to decide whether to turn gating on is
    collected either way. Both holds lift together; a held FAIL is one step,
    straight to PASS, so nothing lands on the WARN the phase does not raise.
    """
    return normalize_external_check(
        payload,
        prefix="qsfp_",
        hold_fail_at=None if gating else PASS,
        hold_warn_at=None if gating else PASS,
        gating_flag="--qsfp-gating",
    )


def qsfp_rev_crosscheck(checks: list[Check], detected_rev: str | None, gating: bool) -> Check:
    """Compare the dump's board revision against the snapshot phase's.

    The one finding in this phase that this file derives rather than reads out
    of the dump, since only the runner knows both halves. It still goes out
    through the same normalizer as everything else here, so one check cannot
    end up carrying a status the rest of the phase would not have been given.
    """
    found = next((c for c in checks if c.name == QSFP_REV_CHECK), None)
    dump_rev = (found.data or {}).get("rev") if found else None
    if detected_rev is None or dump_rev in (None, "unknown"):
        missing = "the tt-smi snapshot" if detected_rev is None else "the dump"
        payload = {
            "name": "board_rev_agrees",
            "status": SKIP,
            "details": f"no revision to compare: {missing} did not determine one",
            "data": {"tt_smi_rev": detected_rev, "dump_rev": dump_rev},
            "ip": "board",
        }
    else:
        expected = QSFP_REV_EQUIVALENT.get(detected_rev)
        agrees = expected == dump_rev
        payload = {
            "name": "board_rev_agrees",
            "status": PASS if agrees else FAIL,
            "details": (
                f"tt-smi and the dump agree: {detected_rev} == {dump_rev}"
                if agrees
                else f"tt-smi read {detected_rev} (expects {expected}) but the dump read {dump_rev}; "
                f"the dump's internal topology table, and so every partner finding above, is "
                f"resolved against its own value"
            ),
            "data": {"tt_smi_rev": detected_rev, "dump_rev": dump_rev, "expected_dump_rev": expected},
            "ip": "board",
        }
    return normalize_qsfp_check(payload, gating)


def run_qsfp_tests(
    tier: str,
    phase: Phase,
    dry_run: bool,
    logs_dir: Path,
    gating: bool = False,
    binary_override: str | None = None,
    descriptor: Path | None = None,
    reason: str | None = None,
    detected_rev: str | None = None,
) -> None:
    """Run the QSFP tests: collect an ETH dump and fold its findings in.

    The dump answers a question nothing else in this suite reaches: the
    snapshot phase reads ``ETH_LIVE_STATUS`` out of tt-smi telemetry, a bitmask
    per chip, and the eth gtests push traffic over links that already came up —
    neither can say *which far end* a port is actually talking to. The dump
    carries the expected partner and the firmware's own ``remote_info`` on every
    one of the 448 port records, so a miscabled or half-trained link is a
    comparison rather than an inference. Its cage and module records are also
    the only inventory of what is physically plugged in.

    By default the phase reports only **SKIP** and **PASS**, because the tool is
    still being validated against the fleet:

    * **SKIP** — no reading was taken. The package isn't installed, the module that reads the dump is missing, the collect
      timed out, or the dump would not parse. Always with the reason attached,
      on the same reasoning as the triage phase: a check that silently vanishes
      reads as coverage we had.
    * **PASS** — a reading was taken. Whatever the dump found is in the check's
      details and data, annotated with what was held, but it is not yet allowed
      to draw the eye of someone triaging a rack.

    ``--qsfp-gating`` is the one switch that changes this, and it changes it
    completely: findings then report at their real severity and the phase gates
    the run like any other. Turning it on is the decision to be made once there
    is enough fleet data to say the findings are right, which is why the phase
    collects that data from the first run either way.

    ``tier`` only labels the dump's ``--reason``; whether a tier runs this phase
    at all is tiers.py's decision, taken before this is called.
    """
    binary, unavailable = resolve_qsfp_tool(binary_override)
    if binary is None:
        phase.add(Check(name="qsfp_collect", status=SKIP, details=unavailable, ip="other"))
        return

    ingest = import_qsfp_ingest()
    if ingest is None:
        # Collecting a dump nothing will read would spend minutes of hardware
        # access to produce a file and no findings.
        phase.add(
            Check(
                name="qsfp_collect",
                status=SKIP,
                details=f"qsfp_ingest.py is not importable beside {Path(__file__).name}",
                ip="other",
            )
        )
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    # Named for the host rather than the tool's default host+UTC stamp so the
    # artifact collector and a ticket reader see a predictable filename. The
    # collector insists on the .jsonl suffix, which is also what keeps a
    # truncated dump readable up to its last newline. The hostname is reduced to
    # filename-safe characters because it reaches this path from the system, not
    # from the caller, and a separator in it would write outside logs_dir.
    safe_host = re.sub(r"[^A-Za-z0-9._-]", "_", socket.gethostname()) or "unknown"
    dump_path = logs_dir / f"qsfp_dump_{safe_host}.jsonl"
    log_path = logs_dir / "qsfp.log"
    text_path = logs_dir / "qsfp.txt"
    # A dump left by an earlier run would be read as this run's ETH state.
    dump_path.unlink(missing_ok=True)

    cmd = [binary, *QSFP_COLLECT_ARGS, "--out", str(dump_path)]
    if descriptor is not None:
        cmd += ["--factory-descriptor-path", str(descriptor)]
    # Free text on the envelope. Recording why a dump exists is what the field
    # is for, and a dump found later with no reason on it is a dump nobody can
    # place.
    cmd += ["--reason", reason or f"tt-metal health check, {tier} tier"]

    log("--- qsfp tests: collect ---")
    log(f"  cmd:  {shlex.join(cmd)}")
    log(f"  dump: {dump_path}")

    if dry_run:
        print(f"  {'qsfp_collect':30} (dry-run)")
        phase.add(Check(name="qsfp_collect", status=SKIP, details=f"(dry) {shlex.join(cmd)}", ip="other"))
        return

    emit_running("qsfp_collect")
    t0 = time.time()
    try:
        rc, out, err, timed_out = run_with_timeout(cmd, QSFP_TIMEOUT_S)
    except OSError as e:
        dt = time.time() - t0
        emit_result("qsfp_collect", SKIP, suffix=f"({dt:.1f}s)")
        phase.add(Check(name="qsfp_collect", status=SKIP, details=f"could not run {binary}: {e!r}", ip="other"))
        return
    dt = time.time() - t0

    write_command_log(log_path, shlex.join(cmd), rc, dt, out, err)

    # The dump, not the exit code, is what this phase is here for. `collect`
    # exits non-zero only when nothing at all could be collected — a descriptor
    # it couldn't open, a BMC that wouldn't answer and a cage sweep that fell
    # over are all recorded in the dump as findings and still exit 0 — so a
    # readable dump after a non-zero exit is still worth reading, and a clean
    # exit with no dump is still a failure to collect.
    checks: list[Check] = []
    problem = ""
    records: list = []
    malformed = 0
    if timed_out:
        log("  timed out; leaving the partial dump unread")
    elif not dump_path.is_file():
        problem = f"wrote no {dump_path.name}"
    else:
        try:
            records, malformed = ingest.load_dump(dump_path)
            payload = ingest.build_report(records, malformed)
            checks = [normalize_qsfp_check(c, gating) for c in payload.get("checks", []) if isinstance(c, dict)]
            if not checks:
                problem = "the dump yielded no usable checks"
        except (OSError, ValueError) as e:
            problem = repr(e)
        else:
            # The readable copy is a convenience for whoever opens the ticket;
            # losing it must not cost the findings themselves, which are already
            # in hand and on their way to the report.
            try:
                text_path.write_text(ingest.render(payload, source=str(dump_path)))
            except OSError as e:
                log(f"  could not write {text_path}: {e!r}")

    if timed_out:
        status = SKIP
        details = f"timed out after {QSFP_TIMEOUT_S}s; " + (
            "the partial dump is kept as an artifact but not read" if dump_path.is_file() else "no dump was written"
        )
    elif problem:
        status = SKIP
        # Why it produced nothing is the only useful thing left to report, and
        # it is almost always the tool's first line of stderr.
        first = next((ln.strip() for ln in err.splitlines() if ln.strip()), "")
        details = f"rc={rc} but no usable findings ({problem})"
        if first:
            details += f": {first[:200]}"
    else:
        status = PASS
        details = f"rc={rc} {len(records)} record(s), {len(checks)} check(s)"
    details += f" dur={dt:.1f}s log={log_path}"

    emit_result("qsfp_collect", status, suffix=f"({dt:.1f}s)")
    phase.add(
        Check(
            name="qsfp_collect",
            status=status,
            details=details,
            data={
                "command": shlex.join(cmd),
                "rc": rc,
                "duration_s": dt,
                "timed_out": timed_out,
                "records": len(records),
                "malformed_lines": malformed,
                "checks_collected": len(checks),
                "dump_file": str(dump_path),
                "dump_bytes": dump_path.stat().st_size if dump_path.is_file() else 0,
                "report_file": str(text_path),
                "stdout_tail": out[-2000:],
                "stderr_tail": err[-2000:],
            },
            ip="other",
        )
    )
    for c in checks:
        phase.add(c)
    if checks:
        phase.add(qsfp_rev_crosscheck(checks, detected_rev, gating))
