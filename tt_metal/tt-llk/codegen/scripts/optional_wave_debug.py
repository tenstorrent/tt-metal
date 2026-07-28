#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Best-effort bridge from the CodeGen tester to the private waveform debugger."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FSDB_ENV = "LLK_DEBUG_FSDB"
INSECURE_HOST_KEY_ENV = "LLK_DEBUG_INSECURE_HOST_KEY"
_FSDB_PATH = re.compile(r"(/[^\s\"'`]+?\.fsdb)(?=$|[\s\"'`,;:)])")


@dataclass(frozen=True)
class WaveDebugOutcome:
    status: str
    summary: str
    evidence_path: Path | None = None


def _single_line(value: object) -> str:
    return " ".join(str(value).split())


def _append_tester_log(
    agent_log: Path,
    *,
    attempt: int,
    outcome: WaveDebugOutcome,
    fsdb: str | None,
    evidence: dict | None = None,
) -> None:
    agent_log.parent.mkdir(parents=True, exist_ok=True)
    if not agent_log.exists() or agent_log.stat().st_size == 0:
        agent_log.write_text("# Agent: llk-tester\n")

    lines = [
        "",
        f"## Optional waveform debugging — attempt {attempt}",
        f"- Status: `{outcome.status}`",
        f"- Summary: {_single_line(outcome.summary)}",
        f"- FSDB: `{fsdb}`" if fsdb else "- FSDB: not available",
    ]
    if outcome.evidence_path:
        lines.append(f"- Evidence: `{outcome.evidence_path}`")
    if evidence:
        source = evidence.get("tool_source") or {}
        revision = source.get("revision")
        if revision:
            dirty = "dirty" if source.get("dirty") else "clean"
            lines.append(
                f"- Private tool: `{source.get('repository', 'llk_code_gen')}@{revision}` ({dirty})"
            )
        findings = evidence.get("findings") or []
        if findings:
            lines.append("- Deterministic findings:")
            for finding in findings:
                lines.append(
                    "  - "
                    f"`{_single_line(finding.get('severity', 'unknown'))}` "
                    f"`{_single_line(finding.get('classification', 'unclassified'))}`: "
                    f"{_single_line(finding.get('summary', ''))}"
                )
    lines.extend(
        [
            "- Pipeline action: continue the normal tester/refiner flow.",
            "",
        ]
    )
    with agent_log.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def _fsdb_from_run_log(run_log: Path) -> str | None:
    if not run_log.is_file():
        return None
    try:
        text = run_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = _FSDB_PATH.findall(text)
    for candidate in reversed(matches):
        if Path(candidate).is_file():
            return candidate
    return None


def _resolve_fsdb(
    requested: str | None, *, run_log: Path, environment: dict[str, str]
) -> tuple[str | None, str | None]:
    candidate = requested or environment.get(FSDB_ENV) or _fsdb_from_run_log(run_log)
    if not candidate:
        return None, (
            f"no FSDB was supplied, {FSDB_ENV} is unset, and no existing FSDB "
            f"path was found in {run_log}"
        )
    path = Path(candidate)
    if not path.is_file():
        return None, f"FSDB does not exist or is not readable: {candidate}"
    return str(path), None


def run_optional_wave_debug(
    *,
    log_dir: Path,
    cycle: int,
    attempt: int,
    failure_kind: str,
    fsdb: str | None,
    launcher: Path,
    timeout_seconds: int,
    environment: dict[str, str] | None = None,
) -> WaveDebugOutcome:
    """Run private diagnosis if possible and always leave a tester-log breadcrumb."""

    env = dict(os.environ if environment is None else environment)
    agent_log = log_dir / f"agent_tester_cycle{cycle}.md"
    test_log_dir = log_dir / f"test_logs_cycle{cycle}"
    run_log = test_log_dir / "run.log"

    resolved_fsdb, reason = _resolve_fsdb(fsdb, run_log=run_log, environment=env)
    if reason:
        outcome = WaveDebugOutcome("unavailable", reason)
        _append_tester_log(agent_log, attempt=attempt, outcome=outcome, fsdb=fsdb)
        return outcome

    if not launcher.is_file():
        outcome = WaveDebugOutcome(
            "unavailable", f"waveform debugger launcher was not found: {launcher}"
        )
        _append_tester_log(
            agent_log, attempt=attempt, outcome=outcome, fsdb=resolved_fsdb
        )
        return outcome

    output_dir = test_log_dir / f"wave_debug_attempt{attempt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(launcher),
        "diagnose",
        "--failure-kind",
        failure_kind,
        "--output-dir",
        str(output_dir),
    ]
    if env.get(INSECURE_HOST_KEY_ENV, "").lower() in {"1", "true", "yes"}:
        command.append("--insecure-host-key")
    command.append(resolved_fsdb)

    stdout_path = output_dir / "command.stdout.json"
    stderr_path = output_dir / "command.stderr.log"
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                env=env,
                check=False,
                text=True,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        outcome = WaveDebugOutcome(
            "failed",
            f"private waveform diagnosis exceeded {timeout_seconds}s; see {stderr_path}",
        )
        _append_tester_log(
            agent_log, attempt=attempt, outcome=outcome, fsdb=resolved_fsdb
        )
        return outcome
    except OSError as error:
        outcome = WaveDebugOutcome(
            "failed", f"could not execute private waveform debugger: {error}"
        )
        _append_tester_log(
            agent_log, attempt=attempt, outcome=outcome, fsdb=resolved_fsdb
        )
        return outcome

    evidence_path = output_dir / "evidence.json"
    if result.returncode != 0:
        try:
            detail = stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            detail = ""
        outcome = WaveDebugOutcome(
            "failed",
            f"private waveform debugger exited {result.returncode}: "
            f"{_single_line(detail[-1000:]) or 'no stderr'}",
        )
        _append_tester_log(
            agent_log, attempt=attempt, outcome=outcome, fsdb=resolved_fsdb
        )
        return outcome

    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        outcome = WaveDebugOutcome(
            "failed", f"private waveform evidence is missing or invalid: {error}"
        )
        _append_tester_log(
            agent_log, attempt=attempt, outcome=outcome, fsdb=resolved_fsdb
        )
        return outcome

    status = str(evidence.get("status") or "inconclusive")
    findings = evidence.get("findings") or []
    outcome = WaveDebugOutcome(
        status,
        (
            f"{len(findings)} deterministic finding(s)"
            if findings
            else "no registered detector matched; continue normal diagnosis"
        ),
        evidence_path=evidence_path,
    )
    _append_tester_log(
        agent_log,
        attempt=attempt,
        outcome=outcome,
        fsdb=resolved_fsdb,
        evidence=evidence,
    )
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Best-effort private waveform diagnosis. Availability and execution "
            "failures are recorded but always remain non-fatal to CodeGen."
        )
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument(
        "--failure-kind",
        choices=("hang", "timeout", "mismatch", "unknown"),
        default="unknown",
    )
    parser.add_argument("--fsdb")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    launcher = Path(__file__).with_name("llk_debug.py")
    try:
        outcome = run_optional_wave_debug(
            log_dir=args.log_dir,
            cycle=args.cycle,
            attempt=args.attempt,
            failure_kind=args.failure_kind,
            fsdb=args.fsdb,
            launcher=launcher,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"WAVE_DEBUG status={outcome.status} summary={outcome.summary}")
    except Exception as error:
        # This bridge is deliberately fail-open: waveform tooling supplements the
        # normal CodeGen loop and must never turn an otherwise valid run terminal.
        agent_log = args.log_dir / f"agent_tester_cycle{args.cycle}.md"
        outcome = WaveDebugOutcome(
            "failed", f"optional waveform bridge raised {type(error).__name__}: {error}"
        )
        try:
            _append_tester_log(
                agent_log,
                attempt=args.attempt,
                outcome=outcome,
                fsdb=args.fsdb,
            )
        except OSError:
            pass
        print(f"WAVE_DEBUG status=failed summary={outcome.summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
