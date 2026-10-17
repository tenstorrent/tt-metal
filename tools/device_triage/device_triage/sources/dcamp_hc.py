# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Machine HC runs rebuilt from DCAMP health-check bot tickets: `cache/dcamp_hc/<host>/<date>/<job>/`.

`jira_sync.sync_hc_runs` lays them out like hc_logs runs. Runs with a diag_report.json (July on) parse exactly
like on-disk machine HC runs; the older fabric health check (April-June) left only its console log, scanned
line by line. Only failing runs were ticketed, so these add failure history, not pass history.
"""

import json
import re
from pathlib import Path

from ..config import DCAMP_HC_ROOT, MAX_EVIDENCE_CHARS
from ..logscan import read_lines, scan_lines
from ..model import Run, UnitResult
from . import machine_hc

GTEST_FAILED_RE = re.compile(r"^\[  FAILED  \] (\S+) \(\d+ ms\)")


def discover() -> list[tuple[str, str]]:
    return [(str(p.parent), str(p)) for p in DCAMP_HC_ROOT.glob("*/*/*/ticket.json")]


def parse(run_dir_str: str) -> UnitResult:
    run_dir = Path(run_dir_str)
    meta = json.loads((run_dir / "ticket.json").read_text())
    if (run_dir / "diag_report.json").exists():
        res = machine_hc.parse_run(run_dir_str)
        _rename_host(res, meta["host"], f"mhc:{meta['host']}/{run_dir.parts[-2]}/{run_dir.parts[-1]}")
        if meta.get("diag") == "repaired" and (run_dir / "suite.log").exists():
            res.events += _scan_console(run_dir, meta, res.runs[0].run_id)
    else:
        res = _parse_console(run_dir, meta)
    for run in res.runs:
        run.extra["ticket"] = meta["key"]
        # Ticketed means it failed, even when the (truncated) report only kept passing checks.
        if run.status != "FAIL":
            run.summary = f"ticketed failure, report {meta.get('diag')}: {run.summary}"[:MAX_EVIDENCE_CHARS]
            run.status = "FAIL"
    return res


def _rename_host(res: UnitResult, host: str, run_id: str) -> None:
    """Reports may name a container (`6ddeaba6e2a2`) or the pre-superpod name (`bh-glx-b05u02`): use the ticket's."""
    reported = res.runs[0].hosts[0] if res.runs and res.runs[0].hosts else None
    for ev in res.events:
        if ev.host in (None, reported):
            ev.host = host
        ev.run_id = run_id
    for run in res.runs:
        run.hosts, run.run_id = [host], run_id
    res.chip_map = [(host if h == reported else h, bdf, chip, o) for h, bdf, chip, o in res.chip_map]


def _scan_console(run_dir: Path, meta: dict, run_id: str) -> list:
    log = run_dir / "suite.log"
    return scan_lines(
        read_lines(str(log)),
        default_host=meta["host"],
        base_ts=meta["ts"],
        source="machine_hc",
        origin="log:console",
        path=str(log),
        run_id=run_id,
    )


def _parse_console(run_dir: Path, meta: dict) -> UnitResult:
    host = meta["host"]
    run_id = f"mhc:{host}/{run_dir.parts[-2]}/{run_dir.parts[-1]}"
    lines = read_lines(str(run_dir / "suite.log"))
    res = UnitResult(unit=str(run_dir))
    res.events = _scan_console(run_dir, meta, run_id)
    failed = list(dict.fromkeys(m.group(1) for line in lines if (m := GTEST_FAILED_RE.match(line))))
    res.runs.append(
        Run(
            run_id=run_id,
            source="machine_hc",
            kind="fabric-hc",
            hosts=[host],
            ts_start=meta["ts"],
            ts_end=None,
            status="FAIL",
            summary=(", ".join(failed) or "fabric health check failed")[:MAX_EVIDENCE_CHARS],
            path=str(run_dir),
            extra={"console_only": True},
        )
    )
    return res
