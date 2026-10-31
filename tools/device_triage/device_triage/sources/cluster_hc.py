# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cluster HC: `exabox.cluster_health.v1` records plus recover / validation logs."""

import json
import os
import re
from pathlib import Path

from .. import hosts
from ..logscan import iso_ts, read_lines, scan_lines, utc_ts
from ..model import Event, Run, UnitResult
from ..signatures import ANALYZER_CODES

STATUS = {"passed": "PASS", "failed": "FAIL", "degraded": "WARN"}
PATH_TS_RES = (
    (re.compile(r"(\d{8}T\d{6})Z"), "%Y%m%dT%H%M%S"),
    (re.compile(r"(\d{8}_\d{6})"), "%Y%m%d_%H%M%S"),
)
DOWNGRADE = {"fail": "warn", "warn": "info", "info": "info"}


def parse_report(path: str) -> UnitResult:
    res = UnitResult(unit=path)
    with open(path) as f:
        rec = json.load(f)
    test_type = rec.get("test_type") or Path(path).parts[-3]
    run_id = f"chc:{rec.get('record_id') or Path(path).stem}"
    run_hosts = sorted({h for h in (hosts.normalize_host(x) for x in rec.get("hosts") or []) if h})
    status = STATUS.get(rec.get("status"), str(rec.get("status") or "").upper())
    ts = iso_ts(rec.get("ts"))
    duration = rec.get("duration_s") if isinstance(rec.get("duration_s"), (int, float)) else None
    code = rec.get("analyzer_code")
    labels = rec.get("labels") or {}
    summary = f"{test_type} {rec.get('status')}" + (f" analyzer_code={code}" if code is not None else "")
    if rec.get("pass_pct") is not None:
        summary += f" pass_pct={rec.get('pass_pct')}"
    res.runs.append(
        Run(
            run_id=run_id,
            source="cluster_hc",
            kind=test_type,
            hosts=run_hosts,
            ts_start=ts - duration if ts and duration else ts,
            ts_end=ts,
            status=status,
            summary=summary,
            path=path,
            artifact=str(rec.get("artifact_uri") or ""),
            extra={
                "labels": labels,
                "trigger": rec.get("trigger_kind"),
                "by": rec.get("triggered_by"),
                "analyzer_code": code,
                "pass_pct": rec.get("pass_pct"),
                "cluster": rec.get("cluster"),
            },
        )
    )
    if status not in ("FAIL", "WARN"):
        return res
    if isinstance(code, int) and code in ANALYZER_CODES:
        category, signal, severity = ANALYZER_CODES[code]
    else:
        category, signal, severity = "unknown", f"{test_type}_failed", "warn"
    # A failed multi-host record does not say which host broke it.
    if len(run_hosts) > 1:
        severity = DOWNGRADE[severity]
    scope = "host" if len(run_hosts) == 1 else f"multi-host:{len(run_hosts)}"
    for h in run_hosts:
        res.events.append(
            Event(
                host=h,
                ts=ts,
                source="cluster_hc",
                origin=f"record:{test_type}",
                category=category,
                signal=signal,
                severity=severity,
                detail=summary,
                scope=scope,
                run_id=run_id,
                path=path,
            )
        )
    return res


def _path_ts(path: str) -> float | None:
    for pattern, fmt in PATH_TS_RES:
        m = pattern.search(path)
        if m:
            return utc_ts(m.group(1), fmt)
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def parse_log(path: str) -> UnitResult:
    res = UnitResult(unit=path)
    kind = "validation" if "cluster_validation_iteration" in path else "recover"
    events = scan_lines(
        read_lines(path),
        default_host=None,
        base_ts=_path_ts(path),
        source="cluster_hc",
        origin=f"log:{kind}",
        path=path,
    )
    res.events = [e for e in events if e.host]
    return res
