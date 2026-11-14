# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Find and parse one host's health-check runs straight from /data. Nothing is stored."""

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from .config import BMC_REBOOT_ROOT, CLUSTER_REPORT_ROOTS, MACHINE_HC_ROOT, RECOVER_LOG_ROOT, STACKSTORM_ROOT
from .model import HostData, UnitResult
from .sources import cluster_hc, machine_hc

DAY = 86400.0
PARSERS = {
    "machine_hc": machine_hc.parse_run,
    "bmc_reboot": machine_hc.parse_run,
    "cluster_report": cluster_hc.parse_report,
    "cluster_log": cluster_hc.parse_log,
}
RECOVER_DIR_DATE_RE = re.compile(r"-(\d{8})T\d{6}Z")
# Cluster HC files that hold runs; they are found by the host name in their text.
CLUSTER_PATTERNS = (
    ("cluster_report", [str(root / "*/*/*.json") for root in CLUSTER_REPORT_ROOTS]),
    (
        "cluster_log",
        [
            str(RECOVER_LOG_ROOT / "*/recover_*.log"),
            str(STACKSTORM_ROOT / "*/logs/recover-*.log"),
            str(STACKSTORM_ROOT / "nightly-health-check/*/physical_validation/*/*/cluster_validation_iteration_*.log"),
        ],
    ),
)


def parse_unit(job: tuple[str, str]) -> UnitResult:
    source, unit = job
    try:
        return PARSERS[source](unit)
    except Exception as exc:  # one unreadable file must not stop the rest
        return UnitResult(unit=unit, error=f"{type(exc).__name__}: {exc}"[:300])


def _run_dirs(root: Path, host: str, since_day: str) -> list[str]:
    """`<root>/<host>/<date>/<run>/` directories with a diag_report.json, dated `since_day` or later."""
    return [
        str(p.parent) for p in sorted((root / host).glob("*/*/diag_report.json")) if p.parent.parent.name >= since_day
    ]


def _files_naming(host: str, roots: list[Path], since_ts: float) -> list[Path]:
    roots = [str(r) for r in roots if r.exists()]
    if not roots:
        return []
    if shutil.which("rg"):
        cmd = ["rg", "-l", "-F", "--no-messages", "-g", "*.json", "-g", "*.log", host, *roots]
    else:
        cmd = ["grep", "-rlF", "--include=*.json", "--include=*.log", host, *roots]
    found = subprocess.run(["nice", "-n", "10", *cmd], capture_output=True, text=True).stdout.splitlines()
    out = []
    for path in map(Path, found):
        try:
            if path.stat().st_mtime >= since_ts and os.access(path, os.R_OK):
                out.append(path)
        except OSError:
            continue
    return out


def _cluster_search_roots(since_day: str, since_ts: float) -> list[Path]:
    """Only directories that can hold files from the window: report `<kind>/<date>/`, recover `<user>-<date>T...`."""
    roots = [d for root in CLUSTER_REPORT_ROOTS for d in root.glob("*/*") if d.name >= since_day]
    compact = since_day.replace("-", "")
    for d in RECOVER_LOG_ROOT.iterdir() if RECOVER_LOG_ROOT.is_dir() else []:
        m = RECOVER_DIR_DATE_RE.search(d.name)
        if not m or m.group(1) >= compact:
            roots.append(d)
    for d in STACKSTORM_ROOT.glob("*/logs"):
        try:
            if d.stat().st_mtime >= since_ts:
                roots.append(d)
        except OSError:
            continue
    return roots + [STACKSTORM_ROOT / "nightly-health-check"]


def host_units(host: str, as_of: float, days: int) -> list[tuple[str, str]]:
    """(source, unit) for every run of `host` that can fall in the window."""
    since_ts = as_of - (days + 1) * DAY
    since_day = datetime.fromtimestamp(since_ts, timezone.utc).strftime("%Y-%m-%d")
    units = [("machine_hc", d) for d in _run_dirs(MACHINE_HC_ROOT, host, since_day)]
    units += [("bmc_reboot", d) for d in _run_dirs(BMC_REBOOT_ROOT, host, since_day)]
    for path in _files_naming(host, _cluster_search_roots(since_day, since_ts), since_ts):
        source = next((s for s, patterns in CLUSTER_PATTERNS if any(path.match(p) for p in patterns)), None)
        if source:
            units.append((source, str(path)))
    return units


def collect(host: str, as_of: float, days: int, workers: int = 8) -> HostData:
    units = host_units(host, as_of, days)
    data = HostData()
    with Pool(max(1, min(workers, len(units) // 8))) as pool:
        for res in pool.imap_unordered(parse_unit, units, chunksize=4):
            data.add(res)
    return data
