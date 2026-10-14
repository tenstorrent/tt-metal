# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Incremental SQLite index of runs and events across all log sources.

A unit (run directory or log file) is re-parsed only when its fingerprint file's
mtime or size changes, or when the parsing code for its source changed; its rows
are replaced wholesale. Parsing runs in a niced worker pool; the parent is the
only writer.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path

from .config import INDEX_DB, MAX_EVIDENCE_CHARS, MAX_LINES_PER_FILE
from .model import UnitResult
from .sources import cluster_hc, dcamp_hc, machine_hc

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (unit TEXT PRIMARY KEY, source TEXT, mtime REAL, size INTEGER,
    indexed_at REAL, n_events INTEGER, error TEXT, version TEXT);
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, unit TEXT, source TEXT, kind TEXT, ts_start REAL,
    ts_end REAL, status TEXT, summary TEXT, path TEXT, artifact TEXT, extra TEXT);
CREATE TABLE IF NOT EXISTS run_hosts (run_id TEXT, host TEXT, PRIMARY KEY (run_id, host));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, unit TEXT, run_id TEXT, host TEXT, ts REAL,
    source TEXT, origin TEXT, category TEXT, signal TEXT, severity TEXT, chips TEXT, scope TEXT, detail TEXT,
    path TEXT, line INTEGER, count INTEGER);
CREATE TABLE IF NOT EXISTS chip_obs (unit TEXT, host TEXT, bdf TEXT, chip TEXT, ordinal INTEGER);
CREATE INDEX IF NOT EXISTS events_host_ts ON events(host, ts);
CREATE INDEX IF NOT EXISTS events_unit ON events(unit);
CREATE INDEX IF NOT EXISTS events_signal ON events(signal);
CREATE INDEX IF NOT EXISTS runs_unit ON runs(unit);
CREATE INDEX IF NOT EXISTS run_hosts_host ON run_hosts(host);
CREATE INDEX IF NOT EXISTS chip_obs_unit ON chip_obs(unit);
CREATE INDEX IF NOT EXISTS chip_obs_host ON chip_obs(host);
"""

# name -> (discover() -> [(unit, fingerprint_path)], parse(unit) -> UnitResult)
SOURCES: dict[str, tuple[Callable[[], list[tuple[str, str]]], Callable[[str], UnitResult]]] = {
    "machine_hc": (machine_hc.discover_machine_hc, machine_hc.parse_run),
    "bmc_reboot": (machine_hc.discover_bmc_reboot, machine_hc.parse_run),
    "cluster_report": (cluster_hc.discover_reports, cluster_hc.parse_report),
    "cluster_log": (cluster_hc.discover_logs, cluster_hc.parse_log),
    "dcamp_hc": (dcamp_hc.discover, dcamp_hc.parse),
}
_PACKAGE = Path(__file__).resolve().parent
_PARSER_FILES = {
    "machine_hc": ("sources/machine_hc.py",),
    "bmc_reboot": ("sources/machine_hc.py",),
    "cluster_report": ("sources/cluster_hc.py",),
    "cluster_log": ("sources/cluster_hc.py",),
    "dcamp_hc": ("sources/dcamp_hc.py", "sources/machine_hc.py"),
    "jira_attachment": (),
}
_COMMON_PARSER_FILES = ("signatures.py", "logscan.py", "hosts.py", "model.py")


def parser_version(source: str) -> str:
    digest = hashlib.sha1()
    for name in (*_COMMON_PARSER_FILES, *_PARSER_FILES[source]):
        digest.update((_PACKAGE / name).read_bytes())
    digest.update(repr((MAX_LINES_PER_FILE, MAX_EVIDENCE_CHARS)).encode())
    return digest.hexdigest()[:12]


def connect(db_path: Path = INDEX_DB) -> sqlite3.Connection:
    """Read-write for the indexer. A database you can't write (a published snapshot) opens read-only."""
    if db_path.exists() and not os.access(db_path, os.W_OK):
        return sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    if "version" not in {row[1] for row in conn.execute("PRAGMA table_info(units)")}:
        conn.execute("ALTER TABLE units ADD COLUMN version TEXT")
    return conn


def publish(conn: sqlite3.Connection, dest_dir: Path) -> Path:
    """Write a read-only snapshot for shared readers. Readers that have the old one open keep reading it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "index.sqlite.tmp"
    tmp.unlink(missing_ok=True)
    conn.execute("VACUUM INTO ?", (str(tmp),))
    tmp.chmod(0o444)
    target = dest_dir / "index.sqlite"
    tmp.replace(target)
    return target


def lower_priority() -> None:
    """Index from a shared box over NFS without competing with workloads."""
    os.nice(10)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=False, capture_output=True)


def _parse(job: tuple[str, str]) -> UnitResult:
    source, unit = job
    try:
        return SOURCES[source][1](unit)
    except Exception as exc:  # one unreadable unit must not stop the sweep
        return UnitResult(unit=unit, error=f"{type(exc).__name__}: {exc}"[:300])


def _delete_unit(conn: sqlite3.Connection, unit: str) -> None:
    conn.execute("DELETE FROM run_hosts WHERE run_id IN (SELECT run_id FROM runs WHERE unit = ?)", (unit,))
    for table in ("runs", "events", "chip_obs", "units"):
        conn.execute(f"DELETE FROM {table} WHERE unit = ?", (unit,))


def _store(conn: sqlite3.Connection, source: str, res: UnitResult, mtime: float, size: int, version: str) -> None:
    _delete_unit(conn, res.unit)
    for r in res.runs:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                r.run_id,
                res.unit,
                r.source,
                r.kind,
                r.ts_start,
                r.ts_end,
                r.status,
                r.summary,
                r.path,
                r.artifact,
                json.dumps(r.extra),
            ),
        )
        conn.executemany("INSERT OR IGNORE INTO run_hosts VALUES (?,?)", [(r.run_id, h) for h in r.hosts])
    conn.executemany(
        "INSERT INTO events (unit, run_id, host, ts, source, origin, category, signal, severity, chips, scope, detail,"
        " path, line, count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                res.unit,
                e.run_id,
                e.host,
                e.ts,
                e.source,
                e.origin,
                e.category,
                e.signal,
                e.severity,
                e.chips,
                e.scope,
                e.detail,
                e.path,
                e.line,
                e.count,
            )
            for e in res.events
        ],
    )
    conn.executemany("INSERT INTO chip_obs VALUES (?,?,?,?,?)", [(res.unit, *c) for c in res.chip_map])
    conn.execute(
        "INSERT OR REPLACE INTO units VALUES (?,?,?,?,?,?,?,?)",
        (res.unit, source, mtime, size, time.time(), len(res.events), res.error or None, version),
    )


def build(
    sources: list[str] | None = None,
    workers: int = 8,
    limit: int | None = None,
    db_path: Path = INDEX_DB,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> None:
    lower_priority()
    conn = connect(db_path)
    for source in sources or list(SOURCES):
        discover, _ = SOURCES[source]
        t0 = time.time()
        units = discover()
        version = parser_version(source)
        known = {
            u: (m, s, v)
            for u, m, s, v in conn.execute("SELECT unit, mtime, size, version FROM units WHERE source = ?", (source,))
        }
        todo: list[tuple[str, float, int]] = []
        for unit, fp in units:
            try:
                st = os.stat(fp)
            except OSError:
                continue
            if known.get(unit) != (st.st_mtime, st.st_size, version):
                todo.append((unit, st.st_mtime, st.st_size))
        gone = set(known) - {u for u, _ in units}
        for unit in gone:
            _delete_unit(conn, unit)
        if limit:
            todo = todo[:limit]
        log(
            f"[{source}] {len(units)} units, {len(todo)} to (re)index, {len(gone)} removed ({time.time() - t0:.0f}s discovery)"
        )
        stamps = {u: (m, s) for u, m, s in todo}
        done = errors = 0
        with Pool(workers) as pool:
            for res in pool.imap_unordered(_parse, [(source, u) for u, _, _ in todo], chunksize=4):
                _store(conn, source, res, *stamps[res.unit], version)
                done += 1
                errors += bool(res.error)
                if done % 500 == 0:
                    conn.commit()
                    log(f"[{source}] {done}/{len(todo)} ({errors} errors, {time.time() - t0:.0f}s)")
        conn.commit()
        log(f"[{source}] done: {done} units, {errors} errors, {time.time() - t0:.0f}s")
    conn.close()
