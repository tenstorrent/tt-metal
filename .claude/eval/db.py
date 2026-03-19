"""Database storage for eval run tracking.

Supports both SQLite (local dev) and PostgreSQL (production/Superset).

Backend selection:
    - Set EVAL_DATABASE_URL env var for PostgreSQL:
        export EVAL_DATABASE_URL="postgresql://user:pass@host:5432/eval"
    - Otherwise falls back to SQLite at DEFAULT_SQLITE_PATH

Schema:
- runs: one row per (prompt, run_number) pair, with live status/phase tracking
- test_results: one row per individual test parametrization
- score_criteria: per-criterion breakdown from score.py
- kernels: generated kernel source files per run
- host_code: host-side Python files per run
- artifacts: self-reflection and other text artifacts per run
- tdd_state: TDD pipeline state JSON per run
- kw_breadcrumbs: agent breadcrumb JSONL files per run
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DEFAULT_SQLITE_PATH = Path(f"/localdev/{os.environ.get('USER', 'unknown')}/eval_runs.db")

# Run status lifecycle
RUN_STATUSES = ("queued", "cloning", "building", "running", "testing", "scoring", "complete", "failed")

# ---------------------------------------------------------------------------
# Schema — written in a dialect-neutral subset of SQL.
# PostgreSQL uses SERIAL instead of INTEGER PRIMARY KEY AUTOINCREMENT, but
# we handle that in _create_schema() per backend.
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    starting_branch TEXT NOT NULL,
    starting_commit TEXT NOT NULL,
    created_branch TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    phase TEXT,
    score_total REAL,
    score_grade TEXT,
    golden_passed INTEGER,
    golden_total INTEGER,
    annotation_score INTEGER,
    annotation_notes TEXT,
    golden_name TEXT,
    duration_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    test_name TEXT NOT NULL,
    test_file TEXT,
    shape TEXT,
    status TEXT NOT NULL,
    failure_category TEXT,
    failure_message TEXT
);

CREATE TABLE IF NOT EXISTS score_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    criterion TEXT NOT NULL,
    raw_score REAL NOT NULL,
    weight REAL NOT NULL,
    weighted_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kernels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    filename TEXT NOT NULL,
    source_code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS host_code (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    filename TEXT NOT NULL,
    source_code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tdd_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kw_breadcrumbs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    agent_name TEXT NOT NULL,
    content TEXT NOT NULL
);
"""

_PG_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    starting_branch TEXT NOT NULL,
    starting_commit TEXT NOT NULL,
    created_branch TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    phase TEXT,
    score_total REAL,
    score_grade TEXT,
    golden_passed INTEGER,
    golden_total INTEGER,
    annotation_score INTEGER,
    annotation_notes TEXT,
    golden_name TEXT,
    duration_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS test_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    test_name TEXT NOT NULL,
    test_file TEXT,
    shape TEXT,
    status TEXT NOT NULL,
    failure_category TEXT,
    failure_message TEXT
);

CREATE TABLE IF NOT EXISTS score_criteria (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    criterion TEXT NOT NULL,
    raw_score REAL NOT NULL,
    weight REAL NOT NULL,
    weighted_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kernels (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    filename TEXT NOT NULL,
    source_code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS host_code (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    filename TEXT NOT NULL,
    source_code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tdd_state (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kw_breadcrumbs (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    agent_name TEXT NOT NULL,
    content TEXT NOT NULL
);
"""

# Migrations for existing databases that predate status/phase columns
_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN golden_name TEXT",
    "ALTER TABLE runs ADD COLUMN duration_seconds INTEGER",
    "ALTER TABLE runs ADD COLUMN status TEXT DEFAULT 'complete'",
    "ALTER TABLE runs ADD COLUMN phase TEXT",
]


# ---------------------------------------------------------------------------
# Connection handling — unified interface for SQLite and PostgreSQL
# ---------------------------------------------------------------------------


def _is_postgres() -> bool:
    return bool(os.environ.get("EVAL_DATABASE_URL"))


def _pg_connect():
    """Connect to PostgreSQL, returning a connection with dict-like rows."""
    import psycopg2
    import psycopg2.extras

    url = os.environ["EVAL_DATABASE_URL"]
    conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _sqlite_connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Connect to SQLite with Row factory."""
    path = str(db_path) if db_path else str(DEFAULT_SQLITE_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _create_schema(conn):
    """Create tables if they don't exist."""
    if _is_postgres():
        cur = conn.cursor()
        cur.execute(_PG_SCHEMA)
        conn.commit()
    else:
        conn.executescript(_SQLITE_SCHEMA)


def _run_migrations(conn):
    """Apply missing column migrations to an existing DB."""
    for stmt in _MIGRATIONS:
        try:
            if _is_postgres():
                cur = conn.cursor()
                cur.execute(stmt)
                conn.commit()
            else:
                conn.execute(stmt)
        except Exception:
            if _is_postgres():
                conn.rollback()


def connect(db_path: Optional[Path] = None):
    """Connect to the eval DB, creating tables if needed.

    Uses PostgreSQL if EVAL_DATABASE_URL is set, otherwise SQLite.
    """
    if _is_postgres():
        conn = _pg_connect()
    else:
        conn = _sqlite_connect(db_path)
    _create_schema(conn)
    _run_migrations(conn)
    return conn


# ---------------------------------------------------------------------------
# Query helpers — abstract %s vs ? placeholder differences
# ---------------------------------------------------------------------------


def _ph():
    """Return the parameter placeholder for the current backend."""
    return "%s" if _is_postgres() else "?"


def _execute(conn, sql, params=None):
    """Execute a query, handling cursor differences between backends."""
    sql = sql.replace("?", "%s") if _is_postgres() else sql
    if _is_postgres():
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur
    else:
        return conn.execute(sql, params or ())


def _executemany(conn, sql, param_list):
    """Execute many, handling placeholder differences."""
    sql = sql.replace("?", "%s") if _is_postgres() else sql
    if _is_postgres():
        cur = conn.cursor()
        cur.executemany(sql, param_list)
        return cur
    else:
        return conn.executemany(sql, param_list)


def _fetchone(conn, sql, params=None) -> Optional[dict]:
    cur = _execute(conn, sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def _fetchall(conn, sql, params=None) -> list:
    cur = _execute(conn, sql, params)
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Insert / update operations
# ---------------------------------------------------------------------------


def insert_run(
    conn,
    *,
    timestamp,
    prompt_name,
    run_number,
    starting_branch,
    starting_commit,
    created_branch,
    status="queued",
    phase=None,
    score_total=None,
    score_grade=None,
    golden_passed=None,
    golden_total=None,
    golden_name=None,
    duration_seconds=None,
) -> int:
    """Insert a run and return its ID. Caller must commit."""
    if _is_postgres():
        row = _fetchone(
            conn,
            """INSERT INTO runs
               (timestamp, prompt_name, run_number, starting_branch, starting_commit,
                created_branch, status, phase, score_total, score_grade, golden_passed,
                golden_total, golden_name, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               RETURNING id""",
            (
                timestamp,
                prompt_name,
                run_number,
                starting_branch,
                starting_commit,
                created_branch,
                status,
                phase,
                score_total,
                score_grade,
                golden_passed,
                golden_total,
                golden_name,
                duration_seconds,
            ),
        )
        return row["id"]
    else:
        cur = _execute(
            conn,
            """INSERT INTO runs
               (timestamp, prompt_name, run_number, starting_branch, starting_commit,
                created_branch, status, phase, score_total, score_grade, golden_passed,
                golden_total, golden_name, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                prompt_name,
                run_number,
                starting_branch,
                starting_commit,
                created_branch,
                status,
                phase,
                score_total,
                score_grade,
                golden_passed,
                golden_total,
                golden_name,
                duration_seconds,
            ),
        )
        return cur.lastrowid


def update_run_status(conn, run_id: int, status: str):
    """Update the status of a run. Caller must commit."""
    _execute(conn, "UPDATE runs SET status = ? WHERE id = ?", (status, run_id))


def update_run_phase(conn, run_id: int, phase: str):
    """Update the current phase of a run. Caller must commit."""
    _execute(conn, "UPDATE runs SET phase = ? WHERE id = ?", (phase, run_id))


def update_run_score(conn, run_id: int, score_total: float, score_grade: str):
    """Update score for a run. Caller must commit."""
    _execute(
        conn,
        "UPDATE runs SET score_total = ?, score_grade = ? WHERE id = ?",
        (score_total, score_grade, run_id),
    )


def update_run_golden(conn, run_id: int, golden_passed: int, golden_total: int):
    """Update golden test results for a run. Caller must commit."""
    _execute(
        conn,
        "UPDATE runs SET golden_passed = ?, golden_total = ? WHERE id = ?",
        (golden_passed, golden_total, run_id),
    )


def update_run_duration(conn, run_id: int, duration_seconds: int):
    """Update duration for a run. Caller must commit."""
    _execute(
        conn,
        "UPDATE runs SET duration_seconds = ? WHERE id = ?",
        (duration_seconds, run_id),
    )


def insert_test_results_batch(conn, run_id: int, results: list):
    """Insert multiple test results at once. Caller must commit."""
    _executemany(
        conn,
        """INSERT INTO test_results
           (run_id, test_name, test_file, shape, status, failure_category, failure_message)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id,
                r["test_name"],
                r.get("test_file"),
                r.get("shape"),
                r["status"],
                r.get("failure_category"),
                r.get("failure_message"),
            )
            for r in results
        ],
    )


def insert_score_criteria(conn, run_id: int, criteria: list):
    """Insert score criteria for a run. Caller must commit."""
    _executemany(
        conn,
        """INSERT INTO score_criteria (run_id, criterion, raw_score, weight, weighted_score)
           VALUES (?, ?, ?, ?, ?)""",
        [(run_id, c["name"], c["raw_score"], c["weight"], c["weighted_score"]) for c in criteria],
    )


def annotate_run(conn, run_id: int, score: int, notes: str = ""):
    """Add or update annotation for a run."""
    _execute(
        conn,
        "UPDATE runs SET annotation_score = ?, annotation_notes = ? WHERE id = ?",
        (score, notes, run_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_all_runs(conn) -> list:
    """Get all runs, newest first."""
    return _fetchall(conn, "SELECT * FROM runs ORDER BY timestamp DESC")


def get_run(conn, run_id: int) -> Optional[dict]:
    """Get a single run by ID."""
    return _fetchone(conn, "SELECT * FROM runs WHERE id = ?", (run_id,))


def find_run_by_branch(conn, created_branch: str) -> Optional[dict]:
    """Find a run by its created_branch. Returns None if not found."""
    return _fetchone(conn, "SELECT * FROM runs WHERE created_branch = ?", (created_branch,))


def get_test_results(conn, run_id: int) -> list:
    """Get all test results for a run."""
    return _fetchall(conn, "SELECT * FROM test_results WHERE run_id = ? ORDER BY id", (run_id,))


def get_score_criteria(conn, run_id: int) -> list:
    """Get score criteria for a run."""
    return _fetchall(conn, "SELECT * FROM score_criteria WHERE run_id = ? ORDER BY id", (run_id,))


def get_failure_summary(conn) -> dict:
    """Get aggregate failure category counts across all runs."""
    rows = _fetchall(
        conn,
        """SELECT failure_category, COUNT(*) as count
           FROM test_results
           WHERE status != 'passed' AND failure_category IS NOT NULL
           GROUP BY failure_category""",
    )
    return {row["failure_category"]: row["count"] for row in rows}


def get_stats(conn) -> dict:
    """Get aggregate statistics."""
    row = _fetchone(conn, "SELECT COUNT(*) as c FROM runs")
    total = row["c"]
    if total == 0:
        return {"total_runs": 0, "avg_score": 0, "pass_rate": 0, "failure_summary": {}}

    avg_row = _fetchone(conn, "SELECT AVG(score_total) as avg FROM runs WHERE score_total IS NOT NULL")
    avg_score = avg_row["avg"] if avg_row["avg"] is not None else 0

    full_pass = _fetchone(
        conn, "SELECT COUNT(*) as c FROM runs WHERE golden_passed = golden_total AND golden_total > 0"
    )

    active_runs = _fetchone(conn, "SELECT COUNT(*) as c FROM runs WHERE status NOT IN ('complete', 'failed')")

    return {
        "total_runs": total,
        "active_runs": active_runs["c"],
        "avg_score": round(avg_score, 1),
        "pass_rate": round(full_pass["c"] / total * 100, 1) if total > 0 else 0,
        "failure_summary": get_failure_summary(conn),
    }


def insert_kernels(conn, run_id: int, kernels: list):
    """Insert kernel C++ source files. Each item: {"filename": str, "source_code": str}. Caller must commit."""
    _executemany(
        conn,
        "INSERT INTO kernels (run_id, filename, source_code) VALUES (?, ?, ?)",
        [(run_id, k["filename"], k["source_code"]) for k in kernels],
    )


def get_kernels(conn, run_id: int) -> list:
    """Get all kernel C++ files for a run."""
    return _fetchall(conn, "SELECT * FROM kernels WHERE run_id = ? ORDER BY filename", (run_id,))


def insert_host_code(conn, run_id: int, files: list):
    """Insert host-side Python files. Each item: {"filename": str, "source_code": str}. Caller must commit."""
    _executemany(
        conn,
        "INSERT INTO host_code (run_id, filename, source_code) VALUES (?, ?, ?)",
        [(run_id, f["filename"], f["source_code"]) for f in files],
    )


def get_host_code(conn, run_id: int) -> list:
    """Get all host-side Python files for a run."""
    return _fetchall(conn, "SELECT * FROM host_code WHERE run_id = ? ORDER BY filename", (run_id,))


def insert_artifact(conn, run_id: int, name: str, content: str):
    """Insert a text artifact (e.g. self_reflection.md). Caller must commit."""
    _execute(
        conn,
        "INSERT INTO artifacts (run_id, name, content) VALUES (?, ?, ?)",
        (run_id, name, content),
    )


def get_artifacts(conn, run_id: int) -> list:
    """Get all artifacts for a run."""
    return _fetchall(conn, "SELECT * FROM artifacts WHERE run_id = ? ORDER BY name", (run_id,))


def insert_tdd_state(conn, run_id: int, content: str):
    """Insert the .tdd_state.json content for a run. Caller must commit."""
    _execute(
        conn,
        "INSERT INTO tdd_state (run_id, content) VALUES (?, ?)",
        (run_id, content),
    )


def get_tdd_state(conn, run_id: int) -> Optional[str]:
    """Get .tdd_state.json content for a run, or None."""
    row = _fetchone(conn, "SELECT content FROM tdd_state WHERE run_id = ?", (run_id,))
    return row["content"] if row else None


def insert_kw_breadcrumbs(conn, run_id: int, breadcrumbs: list):
    """Insert breadcrumb files. Each item: {"agent_name": str, "content": str}. Caller must commit."""
    _executemany(
        conn,
        "INSERT INTO kw_breadcrumbs (run_id, agent_name, content) VALUES (?, ?, ?)",
        [(run_id, b["agent_name"], b["content"]) for b in breadcrumbs],
    )


def get_kw_breadcrumbs(conn, run_id: int) -> list:
    """Get all breadcrumb files for a run."""
    return _fetchall(conn, "SELECT * FROM kw_breadcrumbs WHERE run_id = ? ORDER BY agent_name", (run_id,))
