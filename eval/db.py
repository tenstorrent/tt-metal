"""SQLite storage for eval run tracking.

Schema:
- runs: one row per (prompt, run_number) pair
- test_results: one row per individual test parametrization
- score_criteria: per-criterion breakdown from score.py
- kernels: generated kernel source files per run
- artifacts: self-reflection and other text artifacts per run
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(f"/localdev/{os.environ.get('USER', 'unknown')}/eval_runs.db")

SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    starting_branch TEXT NOT NULL,
    starting_commit TEXT NOT NULL,
    created_branch TEXT NOT NULL,
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


MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN golden_name TEXT",
    "ALTER TABLE runs ADD COLUMN duration_seconds INTEGER",
]


def _run_migrations(conn):
    """Apply any missing column migrations to an existing DB."""
    for stmt in MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Connect to the eval DB, creating tables if needed."""
    path = str(db_path) if db_path else str(DEFAULT_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _run_migrations(conn)
    return conn


def insert_run(
    conn,
    *,
    timestamp,
    prompt_name,
    run_number,
    starting_branch,
    starting_commit,
    created_branch,
    score_total=None,
    score_grade=None,
    golden_passed=None,
    golden_total=None,
    golden_name=None,
    duration_seconds=None,
) -> int:
    """Insert a run and return its ID. Caller must commit."""
    cur = conn.execute(
        """INSERT INTO runs
           (timestamp, prompt_name, run_number, starting_branch, starting_commit,
            created_branch, score_total, score_grade, golden_passed, golden_total,
            golden_name, duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            timestamp,
            prompt_name,
            run_number,
            starting_branch,
            starting_commit,
            created_branch,
            score_total,
            score_grade,
            golden_passed,
            golden_total,
            golden_name,
            duration_seconds,
        ),
    )
    return cur.lastrowid


def insert_test_results_batch(conn, run_id: int, results: list):
    """Insert multiple test results at once. Caller must commit."""
    conn.executemany(
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
    conn.executemany(
        """INSERT INTO score_criteria (run_id, criterion, raw_score, weight, weighted_score)
           VALUES (?, ?, ?, ?, ?)""",
        [(run_id, c["name"], c["raw_score"], c["weight"], c["weighted_score"]) for c in criteria],
    )


def annotate_run(conn, run_id: int, score: int, notes: str = ""):
    """Add or update annotation for a run."""
    conn.execute(
        "UPDATE runs SET annotation_score = ?, annotation_notes = ? WHERE id = ?",
        (score, notes, run_id),
    )
    conn.commit()


def get_all_runs(conn) -> list:
    """Get all runs, newest first."""
    rows = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC").fetchall()
    return [dict(r) for r in rows]


def get_run(conn, run_id: int) -> Optional[dict]:
    """Get a single run by ID."""
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_test_results(conn, run_id: int) -> list:
    """Get all test results for a run."""
    rows = conn.execute("SELECT * FROM test_results WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def get_score_criteria(conn, run_id: int) -> list:
    """Get score criteria for a run."""
    rows = conn.execute("SELECT * FROM score_criteria WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def get_failure_summary(conn) -> dict:
    """Get aggregate failure category counts across all runs."""
    rows = conn.execute(
        """SELECT failure_category, COUNT(*) as count
           FROM test_results
           WHERE status != 'passed' AND failure_category IS NOT NULL
           GROUP BY failure_category"""
    ).fetchall()
    return {row["failure_category"]: row["count"] for row in rows}


def get_stats(conn) -> dict:
    """Get aggregate statistics."""
    total = conn.execute("SELECT COUNT(*) as c FROM runs").fetchone()["c"]
    if total == 0:
        return {"total_runs": 0, "avg_score": 0, "pass_rate": 0, "failure_summary": {}}

    avg_row = conn.execute("SELECT AVG(score_total) as avg FROM runs WHERE score_total IS NOT NULL").fetchone()
    avg_score = avg_row["avg"] if avg_row["avg"] is not None else 0

    full_pass = conn.execute(
        "SELECT COUNT(*) as c FROM runs WHERE golden_passed = golden_total AND golden_total > 0"
    ).fetchone()["c"]

    return {
        "total_runs": total,
        "avg_score": round(avg_score, 1),
        "pass_rate": round(full_pass / total * 100, 1) if total > 0 else 0,
        "failure_summary": get_failure_summary(conn),
    }


def insert_kernels(conn, run_id: int, kernels: list):
    """Insert kernel C++ source files. Each item: {"filename": str, "source_code": str}. Caller must commit."""
    conn.executemany(
        "INSERT INTO kernels (run_id, filename, source_code) VALUES (?, ?, ?)",
        [(run_id, k["filename"], k["source_code"]) for k in kernels],
    )


def get_kernels(conn, run_id: int) -> list:
    """Get all kernel C++ files for a run."""
    rows = conn.execute("SELECT * FROM kernels WHERE run_id = ? ORDER BY filename", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def insert_host_code(conn, run_id: int, files: list):
    """Insert host-side Python files. Each item: {"filename": str, "source_code": str}. Caller must commit."""
    conn.executemany(
        "INSERT INTO host_code (run_id, filename, source_code) VALUES (?, ?, ?)",
        [(run_id, f["filename"], f["source_code"]) for f in files],
    )


def get_host_code(conn, run_id: int) -> list:
    """Get all host-side Python files for a run."""
    rows = conn.execute("SELECT * FROM host_code WHERE run_id = ? ORDER BY filename", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def insert_artifact(conn, run_id: int, name: str, content: str):
    """Insert a text artifact (e.g. self_reflection.md). Caller must commit."""
    conn.execute(
        "INSERT INTO artifacts (run_id, name, content) VALUES (?, ?, ?)",
        (run_id, name, content),
    )


def get_artifacts(conn, run_id: int) -> list:
    """Get all artifacts for a run."""
    rows = conn.execute("SELECT * FROM artifacts WHERE run_id = ? ORDER BY name", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def insert_tdd_state(conn, run_id: int, content: str):
    """Insert the .tdd_state.json content for a run. Caller must commit."""
    conn.execute(
        "INSERT INTO tdd_state (run_id, content) VALUES (?, ?)",
        (run_id, content),
    )


def get_tdd_state(conn, run_id: int) -> Optional[str]:
    """Get .tdd_state.json content for a run, or None."""
    row = conn.execute("SELECT content FROM tdd_state WHERE run_id = ?", (run_id,)).fetchone()
    return row["content"] if row else None


def insert_kw_breadcrumbs(conn, run_id: int, breadcrumbs: list):
    """Insert breadcrumb files. Each item: {"agent_name": str, "content": str}. Caller must commit."""
    conn.executemany(
        "INSERT INTO kw_breadcrumbs (run_id, agent_name, content) VALUES (?, ?, ?)",
        [(run_id, b["agent_name"], b["content"]) for b in breadcrumbs],
    )


def get_kw_breadcrumbs(conn, run_id: int) -> list:
    """Get all breadcrumb files for a run."""
    rows = conn.execute("SELECT * FROM kw_breadcrumbs WHERE run_id = ? ORDER BY agent_name", (run_id,)).fetchall()
    return [dict(r) for r in rows]
