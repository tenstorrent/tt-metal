"""Tests for eval.db — uses in-memory SQLite, no disk or device needed."""

import sqlite3
from pathlib import Path

import pytest

from eval import db


@pytest.fixture
def conn():
    """In-memory database connection with schema applied."""
    c = db.connect(Path(":memory:"))
    yield c
    c.close()


def _make_run(conn, **overrides):
    """Insert a run with sensible defaults, return its ID."""
    defaults = dict(
        timestamp="2026-03-09T14:30:00",
        prompt_name="layer_norm_rm",
        run_number=1,
        starting_branch="mare/eval",
        starting_commit="abc123",
        created_branch="2026_03_09_run1_layer_norm_rm",
    )
    defaults.update(overrides)
    return db.insert_run(conn, **defaults)


# --- Schema ---


def test_schema_creates_tables(conn):
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    names = {r["name"] for r in tables}
    assert "runs" in names
    assert "test_results" in names
    assert "score_criteria" in names


# --- insert_run / get_run ---


def test_insert_and_get_run(conn):
    rid = _make_run(conn, prompt_name="softmax")
    run = db.get_run(conn, rid)
    assert run is not None
    assert run["prompt_name"] == "softmax"
    assert run["starting_branch"] == "mare/eval"
    assert run["annotation_score"] is None


def test_get_run_not_found(conn):
    assert db.get_run(conn, 999) is None


def test_get_all_runs_ordered(conn):
    _make_run(conn, timestamp="2026-03-01T10:00:00", prompt_name="first")
    _make_run(conn, timestamp="2026-03-09T10:00:00", prompt_name="second")
    runs = db.get_all_runs(conn)
    assert len(runs) == 2
    assert runs[0]["prompt_name"] == "second"  # newest first


# --- test_results ---


def test_insert_and_get_test_results(conn):
    rid = _make_run(conn)
    results = [
        {"test_name": "test_foo[32x32]", "shape": "32x32", "status": "passed"},
        {
            "test_name": "test_foo[64x64]",
            "shape": "64x64",
            "status": "failed",
            "failure_category": "numerical",
            "failure_message": "allclose failed",
        },
    ]
    db.insert_test_results_batch(conn, rid, results)

    fetched = db.get_test_results(conn, rid)
    assert len(fetched) == 2
    assert fetched[0]["status"] == "passed"
    assert fetched[1]["failure_category"] == "numerical"


def test_empty_test_results(conn):
    rid = _make_run(conn)
    assert db.get_test_results(conn, rid) == []


# --- score_criteria ---


def test_insert_and_get_criteria(conn):
    rid = _make_run(conn)
    criteria = [
        {"name": "test_success", "raw_score": 80.0, "weight": 0.35, "weighted_score": 28.0},
        {"name": "helper_usage", "raw_score": 100.0, "weight": 0.13, "weighted_score": 13.0},
    ]
    db.insert_score_criteria(conn, rid, criteria)

    fetched = db.get_score_criteria(conn, rid)
    assert len(fetched) == 2
    assert fetched[0]["criterion"] == "test_success"
    assert fetched[1]["weighted_score"] == 13.0


# --- annotate ---


def test_annotate_run(conn):
    rid = _make_run(conn)
    db.annotate_run(conn, rid, score=4, notes="clean run")

    run = db.get_run(conn, rid)
    assert run["annotation_score"] == 4
    assert run["annotation_notes"] == "clean run"


def test_annotate_overwrite(conn):
    rid = _make_run(conn)
    db.annotate_run(conn, rid, score=2, notes="bad")
    db.annotate_run(conn, rid, score=5, notes="actually great")

    run = db.get_run(conn, rid)
    assert run["annotation_score"] == 5
    assert run["annotation_notes"] == "actually great"


# --- stats ---


def test_stats_empty(conn):
    stats = db.get_stats(conn)
    assert stats["total_runs"] == 0
    assert stats["avg_score"] == 0
    assert stats["pass_rate"] == 0


def test_stats_with_data(conn):
    rid1 = _make_run(conn, score_total=80.0, score_grade="B", golden_passed=10, golden_total=10)
    rid2 = _make_run(conn, score_total=60.0, score_grade="D", golden_passed=5, golden_total=10, run_number=2)

    # Add some test failures for failure_summary
    db.insert_test_results_batch(
        conn,
        rid2,
        [
            {"test_name": "t1", "status": "failed", "failure_category": "numerical"},
            {"test_name": "t2", "status": "failed", "failure_category": "numerical"},
            {"test_name": "t3", "status": "failed", "failure_category": "OOM"},
        ],
    )

    stats = db.get_stats(conn)
    assert stats["total_runs"] == 2
    assert stats["avg_score"] == 70.0
    assert stats["pass_rate"] == 50.0  # 1 of 2 runs had all golden passing
    assert stats["failure_summary"]["numerical"] == 2
    assert stats["failure_summary"]["OOM"] == 1


# --- failure_summary ---


def test_failure_summary_empty(conn):
    assert db.get_failure_summary(conn) == {}


def test_failure_summary_excludes_passed(conn):
    rid = _make_run(conn)
    db.insert_test_results_batch(
        conn,
        rid,
        [
            {"test_name": "t1", "status": "passed", "failure_category": None},
            {"test_name": "t2", "status": "failed", "failure_category": "hang"},
        ],
    )
    summary = db.get_failure_summary(conn)
    assert "hang" in summary
    assert summary["hang"] == 1
    # passed tests should not appear
    assert None not in summary
