"""Tests for eval.ingest — uses in-memory SQLite + temp files."""

import json
from pathlib import Path

import pytest

from eval import db
from eval.ingest import ingest_run


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_eval.db"


def test_ingest_minimal(db_path):
    """Ingest with no test results or score — just metadata."""
    rid = ingest_run(
        db_path=db_path,
        prompt_name="softmax",
        run_number=1,
        starting_branch="main",
        starting_commit="abc123",
        created_branch="run1_softmax",
    )
    conn = db.connect(db_path)
    run = db.get_run(conn, rid)
    conn.close()

    assert run["prompt_name"] == "softmax"
    assert run["starting_commit"] == "abc123"
    assert run["golden_passed"] is None
    assert run["score_total"] is None


def test_ingest_with_test_results(db_path, tmp_path):
    test_results = [
        {"test_name": "test_a[32x32]", "shape": "32x32", "status": "passed"},
        {
            "test_name": "test_a[64x64]",
            "shape": "64x64",
            "status": "failed",
            "failure_category": "numerical",
            "failure_message": "allclose failed",
        },
        {
            "test_name": "test_a[big]",
            "shape": "big",
            "status": "skipped",
            "failure_category": "hang",
            "failure_message": "hung",
        },
    ]
    results_path = tmp_path / "test_results.json"
    results_path.write_text(json.dumps(test_results))

    rid = ingest_run(
        db_path=db_path,
        prompt_name="ln",
        run_number=1,
        starting_branch="main",
        starting_commit="def456",
        created_branch="run1_ln",
        test_results_path=results_path,
    )

    conn = db.connect(db_path)
    run = db.get_run(conn, rid)
    tests = db.get_test_results(conn, rid)
    conn.close()

    # golden_total excludes skipped
    assert run["golden_total"] == 2
    assert run["golden_passed"] == 1
    assert len(tests) == 3
    assert tests[1]["failure_category"] == "numerical"


def test_ingest_with_score(db_path, tmp_path):
    score_data = {
        "total_score": 82.5,
        "grade": "B",
        "criteria": [
            {"name": "test_success", "raw_score": 90.0, "weight": 0.35, "weighted_score": 31.5},
            {"name": "helper_usage", "raw_score": 100.0, "weight": 0.13, "weighted_score": 13.0},
        ],
    }
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(score_data))

    rid = ingest_run(
        db_path=db_path,
        prompt_name="reduce",
        run_number=1,
        starting_branch="main",
        starting_commit="xyz",
        created_branch="run1_reduce",
        score_json_path=score_path,
    )

    conn = db.connect(db_path)
    run = db.get_run(conn, rid)
    criteria = db.get_score_criteria(conn, rid)
    conn.close()

    assert run["score_total"] == 82.5
    assert run["score_grade"] == "B"
    assert len(criteria) == 2
    assert criteria[0]["criterion"] == "test_success"


def test_ingest_missing_files(db_path):
    """Should handle missing file paths gracefully."""
    rid = ingest_run(
        db_path=db_path,
        prompt_name="test",
        run_number=1,
        starting_branch="main",
        starting_commit="aaa",
        created_branch="run1",
        test_results_path=Path("/nonexistent/results.json"),
        score_json_path=Path("/nonexistent/score.json"),
    )

    conn = db.connect(db_path)
    run = db.get_run(conn, rid)
    conn.close()

    assert run is not None
    assert run["golden_passed"] is None
    assert run["score_total"] is None


def test_ingest_multiple_runs(db_path):
    """Multiple runs should coexist."""
    for i in range(3):
        ingest_run(
            db_path=db_path,
            prompt_name="ln",
            run_number=i + 1,
            starting_branch="main",
            starting_commit=f"commit_{i}",
            created_branch=f"run{i}_ln",
        )

    conn = db.connect(db_path)
    runs = db.get_all_runs(conn)
    conn.close()

    assert len(runs) == 3


def test_ingest_with_kernels(db_path, tmp_path):
    """Ingest should collect C++ kernels, program descriptor, and entry point."""
    clone = tmp_path / "clone"
    op_dir = clone / "ttnn" / "ttnn" / "operations" / "my_op"
    kernel_dir = op_dir / "kernels"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "reader.cpp").write_text('#include "dataflow_api.h"\nvoid kernel_main() {}')
    (kernel_dir / "compute.cpp").write_text("namespace NAMESPACE {\nvoid MAIN {}\n}")
    (kernel_dir / "empty.cpp").write_text("")  # empty should be skipped
    # Program descriptor and entry point
    (op_dir / "my_op_program_descriptor.py").write_text("def create_program_descriptor(): pass")
    (op_dir / "my_op.py").write_text("def my_op(input): pass")

    rid = ingest_run(
        db_path=db_path,
        prompt_name="my_op",
        run_number=1,
        starting_branch="main",
        starting_commit="abc",
        created_branch="run1",
        clone_dir=clone,
        op_name="my_op",
    )

    conn = db.connect(db_path)
    kernels = db.get_kernels(conn, rid)
    conn.close()

    assert len(kernels) == 4  # 2 cpp + descriptor + entry point
    filenames = {k["filename"] for k in kernels}
    assert "reader.cpp" in filenames
    assert "compute.cpp" in filenames
    assert "my_op_program_descriptor.py" in filenames
    assert "my_op.py" in filenames


def test_ingest_with_self_reflection(db_path, tmp_path):
    """Ingest should collect self_reflection.md from the clone."""
    clone = tmp_path / "clone"
    op_dir = clone / "ttnn" / "ttnn" / "operations" / "my_op"
    op_dir.mkdir(parents=True)
    (op_dir / "self_reflection.md").write_text("## Summary\nAll stages passed.\n## Issues\nNone.")

    rid = ingest_run(
        db_path=db_path,
        prompt_name="my_op",
        run_number=1,
        starting_branch="main",
        starting_commit="abc",
        created_branch="run1",
        clone_dir=clone,
        op_name="my_op",
    )

    conn = db.connect(db_path)
    artifacts = db.get_artifacts(conn, rid)
    conn.close()

    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "self_reflection"
    assert "All stages passed" in artifacts[0]["content"]


def test_ingest_no_clone_dir(db_path):
    """Without clone_dir, no kernels or artifacts should be collected."""
    rid = ingest_run(
        db_path=db_path,
        prompt_name="test",
        run_number=1,
        starting_branch="main",
        starting_commit="abc",
        created_branch="run1",
        clone_dir=None,
        op_name="test",
    )

    conn = db.connect(db_path)
    assert db.get_kernels(conn, rid) == []
    assert db.get_artifacts(conn, rid) == []
    conn.close()
