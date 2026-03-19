"""Tests for eval.dashboard — generates HTML from in-memory DB."""

from pathlib import Path

import pytest

from eval import db
from eval.dashboard import generate_html


@pytest.fixture
def conn():
    c = db.connect(Path(":memory:"))
    yield c
    c.close()


def _make_run(conn, **overrides):
    defaults = dict(
        timestamp="2026-03-09T14:30:00",
        prompt_name="layer_norm_rm",
        run_number=1,
        starting_branch="mare/eval",
        starting_commit="abc123",
        created_branch="run1_ln",
    )
    defaults.update(overrides)
    rid = db.insert_run(conn, **defaults)
    conn.commit()
    return rid


def test_empty_db(conn):
    html = generate_html(conn)
    assert "<!DOCTYPE html>" in html
    assert "No runs recorded" in html


def test_basic_structure(conn):
    _make_run(conn)
    html = generate_html(conn)
    assert "Eval Dashboard" in html
    assert "Total Runs" in html
    assert "layer_norm_rm" in html


def test_grade_displayed(conn):
    _make_run(conn, score_total=85.0, score_grade="B")
    html = generate_html(conn)
    assert "85.0" in html
    assert "(B)" in html


def test_golden_counts(conn):
    _make_run(conn, golden_passed=89, golden_total=105)
    html = generate_html(conn)
    assert "89/105" in html


def test_annotation_stars(conn):
    rid = _make_run(conn)
    db.annotate_run(conn, rid, score=4, notes="good")
    html = generate_html(conn)
    # 4 filled stars + 1 empty
    assert "&#9733;" in html


def test_failure_breakdown_bar(conn):
    rid = _make_run(conn)
    db.insert_test_results_batch(
        conn,
        rid,
        [
            {"test_name": "t1", "status": "failed", "failure_category": "numerical"},
            {"test_name": "t2", "status": "failed", "failure_category": "numerical"},
            {"test_name": "t3", "status": "failed", "failure_category": "OOM"},
        ],
    )
    conn.commit()
    html = generate_html(conn)
    assert "Failure Breakdown" in html
    assert "numerical" in html
    assert "OOM" in html


def test_detail_section_has_tests(conn):
    rid = _make_run(conn)
    db.insert_test_results_batch(
        conn,
        rid,
        [
            {"test_name": "test_a[32x32]", "shape": "32x32", "status": "passed"},
            {
                "test_name": "test_a[64x64]",
                "shape": "64x64",
                "status": "failed",
                "failure_category": "numerical",
                "failure_message": "allclose fail",
            },
        ],
    )
    conn.commit()
    html = generate_html(conn)
    assert "test_a[32x32]" in html
    assert "test_a[64x64]" in html
    assert "PASSED" in html
    assert "FAILED" in html


def test_criteria_in_detail(conn):
    rid = _make_run(conn)
    db.insert_score_criteria(
        conn,
        rid,
        [
            {"name": "test_success", "raw_score": 80.0, "weight": 0.35, "weighted_score": 28.0},
        ],
    )
    conn.commit()
    html = generate_html(conn)
    assert "Test Success" in html
    assert "28.0" in html


def test_multiple_runs(conn):
    _make_run(conn, prompt_name="softmax", run_number=1)
    _make_run(conn, prompt_name="reduce", run_number=1, timestamp="2026-03-10T10:00:00")
    html = generate_html(conn)
    assert "softmax" in html
    assert "reduce" in html


def test_html_escaping(conn):
    """Ensure special characters are escaped."""
    _make_run(conn, prompt_name="test<script>alert(1)</script>")
    html = generate_html(conn)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_valid_html(conn):
    _make_run(conn)
    html = generate_html(conn)
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_kernels_in_detail(conn):
    rid = _make_run(conn)
    db.insert_kernels(
        conn,
        rid,
        [
            {"filename": "reader.cpp", "source_code": '#include "dataflow_api.h"\nvoid kernel_main() {}'},
            {"filename": "compute.cpp", "source_code": "namespace NAMESPACE {\nvoid MAIN {}\n}"},
        ],
    )
    conn.commit()
    html = generate_html(conn)
    assert "reader.cpp" in html
    assert "compute.cpp" in html
    assert "kernel_main" in html
    assert "Kernels (2)" in html
    assert "language-cpp" in html


def test_self_reflection_in_detail(conn):
    rid = _make_run(conn)
    db.insert_artifact(conn, rid, "self_reflection", "## Summary\nAll stages passed.")
    conn.commit()
    html = generate_html(conn)
    assert "Self-Reflection" in html
    assert "All stages passed" in html


def test_kernel_code_escaped(conn):
    """Kernel source with HTML-like content should be escaped."""
    rid = _make_run(conn)
    db.insert_kernels(
        conn,
        rid,
        [{"filename": "test.cpp", "source_code": "int x = a<b && c>d; // template<T>"}],
    )
    conn.commit()
    html = generate_html(conn)
    # The < and > should be escaped in the HTML
    assert "a&lt;b" in html
    assert "c&gt;d" in html


def test_host_code_in_detail(conn):
    rid = _make_run(conn)
    db.insert_host_code(
        conn,
        rid,
        [
            {"filename": "my_op.py", "source_code": "def my_op(input): pass"},
            {"filename": "my_op_program_descriptor.py", "source_code": "def create_pd(): pass"},
        ],
    )
    conn.commit()
    html = generate_html(conn)
    assert "Host-Side (2)" in html
    assert "my_op.py" in html
    assert "my_op_program_descriptor.py" in html
    assert "language-python" in html


def test_section_tabs_present(conn):
    """When kernels, host code, and artifacts exist, section tabs should appear."""
    rid = _make_run(conn)
    db.insert_kernels(conn, rid, [{"filename": "k.cpp", "source_code": "code"}])
    db.insert_host_code(conn, rid, [{"filename": "op.py", "source_code": "code"}])
    db.insert_artifact(conn, rid, "self_reflection", "content")
    conn.commit()
    html = generate_html(conn)
    assert "showSection" in html
    assert "Kernels (1)" in html
    assert "Host-Side (1)" in html
    assert "Self-Reflection" in html
