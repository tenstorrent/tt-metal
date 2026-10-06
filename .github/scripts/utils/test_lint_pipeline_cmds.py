#!/usr/bin/env python3
"""Tests for lint_pipeline_cmds.py -- cmd blocks must stop at the first failed test command."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lint_pipeline_cmds as lint  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "lint_pipeline_cmds.py"


def _entry(cmd: str, name: str = "leg") -> dict:
    return {"name": name, "cmd": textwrap.dedent(cmd)}


def _findings(cmd: str) -> list[lint.Finding]:
    return lint.lint_entry(_entry(cmd))


def test_flags_assignment_after_failed_pytest():
    findings = _findings(
        """\
        fail=0
        pytest tests/a.py || fail=1
        pytest tests/b.py || fail=1
        exit $fail
        """
    )
    # Only the first recording matters: after it another test runs on un-reset devices.
    assert [f.line for f in findings] == [2]
    assert "fail=1" in findings[0].message


def test_allows_capturing_the_last_tests_exit_code_for_post_processing():
    """The tt-train idiom: one test command, keep its status, publish a summary, exit with it."""
    findings = _findings(
        """\
        rc=0
        mpirun -np 1 bash -lc "python tt-train/scripts/run_models.py --summary-file $SUMMARY" || rc=$?
        if [ -s "$SUMMARY" ]; then cat "$SUMMARY" >> "$GITHUB_STEP_SUMMARY"; fi
        exit $rc
        """
    )
    assert findings == []


def test_flags_recording_brace_group_followed_by_another_test():
    findings = _findings(
        """\
        pytest tests/a.py || { rc=$?; echo "a failed"; }
        pytest tests/b.py
        exit ${rc:-0}
        """
    )
    assert [f.line for f in findings] == [1]


def test_allows_brace_group_that_exits_even_with_braces_in_its_body():
    findings = _findings(
        """\
        test -f "${TTRUN_PY}" || { echo "ERROR: missing ${TTRUN_PY}"; exit 1; }
        test -f "${MGD}" || { echo "ERROR: missing ${MGD}"; exit 1; }
        python3 -m pytest tests/a.py
        """
    )
    assert findings == []


def test_or_true_on_a_non_test_command_after_a_test_on_the_same_line_is_fine():
    findings = _findings(
        """\
        pytest tests/a.py && rm -rf generated/tmp || true
        pytest tests/b.py; rm -f stale.lock || true
        """
    )
    assert [f.line for f in findings] == [1]


def test_flags_assignment_inside_run_helper():
    findings = _findings(
        """\
        rc=0
        run() { mpirun -np 1 python3 -m pytest "$@" || rc=1; }
        run tests/a.py
        exit $rc
        """
    )
    assert [f.line for f in findings] == [2]


def test_flags_exit_code_capture():
    findings = _findings(
        """\
        exit_code=0
        pytest tests/a.py || exit_code=$?
        pytest tests/b.py || exit_code=$?
        exit $exit_code
        """
    )
    assert [f.line for f in findings] == [2]


def test_flags_or_true_after_test_runner():
    findings = _findings("pytest tests/a.py || true\n")
    assert len(findings) == 1
    assert "|| true" in findings[0].message


def test_allows_or_true_on_non_test_command():
    findings = _findings(
        """\
        mpirun --pernode bash -lc 'mkdir -p ~/.cache && ln -sfn /mnt/x ~/.cache/x' || true
        pytest tests/a.py
        """
    )
    assert findings == []


def test_allows_exit_on_failure():
    findings = _findings(
        """\
        LORA_PATH=$(python3 -m fetch --dir "$D") || exit 1
        test -f "$MGD" || { echo "ERROR: missing $MGD"; exit 1; }
        pytest tests/a.py -k "fast" || exit $?
        pytest tests/b.py
        """
    )
    assert findings == []


def test_flags_set_plus_e():
    findings = _findings(
        """\
        set +e
        pytest tests/a.py
        """
    )
    assert len(findings) == 1
    assert "set +e" in findings[0].message


def test_reports_first_line_of_a_continued_command():
    findings = _findings(
        """\
        export X=1
        pytest --timeout 2400 tests/a.py \\
          -k "bh_qb and 1024x1024" || status=1
        pytest tests/b.py || status=1
        exit $status
        """
    )
    # The continued command is reported at its first physical line; the last test's
    # recording is not a finding because nothing runs on the device after it.
    assert [f.line for f in findings] == [2]


def test_ignores_comment_lines():
    findings = _findings(
        """\
        # Run all, accumulate failures: || fail=1
        pytest tests/a.py
        """
    )
    assert findings == []


def _write_yaml(tmp_path: Path, name: str, entries: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(entries))
    return path


def test_baseline_suppresses_known_entry(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        "legacy_tests.yaml",
        """\
        - name: legacy leg
          cmd: |
            pytest tests/a.py || fail=1
            pytest tests/b.py || fail=1
            exit $fail
        """,
    )
    baseline = {"legacy_tests.yaml": ["legacy leg"]}
    problems = lint.lint_files([yaml_path], baseline)
    assert problems == []


def test_stale_baseline_entry_is_reported(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        "legacy_tests.yaml",
        """\
        - name: fixed leg
          cmd: |
            pytest tests/a.py || exit $?
        """,
    )
    baseline = {"legacy_tests.yaml": ["fixed leg"]}
    problems = lint.lint_files([yaml_path], baseline)
    assert len(problems) == 1
    assert "baseline" in problems[0].lower()
    assert "fixed leg" in problems[0]


CONTINUE_LEG = """\
- name: continuing leg
  cmd: |
    export CI_CONTINUE_ON_TEST_FAILURE=1
    pytest tests/a.py || fail=1
    pytest tests/b.py || fail=1
    exit $fail
"""


def test_continue_marker_skips_the_leg_in_the_blaze_prefill_yaml(tmp_path):
    yaml_path = _write_yaml(tmp_path, "blaze_models_prefill_tests.yaml", CONTINUE_LEG)
    assert lint.lint_files([yaml_path], {}) == []


def test_continue_marker_is_rejected_outside_the_allowed_yamls(tmp_path):
    yaml_path = _write_yaml(tmp_path, "models_unit_tests.yaml", CONTINUE_LEG)
    problems = lint.lint_files([yaml_path], {})
    assert len(problems) == 1
    assert "CI_CONTINUE_ON_TEST_FAILURE is not allowed" in problems[0]


def test_baseline_growth_is_rejected_against_the_base_baseline():
    base = {"legacy_tests.yaml": ["old leg"]}
    grown = {"legacy_tests.yaml": ["old leg", "new leg"], "other_tests.yaml": ["another"]}
    problems = lint.baseline_growth(grown, base)
    assert len(problems) == 2
    assert any("new leg" in p for p in problems) and any("another" in p for p in problems)
    assert lint.baseline_growth(base, grown) == []


def test_cli_skips_the_shrink_check_when_the_base_baseline_is_missing(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        "legacy_tests.yaml",
        """\
        - name: legacy leg
          cmd: |
            pytest tests/a.py || fail=1
            pytest tests/b.py || fail=1
            exit $fail
        """,
    )
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text("legacy_tests.yaml:\n  - legacy leg\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline_path),
            "--base-baseline",
            str(tmp_path / "missing.yaml"),
            str(yaml_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipping the shrink-only check" in result.stdout


def test_unbaselined_finding_is_reported_with_location(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        "models_unit_tests.yaml",
        """\
        - name: clean leg
          cmd: |
            pytest tests/a.py
        - name: leaky leg
          cmd: |
            status=0
            pytest tests/b.py || status=1
            pytest tests/c.py || status=1
            exit $status
        """,
    )
    problems = lint.lint_files([yaml_path], {})
    assert len(problems) == 1
    assert "models_unit_tests.yaml" in problems[0]
    assert "leaky leg" in problems[0]
    assert "status=1" in problems[0]


def test_cli_exits_nonzero_on_finding(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        "models_e2e_tests.yaml",
        """\
        - name: leaky leg
          cmd: |
            pytest tests/b.py || rc=1
            pytest tests/c.py || rc=1
            exit $rc
        """,
    )
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text("{}\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline", str(baseline_path), str(yaml_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "leaky leg" in result.stdout + result.stderr


def test_cli_exits_zero_on_clean_file(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        "models_e2e_tests.yaml",
        """\
        - name: clean leg
          cmd: |
            pytest tests/a.py || exit $?
            pytest tests/b.py
        """,
    )
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text("{}\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline", str(baseline_path), str(yaml_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
