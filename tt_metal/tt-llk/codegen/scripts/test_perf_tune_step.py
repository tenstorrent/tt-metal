# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Step 6c tuning helpers (quasar orchestrator_steps.sh): gate, attempt accounting, next."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

STEPS = Path(__file__).parent / "quasar" / "orchestrator_steps.sh"
STATE = Path(__file__).parent / "state.py"


@pytest.fixture
def run(tmp_path):
    log = tmp_path / "log"
    diag = log / "perf" / "diag"
    diag.mkdir(parents=True)
    wt = tmp_path / "wt"
    (wt / "tt_metal" / "tt-llk").mkdir(parents=True)

    def state(**values):
        for key, value in values.items():
            subprocess.run(
                [
                    sys.executable,
                    str(STATE),
                    "--log-dir",
                    str(log),
                    "set",
                    key,
                    str(value),
                ],
                check=True,
            )

    def get(key):
        return subprocess.run(
            [sys.executable, str(STATE), "--log-dir", str(log), "get", key],
            capture_output=True,
            text=True,
        ).stdout.strip()

    def sh(body):
        script = f"""
            source {STEPS}
            _LOG() {{ echo {log}; }}
            _wt()  {{ echo {wt}; }}
            rj()   {{ :; }}
            _L={log}
            {body}
        """
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""

    def diagnosis(headroom, status="ok", where=diag):
        where.mkdir(parents=True, exist_ok=True)
        (where / "diag.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "stall_headroom_pct": headroom,
                    "top_causes": [["math.bubble.SFPNONLINEAR->SFPSTORE", 32.0]],
                }
            )
        )
        state(
            PERF_DIAG_STATUS=status,
            PERF_DIAG_DIR=str(where),
            PERF_DIAG_LINE=f"PERF_DIAG status={status}",
        )

    state(PERF_ENABLED="true", KERNEL_TYPE="sfpu")
    diagnosis(13.97)
    return {"state": state, "get": get, "sh": sh, "diagnosis": diagnosis, "log": log}


# ── gate ─────────────────────────────────────────────────────────────────────


def test_gate_runs_when_headroom_is_large(run):
    line = run["sh"]("execute_step_perf_tune_gate")
    assert line.startswith("PERF_TUNE: run next=tune_1")
    assert "headroom=13.97%" in line
    assert "math.bubble.SFPNONLINEAR->SFPSTORE" in line
    assert run["get"]("PERF_TUNE_STATUS") == "run"
    assert run["get"]("PERF_TUNE_ATTEMPTS") == "0"
    assert run["get"]("PERF_TUNE_MAX_ATTEMPTS") == "2"


@pytest.mark.parametrize(
    "setup, reason",
    [
        (lambda r: r["diagnosis"](4.9), "below 5"),
        (lambda r: r["diagnosis"](13.0, status="not_reconciled"), "diagnosis"),
        (lambda r: r["state"](PERF_TUNE="false"), "PERF_TUNE=false"),
        (lambda r: r["state"](KERNEL_TYPE="math"), "sfpu"),
        (lambda r: r["state"](PERF_ENABLED="false"), "perf"),
    ],
)
def test_gate_skips(run, setup, reason):
    setup(run)
    line = run["sh"]("execute_step_perf_tune_gate")
    assert line.startswith("PERF_TUNE: skip") and reason in line
    assert run["get"]("PERF_TUNE_STATUS") == "skipped"


def test_gate_threshold_is_configurable(run):
    run["state"](PERF_TUNE_MIN_HEADROOM_PCT="15")
    assert run["sh"]("execute_step_perf_tune_gate").startswith("PERF_TUNE: skip")


# ── accounting ───────────────────────────────────────────────────────────────


def test_each_tune_label_counts_once(run):
    run["sh"]("execute_step_perf_tune_gate")
    run["sh"](
        "_perf_count_attempt tune_1; ss PERF_LAST_LABEL tune_1; _perf_count_attempt tune_1"
    )
    assert run["get"]("PERF_TUNE_ATTEMPTS") == "1"
    assert run["get"]("PERF_ATTEMPTS") in ("", "0"), "tuning has its own budget"


def test_optimizer_attempts_still_count_as_before(run):
    run["state"](PERF_ATTEMPTS="1")
    run["sh"]("_perf_count_attempt attempt_2")
    assert run["get"]("PERF_ATTEMPTS") == "2"


def test_keep_and_revert_record_the_tune_result(run):
    run["sh"]("execute_step_perf_tune_gate")
    run["state"](PERF_KEPT="1")
    run["sh"]("_perf_count_keep tune_1")
    assert run["get"]("PERF_TUNE_KEPT") == "1"
    assert run["get"]("PERF_KEPT") == "2"
    assert run["get"]("PERF_TUNE_LAST_RESULT") == "kept"
    run["sh"]("_perf_count_revert tune_2")
    assert run["get"]("PERF_TUNE_LAST_RESULT") == "reverted"


# ── next ─────────────────────────────────────────────────────────────────────


def _start(run):
    run["sh"]("execute_step_perf_tune_gate")


def test_next_after_a_kept_change_with_headroom_left(run):
    _start(run)
    run["state"](PERF_TUNE_ATTEMPTS="1", PERF_TUNE_LAST_RESULT="kept")
    run["diagnosis"](8.0, where=run["log"] / "perf" / "diag_tune_1")
    assert run["sh"]("execute_step_perf_tune_next").startswith("PERF_TUNE: next=tune_2")


def test_done_when_headroom_drops_below_threshold(run):
    _start(run)
    run["state"](PERF_TUNE_ATTEMPTS="1", PERF_TUNE_LAST_RESULT="kept")
    run["diagnosis"](4.4, where=run["log"] / "perf" / "diag_tune_1")
    line = run["sh"]("execute_step_perf_tune_next")
    assert line.startswith("PERF_TUNE: next=done") and "below 5" in line


def test_done_when_a_change_did_not_help(run):
    _start(run)
    run["state"](PERF_TUNE_ATTEMPTS="1", PERF_TUNE_LAST_RESULT="reverted")
    line = run["sh"]("execute_step_perf_tune_next")
    assert line.startswith("PERF_TUNE: next=done") and "did not help" in line


def test_done_when_attempts_are_used(run):
    _start(run)
    run["state"](PERF_TUNE_ATTEMPTS="2", PERF_TUNE_LAST_RESULT="kept")
    line = run["sh"]("execute_step_perf_tune_next")
    assert line.startswith("PERF_TUNE: next=done") and "attempts" in line


def test_done_when_tuning_never_started(run):
    run["diagnosis"](3.0)
    run["sh"]("execute_step_perf_tune_gate")
    assert run["sh"]("execute_step_perf_tune_next").startswith("PERF_TUNE: next=done")


# ── diagnosis labels and the report ──────────────────────────────────────────


def test_labelled_diagnosis_keeps_the_first_one(run, tmp_path):
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import sys\nout=sys.argv[sys.argv.index('--out-dir')+1]\n"
        "print('PERF_DIAG status=ok dir='+out)\n"
    )
    run["state"](
        QSR_SIM_BACKEND="emu",
        TARGET_ARCH="quasar",
        PERF_MODULE="m.py",
        PERF_TEST_ID="m.py::t",
        PERF_METRIC="mean(MATH_ISOLATE)",
        PERF_DIAG_STATUS="",
        PERF_DIAG_LINE="",
    )
    first = run["sh"](f"PERF_DIAGNOSE_PY={fake} execute_step_perf_diagnose")
    second = run["sh"](f"PERF_DIAGNOSE_PY={fake} execute_step_perf_diagnose tune_1")
    assert first.endswith(str(run["log"] / "perf" / "diag"))
    assert second.endswith(str(run["log"] / "perf" / "diag_tune_1"))
    assert run["get"]("PERF_DIAG_FIRST_LINE") == first
    assert run["get"]("PERF_DIAG_LINE") == second


def test_perf_json_reports_tuning(run):
    _start(run)
    run["state"](
        PERF_TUNE_ATTEMPTS="1", PERF_TUNE_KEPT="1", PERF_TUNE_LAST_RESULT="kept"
    )
    out = run["sh"]("_perf_json")
    tuning = json.loads(out)["perf"]["tuning"]
    assert tuning == {
        "status": "run",
        "attempts": 1,
        "kept": 1,
        "max_attempts": 2,
        "last_result": "kept",
        "min_headroom_pct": 5.0,
    }
