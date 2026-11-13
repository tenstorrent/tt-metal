# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""execute_step_perf_diagnose (quasar orchestrator_steps.sh): gating, arguments, recorded state."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

STEPS = Path(__file__).parent / "quasar" / "orchestrator_steps.sh"
STATE = Path(__file__).parent / "state.py"

FAKE = """#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
Path(sys.argv[0]).with_suffix(".argv.json").write_text(json.dumps(sys.argv[1:]))
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
print("noise before the result")
print("PERF_DIAG status=ok run_type=X measured=10.0 floor=9.0 stall_headroom=10.0% "
      "reconcile=+0.00% top=math.idle:1.0 dir=/d")
"""


@pytest.fixture
def run(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    wt = tmp_path / "wt"
    (wt / "tt_metal" / "tt-llk").mkdir(parents=True)
    fake = tmp_path / "fake_diag.py"
    fake.write_text(FAKE)

    def state(**values):
        for key, value in values.items():
            subprocess.run(
                [sys.executable, str(STATE), "--log-dir", str(log), "set", key, value],
                check=True,
            )

    def call(step="execute_step_perf_diagnose", sleep="0", wait="30"):
        script = f"""
            source {STEPS}
            _LOG() {{ echo {log}; }}
            _wt()  {{ echo {wt}; }}
            rj()   {{ :; }}
            FAKE_SLEEP={sleep} PERF_DIAG_WAIT_SECS={wait} PERF_DIAGNOSE_PY={fake} {step}
        """
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        return out.stdout.strip().splitlines()[-1]

    def get(key):
        return subprocess.run(
            [sys.executable, str(STATE), "--log-dir", str(log), "get", key],
            capture_output=True,
            text=True,
        ).stdout.strip()

    argv_file = fake.with_suffix(".argv.json")
    state(
        PERF_ENABLED="true",
        QSR_SIM_BACKEND="emu",
        TARGET_ARCH="quasar",
        PERF_MODULE="perf_eltwise_unary_sfpu_quasar.py",
        PERF_TEST_ID="perf_eltwise_unary_sfpu_quasar.py::t[Sigmoid]",
        PERF_METRIC="mean(MATH_ISOLATE)",
    )
    return {
        "state": state,
        "call": call,
        "get": get,
        "argv": argv_file,
        "log": log,
        "wt": wt,
    }


def test_runs_one_measure_only_capture_and_records_it(run):
    line = run["call"]()
    assert line.startswith("PERF_DIAG status=ok")
    argv = json.loads(run["argv"].read_text())
    assert argv[argv.index("--run-type") + 1] == "MATH_ISOLATE"
    assert (
        argv[argv.index("--test-id") + 1]
        == "perf_eltwise_unary_sfpu_quasar.py::t[Sigmoid]"
    )
    assert argv[argv.index("--llk") + 1] == str(run["wt"] / "tt_metal" / "tt-llk")
    assert argv[argv.index("--out-dir") + 1] == str(run["log"] / "perf" / "diag")
    assert run["get"]("PERF_DIAG_STATUS") == "ok"
    assert run["get"]("PERF_DIAG_LINE") == line


def test_non_math_metric_diagnoses_l1_to_l1(run):
    run["state"](PERF_METRIC="mean(L1_TO_L1)")
    run["call"]()
    argv = json.loads(run["argv"].read_text())
    assert argv[argv.index("--run-type") + 1] == "L1_TO_L1"


@pytest.mark.parametrize(
    "values, reason",
    [
        ({"PERF_ENABLED": "false"}, "perf not enabled"),
        ({"PERF_DIAG": "false"}, "PERF_DIAG=false"),
        ({"QSR_SIM_BACKEND": "vcs"}, "emulator only"),
        ({"PERF_TEST_ID": ""}, "no perf variant"),
    ],
)
def test_skips_without_capturing(run, values, reason):
    run["state"](**values)
    line = run["call"]()
    assert line.startswith("PERF_DIAG status=skipped") and reason in line
    assert not run["argv"].exists()
    assert run["get"]("PERF_DIAG_STATUS") == "skipped"


def test_perf_json_carries_the_diagnosis(run):
    run["call"]()
    script = f"""
        source {STEPS}
        _L={run['log']}
        _perf_json
    """
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout
    diag = json.loads(out)["perf"]["diagnosis"]
    assert diag["status"] == "ok"
    assert diag["line"].startswith("PERF_DIAG status=ok")


def test_long_capture_returns_running_then_wait_collects_it(run):
    first = run["call"](sleep="4", wait="1")
    assert first.startswith("PERF_DIAG status=running")
    assert "execute_step_perf_diagnose_wait" in first
    assert run["get"]("PERF_DIAG_STATUS") == "running"
    done = run["call"]("execute_step_perf_diagnose_wait", wait="30")
    assert done.startswith("PERF_DIAG status=ok")
    assert run["get"]("PERF_DIAG_STATUS") == "ok"


def test_wait_after_completion_repeats_the_recorded_line(run):
    line = run["call"]()
    assert run["call"]("execute_step_perf_diagnose_wait") == line


def test_a_dead_capture_is_reported_failed(run, tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("import sys; sys.exit(3)\n")
    script = f"""
        source {STEPS}
        _LOG() {{ echo {run['log']}; }}
        _wt()  {{ echo {run['wt']}; }}
        rj()   {{ :; }}
        PERF_DIAG_WAIT_SECS=10 PERF_DIAGNOSE_PY={broken} execute_step_perf_diagnose
    """
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1].startswith("PERF_DIAG status=failed")
    assert run["get"]("PERF_DIAG_STATUS") == "failed"
