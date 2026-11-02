# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import csv
import os
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "tools" / "mcp"))
import llk_tools


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_COMPARE_HOME", str(tmp_path))
    out = tmp_path / "reports" / "blackhole" / "perf_math_matmul"
    out.mkdir(parents=True)
    return out


def test_the_perf_tests_are_the_measurable_perf_modules():
    assert "perf_math_matmul" in llk_tools.perf_tests()
    assert "perf_fused" not in llk_tools.perf_tests()
    assert "perf_fast_untilize" in llk_tools.perf_tests("blackhole")
    assert "perf_fast_untilize" not in llk_tools.perf_tests("wormhole")


def test_the_command_passes_only_what_was_asked():
    cmd = llk_tools.compare_command(
        "blackhole", "perf_math_matmul", baseline="origin/main~2", dry_run=True
    )
    assert cmd[1:] == [
        "blackhole",
        "perf_math_matmul",
        "--baseline",
        "origin/main~2",
        "--dry-run",
    ]


def test_the_command_forwards_the_gate_knobs():
    cmd = llk_tools.compare_command(
        "wormhole",
        "perf_math_matmul",
        threshold=0.13,
        module_threshold="perf_math_matmul=0.25",
        run_types="L1_TO_L1,MATH_ISOLATE",
        code_shift=24,
    )
    for flag, value in (
        ("--threshold", "0.13"),
        ("--module-threshold", "perf_math_matmul=0.25"),
        ("--run-types", "L1_TO_L1,MATH_ISOLATE"),
        ("--code-shift", "24"),
    ):
        assert cmd[cmd.index(flag) + 1] == value


@pytest.mark.parametrize(
    "arch, test, options",
    [
        ("quasar", "perf_matmul", {}),
        ("wormhole", "nope", {}),
        ("wormhole", "perf_fast_untilize", {}),
        ("blackhole", "perf_matmul", {"baseline": "--refresh"}),
        ("blackhole", "perf_matmul", {"current": "-x"}),
        ("blackhole", "perf_matmul", {"threshold": 5}),
        ("blackhole", "perf_matmul", {"threshold": 0}),
        ("blackhole", "perf_matmul", {"iterations": 0}),
        ("blackhole", "perf_matmul", {"code_shift": -4}),
        ("blackhole", "perf_matmul", {"run_types": "L1_TO_L1 --refresh"}),
        ("blackhole", "perf_matmul", {"module_threshold": "--dry-run"}),
    ],
)
def test_a_bad_argument_is_refused(arch, test, options):
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        llk_tools.compare_command(arch, test, **options)


def test_the_verdict_follows_the_report_and_the_exit_code(reports):
    report = reports / "regression_report.md"
    report.write_text("**❌ REGRESSIONS FOUND**")
    log = f"== comparing\n\nreport: {report}\n"
    assert llk_tools.compare_result(1, log)["verdict"] == "regressed"
    assert llk_tools.compare_result(0, log)["report"] == "**❌ REGRESSIONS FOUND**"
    assert llk_tools.compare_result(0, log)["verdict"] == "clean"
    assert llk_tools.compare_result(2, log)["verdict"] == "failed"
    assert llk_tools.compare_result(1, "error: no perf CSV")["verdict"] == "failed"
    assert (
        llk_tools.compare_result(0, "== dry run", dry_run=True)["verdict"] == "dry run"
    )


def test_a_report_outside_the_reports_directory_is_not_read(reports, tmp_path):
    outside = tmp_path / "secret.md"
    outside.write_text("not a report")
    result = llk_tools.compare_result(0, f"report: {outside}\n")
    assert result["report"] is None and result["verdict"] == "failed"


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(marker, delta):
    return {
        "marker": marker,
        "run_type": "L1_TO_L1",
        "current": 100 + delta,
        "baseline": 100,
        "delta_pct": delta,
        "delta_cycles": delta,
        "test_module": "perf_math_matmul",
        "tile_cnt": "",
    }


def test_placement_outcomes_follow_the_five_positions(reports):
    runs = {}
    for shift in llk_tools.POSITIONS:
        report = reports / f"p{shift}" / "regression_report.md"
        report.parent.mkdir()
        report.write_text("")
        moved = [_row("TILE_LOOP", 40.0)] + (
            [_row("INIT", 50.0)] if shift in (0, 24) else []
        )
        _write(report.with_suffix(".regressions.csv"), moved)
        _write(report.with_suffix(".points.csv"), moved + [_row("KERNEL", 1.0)])
        runs[shift] = {"verdict": "regressed", "report_path": str(report)}

    outcomes = {o["point"]["marker"]: o for o in llk_tools.placement_outcomes(runs)}
    assert outcomes["TILE_LOOP"]["outcome"] == "regression"
    assert outcomes["INIT"]["outcome"] == "placement"
    assert outcomes["INIT"]["delta_pct"]["p24"] == 50.0
    assert outcomes["INIT"]["delta_pct"]["p4"] is None
    assert "KERNEL" not in outcomes

    runs[44] = {"verdict": "failed", "report_path": None}
    assert {o["outcome"] for o in llk_tools.placement_outcomes(runs)} == {
        "not re-measured"
    }


def test_the_child_runs_in_its_own_session_with_this_python_first():
    probe = "import os; print(os.environ['PATH']); print(os.getsid(0) == os.getpid())"
    code, out = asyncio.run(llk_tools.run([sys.executable, "-c", probe]))
    path, leader = out.splitlines()
    assert code == 0 and leader == "True"
    assert path.split(os.pathsep)[0] == str(pathlib.Path(sys.executable).parent)


def test_a_cancelled_call_stops_the_whole_sweep(tmp_path):
    pidfile = tmp_path / "pid"
    sweep = ["sh", "-c", f"sleep 30 & echo $! > {pidfile}; wait"]

    async def cancel():
        with pytest.raises(
            asyncio.TimeoutError
        ):  # allow-pytest.raises: no expect_error in LLK suite
            await asyncio.wait_for(llk_tools.run(sweep), 1)

    asyncio.run(cancel())
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"the sweep's child {pid} still runs")
