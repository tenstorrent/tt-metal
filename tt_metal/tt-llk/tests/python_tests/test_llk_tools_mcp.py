# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "tools" / "mcp"))
import llk_tools


def test_the_perf_tests_are_the_perf_modules():
    tests = llk_tools.perf_tests()
    assert "perf_math_matmul" in tests
    assert all(t.startswith("perf_") for t in tests)


def test_the_command_passes_only_what_was_asked():
    cmd = llk_tools.compare_command(
        "blackhole", "perf_math_matmul", baseline="abc", dry_run=True
    )
    assert cmd[1:3] == ["blackhole", "perf_math_matmul"]
    assert cmd[cmd.index("--baseline") + 1] == "abc"
    assert "--current" not in cmd and "--refresh" not in cmd
    assert cmd[-1] == "--dry-run"


@pytest.mark.parametrize(
    "arch, test", [("quasar", "perf_matmul"), ("wormhole", "nope")]
)
def test_an_unknown_arch_or_test_is_refused(arch, test):
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        llk_tools.compare_command(arch, test)


def test_the_verdict_follows_the_report_and_the_exit_code(tmp_path):
    report = tmp_path / "regression_report.md"
    report.write_text("**❌ REGRESSIONS FOUND**")
    log = f"== comparing\n\nreport: {report}\n"
    assert llk_tools.compare_result(1, log)["verdict"] == "regressed"
    assert llk_tools.compare_result(0, log)["report"] == "**❌ REGRESSIONS FOUND**"
    assert llk_tools.compare_result(0, log)["verdict"] == "clean"
    assert llk_tools.compare_result(1, "error: no perf CSV")["verdict"] == "failed"
    assert (
        llk_tools.compare_result(0, "== dry run", dry_run=True)["verdict"] == "dry run"
    )
