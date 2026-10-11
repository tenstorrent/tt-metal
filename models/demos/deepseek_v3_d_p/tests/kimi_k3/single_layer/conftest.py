# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Non-skip guard for the single-layer suite's CI legs (plan §6).

Every test here is gated on a mesh topology mark and a box; a selector that matches nothing, or a
runner whose box fails the gate, skips the whole leg -- and a leg that skipped everything is green.
With ``K3_SINGLE_LAYER_EXPECT_PASSED=N`` the session fails unless at least N DEVICE tests (those
taking ``mesh_device``) actually PASSED; host-only tests never count toward it.
"""

import os

import pytest

_passed = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed and "mesh_device" in item.fixturenames:
        _passed.append(report.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    expected = os.environ.get("K3_SINGLE_LAYER_EXPECT_PASSED")
    if not expected:
        return
    if len(_passed) < int(expected):
        session.config.get_terminal_writer().line(
            f"K3_SINGLE_LAYER_EXPECT_PASSED={expected} but only {len(_passed)} device test(s) passed: "
            "the leg must not go green on skipped coverage",
            red=True,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
