# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for Seamless M4T v2 Large tests."""

from __future__ import annotations

import pytest

from models.experimental.seamless_m4t_v2_large.tests.pcc.token_matching_result_store import (
    clear_token_matching_results,
    get_token_matching_results,
    print_token_matching_summary,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "sanity: quick CI subset (ISL 32/64/128)")
    config.addinivalue_line("markers", "sweep: full ISL ladder 32→4096 (nightly)")


def pytest_sessionstart(session: pytest.Session) -> None:
    clear_token_matching_results()


def _write_summary_table(terminalreporter) -> None:
    results = get_token_matching_results()
    if not results:
        return

    tw = terminalreporter._tw
    tw.line("")
    terminalreporter.write_sep("=", "Seamless M4T v2 E2E token matching summary")
    header = f"{'Label':<18} {'Steps':>5}  {'Top1':>7}  {'Top5':>7}  {'Thresholds':>12}  {'Status':>6}"
    tw.line(header)
    tw.line("-" * len(header))

    for row in results:
        thresholds = f"{row.top1_threshold_pct:.0f}%/{row.top5_threshold_pct:.0f}%"
        status = "PASS" if row.passed else "FAIL"
        tw.line(
            f"{row.label:<18} {row.steps:>5}  {row.top1_pct:>6.2f}%  {row.top5_pct:>6.2f}%  "
            f"{thresholds:>12}  {status:>6}"
        )

    passed = sum(1 for r in results if r.passed)
    tw.line("-" * len(header))
    tw.line(f"Total: {passed}/{len(results)} passed")
    tw.line("")


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    _write_summary_table(terminalreporter)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fallback: ensure summary is visible even if terminal reporter hook is skipped."""
    if get_token_matching_results():
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            _write_summary_table(tr)
        else:
            print_token_matching_summary()
