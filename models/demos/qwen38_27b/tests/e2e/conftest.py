# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""pytest hooks for the e2e test (test_prefill_e2e.py in this directory); they touch its items only.

* A skip fails it. The workspace's mesh_device fixture (conftest.py) skips when the system mesh is smaller than the 8x4
  the test asks for, as fabric discovery makes it when links fail to train, and a session whose tests were all skipped
  exits 0, which scripts/run_safe_pytest.sh reports as PASS: the e2e verdict would be a pass with no check run. A skip
  or an error in setup also writes the test's report (status "error"), so a report an earlier run left is never taken
  for this run's.
* Without -k, -m or node ids, only the "all" case set runs: the other items' cases are part of it, and every item loads
  the model (the file as a whole would load it three times and run isl5120 and isl56320 twice each).
"""

from __future__ import annotations

from pathlib import Path

import pytest

E2E_TEST = "test_prefill_e2e"
FULL_SET = "all"
HERE = Path(__file__).resolve().parent


def _e2e(item) -> bool:
    return (
        getattr(item, "originalname", None) == E2E_TEST
        and hasattr(item, "callspec")
        and Path(item.path).resolve().parent == HERE
    )


def _case_set(item) -> str | None:
    return item.callspec.params.get("case_set")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when not in ("setup", "call") or rep.passed or not _e2e(item):
        return
    exc = call.excinfo
    if rep.skipped:
        what = f"skipped in {rep.when}: {getattr(exc.value, 'msg', None) if exc is not None else rep.longrepr}"
        rep.outcome = "failed"
        rep.longrepr = f"{item.nodeid}: {what}; the e2e test may not skip (no check ran)"
    elif rep.when == "setup":
        what = f"setup failed: {exc.exconly() if exc is not None else rep.longrepr}"
    else:
        return  # the test body failed: its own report says why
    if rep.when != "setup":
        return  # the test body wrote its report
    try:
        path = item.module.record_setup_problem(_case_set(item), what)
    except (OSError, AttributeError, TypeError, ValueError) as e:  # never break pytest's own reporting
        print(f"tests/e2e: the report of a setup problem was not written: {type(e).__name__}: {e}")
    else:
        print(f"tests/e2e: setup did not pass ({what}); report: {path}")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    mine = [it for it in items if _e2e(it)]
    if config.getoption("keyword") or config.getoption("markexpr") or any("::" in str(a) for a in config.args):
        return  # an explicit selection runs as selected
    sets = {_case_set(it) for it in mine}
    if not mine or FULL_SET not in sets or sets != set(mine[0].module.CASE_SETS):
        return
    full = set(mine[0].module.CASE_SETS[FULL_SET])
    drop = [it for it in mine if _case_set(it) != FULL_SET and set(it.module.CASE_SETS[_case_set(it)]) <= full]
    if not drop:
        return
    config.hook.pytest_deselected(items=drop)
    gone = {id(it) for it in drop}
    items[:] = [it for it in items if id(it) not in gone]
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        names = ", ".join(sorted(_case_set(it) for it in drop))
        reporter.write_line(f"tests/e2e: deselected the {names} items: the '{FULL_SET}' item runs their cases")
