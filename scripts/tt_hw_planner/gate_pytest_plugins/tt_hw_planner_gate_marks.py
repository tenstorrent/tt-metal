# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The e2e gate's per-test record: which tests have PASSED, said as each one finishes.

WHY. The gate runs its tests as one pytest session, and a session ended from outside -- the thermal
abort at 95C, a wedge reset -- leaves no summary, so the retry knew nothing and ran every test again
from the first. Measured 2026-10-12 on a 4-chip p300c: Devstral's four e2e tests take ~12 min back to
back and the board reaches the abort limit at ~11-12 min even from 65C, so every retry was ended near
its end, twelve times; the same four tests passed when run one at a time.

So this plugin prints a mark when a test starts and when it has passed every phase, flushed as it
happens (a killed session keeps everything it printed), and on a later session leaves out exactly the
tests the gate says were already earned. Loaded by the gate with `-p`, from a directory that holds only
plugins, so putting it on the path cannot shadow anything else.

MUST ALWAYS IMPORT: standard library and pytest only.
"""

from __future__ import annotations

import os
import sys

import pytest

EARNED_ENV = "TT_HW_PLANNER_GATE_EARNED"  # a file of node ids, one per line, already passed this gate
START_MARK, PASSED_MARK = "GATE_TEST_START=", "GATE_TEST_PASSED="

_FAILED: set = set()
_CALLED: set = set()


def _say(line: str) -> None:
    print(line, file=sys.stdout, flush=True)


def _earned() -> set:
    path = os.environ.get(EARNED_ENV, "")
    if not path:
        return set()
    try:
        with open(path) as fh:
            return {ln.strip() for ln in fh if ln.strip()}
    except OSError:
        return set()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    """Leave out the tests already earned -- by EXACT node id (--deselect matches prefixes, so
    `test_a` would also drop `test_ab`)."""
    earned = _earned()
    if not earned:
        return
    keep = [it for it in items if it.nodeid not in earned]
    gone = [it for it in items if it.nodeid in earned]
    if gone:
        config.hook.pytest_deselected(items=gone)
        items[:] = keep


def pytest_runtest_logstart(nodeid, location):
    _say(START_MARK + nodeid)


def pytest_runtest_logreport(report):
    """PASSED only when setup, call and teardown all passed -- not a skip, not an xfail or xpass."""
    nid = report.nodeid
    if report.failed or report.skipped or hasattr(report, "wasxfail"):
        _FAILED.add(nid)
    if report.when == "call" and report.passed:
        _CALLED.add(nid)
    if report.when == "teardown" and nid in _CALLED and nid not in _FAILED:
        _say(PASSED_MARK + nid)
