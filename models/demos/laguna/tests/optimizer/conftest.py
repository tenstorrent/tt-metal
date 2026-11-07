# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exit-hang guard for the optimizer gates.

A full 48-layer Laguna process spends minutes in interpreter garbage collection at exit (the same reason
tests/full_model_checks.py and tests/perf_full_model.py end with os._exit). The optimizer runs these gates many
times per candidate, so skip that cleanup: leave with os._exit and pytest's final exit status once pytest is
done. Under the device profiler the normal exit stays intact so the tracy client and profiler can flush.
"""

import os
import sys

import pytest

_SESSION = {"value": None}


def pytest_sessionfinish(session, exitstatus):
    _SESSION["value"] = session


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    session = _SESSION["value"]
    # under the device profiler, or when pytest stopped before a session (a usage or collection-setup error), leave
    # the exit to pytest so its status stands
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" or session is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    # read the status now, after every session-finish hook ran (models/conftest.py's perf-target validation can turn
    # a passing session into a failure there)
    os._exit(int(session.exitstatus))
