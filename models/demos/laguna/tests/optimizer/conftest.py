# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exit-hang guard for the optimizer gates.

A full 48-layer Laguna process spends minutes in interpreter garbage collection at exit (the same reason
tests/full_model_checks.py and tests/perf_full_model.py end with os._exit). The optimizer runs these gates many
times per candidate, so skip that cleanup: record pytest's exit status and leave with os._exit once pytest is
done. Under the device profiler the normal exit stays intact so the tracy client and profiler can flush.
"""

import os
import sys

import pytest

_EXIT_STATUS = {"value": 0}


def pytest_sessionfinish(session, exitstatus):
    _EXIT_STATUS["value"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS["value"])
