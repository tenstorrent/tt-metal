# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary CI: print every BenchmarkProfiler step a test ends, as "EB_BENCH <nodeid> <step> <us>", so that
a test's own end-to-end timing (for example a trace replay) can be compared main against an opt-in without the profiler."""
import os as _os_guard, sys as _sys_guard
if not (_os_guard.environ.get("HWLOCK_HELD") or _os_guard.environ.get("GITHUB_ACTIONS")):
    _sys_guard.exit("not under hwlock")
import pytest

from tt_py_test_utils_common.perf import benchmarking_utils as _bu

_state = {"test": None}
_orig_end = _bu.BenchmarkProfiler.end


def _end(self, step_name, iteration=0):
    _orig_end(self, step_name, iteration)
    try:
        us = self.get_duration(step_name, iteration) * 1e6
        print(f"EB_BENCH {_state['test']} {step_name} {us:.1f}", flush=True)
    except KeyError:
        pass


_bu.BenchmarkProfiler.end = _end


def pytest_runtest_setup(item):
    _state["test"] = item.nodeid
