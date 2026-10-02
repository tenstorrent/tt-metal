# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: record each test's outcome and total on-device kernel time.

After every test that uses the `device` fixture it reads the device profiler and sums DEVICE KERNEL DURATION over
all programs the test launched. One JSON line per test goes to $DEVICE_TIME_OUT. Running the same tests with
ttnn.CONFIG.matmul_auto_config_v2 off and on (TTNN_CONFIG_OVERRIDES) and comparing the two files shows per-test
correctness and device-time changes of the new matmul default selection.

  PYTHONPATH=tests/ttnn/unit_tests/benchmarks/matmul_oob:$PYTHONPATH DEVICE_TIME_OUT=off.jsonl pytest -p pytest_device_time <tests>

Each line also records auto_config: the last program config matmul's default selection chose during the test, or
null when no matmul in the test went through it (every matmul passed its own program_config, or the test has no
matmul). compare_pytest_times.py --auto-only compares just the tests where it is set.

DEVICE_TIME_SAMPLE=N keeps every Nth collected test (a fixed, deterministic sample for quick runs).
DEVICE_TIME_TESTS=FILE keeps only the tests whose node ids are listed in FILE, one per line.
"""

import json
import os

for _var in (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES",
):
    os.environ.setdefault(_var, "1")

import pytest  # noqa: E402

DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def pytest_collection_modifyitems(config, items):
    tests_file = os.environ.get("DEVICE_TIME_TESTS")
    if tests_file:
        with open(tests_file) as f:
            wanted = {line.strip() for line in f if line.strip()}
        config.hook.pytest_deselected(items=[i for i in items if i.nodeid not in wanted])
        items[:] = [i for i in items if i.nodeid in wanted]
    sample = int(os.environ.get("DEVICE_TIME_SAMPLE", "1"))
    if sample > 1:
        kept = items[::sample]
        config.hook.pytest_deselected(items=[i for i in items if i not in kept])
        items[:] = kept


_results = {}


def _read_device_time(device):
    import ttnn

    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    total, programs = 0, 0
    for entries in ttnn.get_latest_programs_perf_data().values():
        for p in entries:
            r = p.program_analyses_results.get(DURATION_KEY)
            if r is not None:
                total += r.duration
                programs += 1
    return total, programs


def _last_auto_config(reset):
    import ttnn

    return ttnn._ttnn.operations.matmul.matmul_last_auto_program_config(reset)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _last_auto_config(reset=True)  # from here to the end of the call, so matmuls in fixtures count too
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    device = item.funcargs.get("device") if hasattr(item, "funcargs") else None
    if device is not None:
        try:
            _read_device_time(device)  # drop anything left over from setup
        except Exception:
            device = None
    outcome = yield
    auto_config = _last_auto_config(reset=True)
    if device is None:
        _results[item.nodeid] = {"auto_config": auto_config}
    else:
        try:
            total, programs = _read_device_time(device)
            _results[item.nodeid] = {"device_ns": total, "programs": programs, "auto_config": auto_config}
        except Exception as e:
            _results[item.nodeid] = {
                "device_ns": None,
                "programs": None,
                "auto_config": auto_config,
                "error": str(e)[:200],
            }


def pytest_runtest_logreport(report):
    # One line per test: the call phase, or setup when the test never ran (skip or setup error)
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        entry = _results.pop(report.nodeid, {})
        entry["outcome"] = report.outcome
        path = os.environ.get("DEVICE_TIME_OUT")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps({"test": report.nodeid, **entry}) + "\n")
