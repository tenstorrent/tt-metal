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

Each line also records custom_configs: how many ttnn.matmul / ttnn.linear calls in the test passed their own
program_config. With DEVICE_TIME_STRIP_CONFIGS=1 those calls go through the default selection instead: the
program_config is dropped, and a fused_activation it carried is passed as activation. A call without its own
compute_kernel_config gets the one it resolved to with the program config: matmul's default math fidelity is LoFi
with a program config and HiFi2 without (#55889), which would otherwise dominate the comparison.

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
_custom_configs = 0


def _default_compute_config(a, b, dtype):
    """The compute config matmul resolves for a call with a program config and no compute config of its own"""
    import ttnn

    arch = a.device().arch()
    fp32_inputs = a.dtype == ttnn.float32 and b.dtype == ttnn.float32
    if fp32_inputs:
        fidelity = ttnn.MathFidelity.HiFi3 if arch == ttnn.device.Arch.WORMHOLE_B0 else ttnn.MathFidelity.HiFi4
    else:
        fidelity = ttnn.MathFidelity.LoFi
    fp32_out = (dtype or a.dtype) == ttnn.float32
    return ttnn.init_device_compute_kernel_config(
        arch,
        None,
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_out,
        packer_l1_acc=not fp32_out,
    )


def _wrap(module, name):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        global _custom_configs
        config = kwargs.get("program_config")
        if config is not None:
            _custom_configs += 1
            if os.environ.get("DEVICE_TIME_STRIP_CONFIGS") == "1":
                kwargs["program_config"] = None
                fused = getattr(config, "fused_activation", None)
                if fused is not None and kwargs.get("activation") is None:
                    kwargs["activation"] = fused
                if kwargs.get("compute_kernel_config") is None:
                    a = args[0] if len(args) > 0 else kwargs.get("input_tensor_a")
                    b = args[1] if len(args) > 1 else kwargs.get("input_tensor_b")
                    kwargs["compute_kernel_config"] = _default_compute_config(a, b, kwargs.get("dtype"))
        return original(*args, **kwargs)

    setattr(module, name, wrapper)


def pytest_configure(config):
    import ttnn

    for name in ("matmul", "linear"):
        _wrap(ttnn, name)


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
    global _custom_configs
    _last_auto_config(reset=True)  # from here to the end of the call, so matmuls in fixtures count too
    _custom_configs = 0
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
        _results[item.nodeid] = {"auto_config": auto_config, "custom_configs": _custom_configs}
    else:
        try:
            total, programs = _read_device_time(device)
            _results[item.nodeid] = {
                "device_ns": total,
                "programs": programs,
                "auto_config": auto_config,
                "custom_configs": _custom_configs,
            }
        except Exception as e:
            _results[item.nodeid] = {
                "device_ns": None,
                "programs": None,
                "auto_config": auto_config,
                "custom_configs": _custom_configs,
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
