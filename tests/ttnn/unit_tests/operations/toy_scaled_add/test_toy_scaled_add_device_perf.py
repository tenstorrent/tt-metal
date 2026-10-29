# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device kernel time of toy_scaled_add: the Python generic_op version and the C++ device operation
run the same kernels on the same work split, so their device time must match. The host-time test
(test_toy_scaled_add_host_perf.py) relies on this to attribute every difference to host code.

Runs in its own process because the device profiler is configured before ttnn starts.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/toy_scaled_add/test_toy_scaled_add_device_perf.py
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import json
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.toy_scaled_add import toy_scaled_add_generic

ROUTES = {"generic": toy_scaled_add_generic, "native": ttnn.toy_scaled_add}
DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
WARMUP = 2
SAMPLES = 15
RESULTS = Path("generated/toy_scaled_add/device_perf.json")


def _kernel_ns(device):
    """DEVICE KERNEL DURATION of every program since the previous read, summed."""
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _tensor(device, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


CASES = {
    "one_tile": ([1, 1, 32, 32], False),
    "grid_64_cores": ([1, 1, 32 * 256, 256], False),
    "grid_64_cores_gamma": ([1, 1, 32 * 256, 256], True),
}


@pytest.mark.parametrize("case_id", list(CASES))
def test_device_time_matches(device, case_id):
    shape, with_gamma = CASES[case_id]
    a, b = _tensor(device, shape), _tensor(device, shape)
    gamma = _tensor(device, [1, 1, 1, shape[-1]]) if with_gamma else None

    medians = {}
    for route, op in ROUTES.items():
        for _ in range(WARMUP):
            ttnn.deallocate(op(a, b, alpha=0.5, gamma=gamma))
        ttnn.synchronize_device(device)
        _kernel_ns(device)
        samples = []
        for _ in range(SAMPLES):
            out = op(a, b, alpha=0.5, gamma=gamma)
            ttnn.synchronize_device(device)
            ns = _kernel_ns(device)
            assert ns is not None, "the device profiler reported no kernel duration"
            samples.append(ns)
            ttnn.deallocate(out)
        medians[route] = statistics.median(samples) / 1e3

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[case_id] = {route: {"device_kernel_us": us} for route, us in medians.items()}
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\n{case_id}: " + ", ".join(f"{route} {us:.2f} us" for route, us in medians.items()))

    generic, native = medians["generic"], medians["native"]
    assert (
        abs(native - generic) <= 0.05 * generic
    ), f"device time differs: generic {generic:.2f} us, native {native:.2f} us"
