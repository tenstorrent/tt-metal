# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seamless M4T v2 Large — device performance test.

Measures raw device kernel execution time using the device profiler. Runs the matching PCC
forward with profiler enabled and reports device throughput (samples/s).

Important: this outer pytest test must **not** import ttnn or open the cluster. Tracy spawns the
inner PCC test as a subprocess (``python3 -m tracy -p -r ...``) which opens devices itself; if
the outer process has already touched UMD, the subprocess deadlocks waiting for
``CHIP_IN_USE_0_PCIe``. The inner command targets ``test_<task>``; pytest skips the wrong
mesh id (``1x1`` vs ``1x4``) for this machine via ``skipif`` on each parametrization.

Usage::

    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \\
        -v -m models_device_performance_bare_metal
"""

import pytest

from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf

_PCC_TEST = "models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_m4t_v2_model.py"
_TASKS = ("t2tt", "s2tt", "t2st", "s2st", "asr")
# Tracy default buffer (~1333 ops) overflows on speech / T2U paths (2.5k+ device programs).
# Sized for worst case (BH QB 4-device mesh): outer process cannot query device count without
# opening the cluster (deadlocks the Tracy subprocess).
_MAX_MESH_DEVICES = 4
_TASK_OP_SUPPORT_COUNT: dict[str, int] = {
    "t2tt": 1500,
    "s2tt": 3500,
    "t2st": 2500,
    "s2st": 4000,
    "asr": 3500,
}


# Do not open or touch ttnn in this process — see module docstring.
@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("task", _TASKS)
def test_perf_device_bare_metal_seamless(task: str):
    """
    Device performance test for Seamless M4T v2 Large (one task per parametrization).

    Runs the matching PCC forward with device profiler. Inner pytest uses ``--timeout=0`` so the
    PCC test's own timeout governs the workload. The matching ``1x1`` or ``1x4`` case is selected
    by device-count skip marks on the PCC test parametrization.
    """
    batch_size = 1
    subdir = f"ttnn_seamless_m4t_v2_large_{task}"
    num_iterations = 1
    command = f"pytest --timeout=0 {_PCC_TEST}::test_{task} -sv"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        op_support_count=_TASK_OP_SUPPORT_COUNT[task] * _MAX_MESH_DEVICES,
    )
    actual_perf = post_processed_results.get(inference_time_key, 0)
    kernel_ns = post_processed_results.get("AVG DEVICE KERNEL DURATION [ns]", 0)
    print(f"\n{'='*60}")
    print(f"Seamless M4T v2 Large Device Performance ({task.upper()} PCC forward)")
    print(f"{'='*60}")
    print(f"  Measured:  {actual_perf:.2f} samples/s  ({kernel_ns / 1e6:.2f} ms kernel)")
    print(f"{'='*60}\n")

    prep_device_perf_report(
        model_name=f"ttnn_seamless_m4t_v2_large_batch{batch_size}_{task}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments=f"seamless_m4t_v2_large_{task}",
    )
