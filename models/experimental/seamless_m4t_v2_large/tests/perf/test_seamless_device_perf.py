# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seamless M4T v2 Large — device performance (per-op kernel time, tracy-measured, **no trace**).

Measures the device-bound floor for each of the five inference tasks: how fast the device kernels
can run, with all host overhead and trace-replay optimizations stripped out. This is what e2e
perf approaches but cannot beat — the e2e test (``test_e2e_perf_2cq.py``) measures the production
wall-clock (with trace + 2 CQ), so a healthy state has ``device tokens/sec ≥ e2e tokens/sec``.

How:
  * Outer test (this file) does **not** import ttnn or open the cluster — tracy spawns the inner
    pytest as a subprocess and opens devices itself. Touching UMD from the outer process makes
    the subprocess deadlock waiting for ``CHIP_IN_USE_0_PCIe``.
  * Inner ``test_device_perf_forwards.py::test_<task>`` runs **eager** (``use_decode_trace=False``,
    ``use_2cq=False``) — tracy's per-op profiler can't reconcile host/device records across metal
    trace replays, which is why traditional device perf disables trace.
  * Tracy writes per-op kernel timings to ``cpp_device_perf_report.csv``. We sum kernel duration
    across every row, then **divide by ``num_devices``** to get the per-device wall-clock-equivalent
    kernel time (under TP all mesh devices run the same op in parallel; the sum is N× the
    per-device floor).
  * Throughput metric: tokens/sec for text outputs, samples/sec for speech outputs (sample count
    side-channeled from the inner test via ``SAMPLES_PATH_FMT``).

The inner pytest occasionally segfaults during teardown after PASSED on speech-output paths
(``ReadDeviceProfiler`` + ``clear_program_cache`` race). We use ``check_test_return_code=False``
so the already-captured per-op CSV survives — the timings are valid regardless of exit code.

Usage::

    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \\
        -v -m models_device_performance_bare_metal
"""

import json
import os

import pytest
from loguru import logger
from tracy.common import clear_profiler_runtime_artifacts
from tracy.process_model_log import get_samples_per_s, post_process_ops_log, run_device_profiler

from models.perf.device_perf_utils import prep_device_perf_report

_FWD_TEST = "models/experimental/seamless_m4t_v2_large/tests/perf/test_device_perf_forwards.py"
_SAMPLES_PATH_FMT = "/tmp/seamless_dperf_{task}_samples.txt"

# Task definitions — mirror the inner forward tests' parametrization.
# (task_id, generate_speech, max_new_tokens) — max_new_tokens must match ``_TEXT_KWARGS`` /
# ``_SPEECH_KWARGS`` in ``test_device_perf_forwards.py``.
_TASKS = (
    ("t2tt", False, 10),
    ("s2tt", False, 10),
    ("t2st", True, 4),
    ("s2st", True, 4),
    ("asr", False, 10),
)

# Per-op profiler buffer budget (ops × mesh_devices). Speech-output paths run a smaller
# ``max_new_tokens`` (see ``_MAX_NEW_TOKENS_SPEECH``) so a moderate buffer is enough — pushing
# this too high stresses the per-device DRAM allocation and triggers segfaults inside
# ``ReadDeviceProfiler``.
_MAX_MESH_DEVICES = 4
_TASK_OP_SUPPORT_COUNT = {
    "t2tt": 20000,
    "s2tt": 30000,
    "t2st": 30000,
    "s2st": 30000,
    "asr": 30000,
}


def _task_params():
    return [pytest.param(t, gs, mnt, id=t) for (t, gs, mnt) in _TASKS]


# Do not open or touch ttnn in this process — tracy spawns the inner test as a subprocess.
@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("task,generate_speech,max_new_tokens", _task_params())
def test_perf_device_bare_metal_seamless(task: str, generate_speech: bool, max_new_tokens: int):
    """Per-task device-bound perf via tracy on the eager (no-trace) forward."""
    batch_size = 1
    subdir = f"ttnn_seamless_m4t_v2_large_{task}"
    command = f"pytest --timeout=0 {_FWD_TEST}::test_{task} -sv"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    duration_cols = [c + " DURATION [ns]" for c in cols]
    samples_cols = [c + " SAMPLES/S" for c in cols]

    # Clear the side-channel file before the inner run so we don't pick up a stale sample count
    # if the speech inner test failed early.
    sample_path = _SAMPLES_PATH_FMT.format(task=task)
    if generate_speech and os.path.exists(sample_path):
        os.remove(sample_path)

    # ``check_test_return_code=False``: the inner pytest occasionally segfaults during teardown
    # on speech-output paths *after* the test reports PASSED — by then tracy has already written
    # the per-op CSV, so we want the post-processing to proceed even on non-zero exit.
    clear_profiler_runtime_artifacts()
    run_device_profiler(
        command,
        subdir,
        check_test_return_code=False,
        device_analysis_types=["device_kernel_duration"],
        op_support_count=_TASK_OP_SUPPORT_COUNT[task] * _MAX_MESH_DEVICES,
    )

    # Tracy's ``post_process_ops_log(sum_vals=True)`` sums kernel durations across *every row* in
    # the OPs CSV — under TP all mesh devices run the same op in parallel, so the sum is N× the
    # per-device wall-clock-equivalent kernel time. Divide by ``num_devices`` to get the floor.
    raw = post_process_ops_log(subdir, duration_cols)
    num_devices = _MAX_MESH_DEVICES
    post_processed_results = {}
    for s_col, d_col in zip(samples_cols, duration_cols):
        per_device_ns = raw[d_col] / num_devices
        post_processed_results[f"AVG {s_col}"] = get_samples_per_s(per_device_ns, batch_size)
        post_processed_results[f"MIN {s_col}"] = get_samples_per_s(per_device_ns, batch_size)
        post_processed_results[f"MAX {s_col}"] = get_samples_per_s(per_device_ns, batch_size)
        post_processed_results[f"AVG {d_col}"] = per_device_ns
        post_processed_results[f"MIN {d_col}"] = per_device_ns
        post_processed_results[f"MAX {d_col}"] = per_device_ns

    kernel_ns = post_processed_results.get("AVG DEVICE KERNEL DURATION [ns]", 0.0)
    kernel_seconds = kernel_ns / 1e9

    # Pick the right throughput unit per task type:
    #   text outputs  → tokens/sec  = max_new_tokens / per-device kernel time
    #   speech outputs → samples/sec = num_audio_samples / per-device kernel time
    # ``num_audio_samples`` comes from the side-channel file written by the inner forward test.
    if generate_speech:
        try:
            with open(sample_path) as f:
                num_samples = int(f.read().strip())
        except FileNotFoundError:
            logger.warning(f"Speech sample-count side-channel missing at {sample_path}; reporting 0")
            num_samples = 0
        throughput = (num_samples / kernel_seconds) if kernel_seconds > 0 else 0.0
        throughput_unit = "samples/s"
        workload_str = f"{num_samples} audio samples"
        post_processed_results["AVG SAMPLES/S"] = throughput
        post_processed_results["NUM_AUDIO_SAMPLES"] = num_samples
    else:
        throughput = (max_new_tokens / kernel_seconds) if kernel_seconds > 0 else 0.0
        throughput_unit = "tokens/s"
        workload_str = f"{max_new_tokens} new tokens"
        post_processed_results["AVG TOKENS/S"] = throughput
        post_processed_results["MAX_NEW_TOKENS"] = max_new_tokens

    logger.info(f"\nTest: {command}\n{json.dumps(post_processed_results, indent=4)}")
    print(f"\n{'='*60}")
    print(f"Seamless M4T v2 Large Device Performance ({task.upper()})")
    print(f"{'='*60}")
    print(
        f"  Measured: {throughput:.2f} {throughput_unit}  "
        f"({kernel_ns / 1e6:.2f} ms per-device kernel, {workload_str}, TP={num_devices})"
    )
    print(f"{'='*60}\n")

    prep_device_perf_report(
        model_name=f"ttnn_seamless_m4t_v2_large_batch{batch_size}_{task}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments=f"seamless_m4t_v2_large_{task}_TP{num_devices}_eager_max_new_tokens{max_new_tokens}",
    )
