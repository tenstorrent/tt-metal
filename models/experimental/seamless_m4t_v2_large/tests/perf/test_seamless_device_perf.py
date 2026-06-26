# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seamless M4T v2 Large — device performance (per-op kernel time, tracy-measured, **no trace**).

Measures the device-bound kernel floor for each of the five inference tasks. The inner forward
test also runs ``generate(return_timings=True)`` and side-channels TT-catalog phase metrics
(steady decode t/s/u, TTFT, encoder ms) so this driver reports the same headline numbers as
``demo/demo.py``, alongside Tracy per-device kernel time.

How:
  * Outer test (this file) does **not** import ttnn or open the cluster — tracy spawns the inner
    pytest as a subprocess and opens devices itself.
  * Inner ``test_device_perf_forwards.py::test_<task>`` runs **eager**
    (``use_decode_trace=False``, ``use_2cq=False``).
  * Tracy sums kernel duration across OP rows, **divides by ``num_devices``** for TP per-device floor.
  * Inner test writes ``/tmp/seamless_dperf_<task>_timings.json`` and (speech tasks) sample count.

Usage::

    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \\
        -v -m models_device_performance_bare_metal

    MESH_DEVICE=P150 pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \\
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
_TIMINGS_PATH_FMT = "/tmp/seamless_dperf_{task}_timings.json"

_TASKS = (
    ("t2tt", False, 10),
    ("s2tt", False, 10),
    ("t2st", True, 4),
    ("s2st", True, 4),
    ("asr", False, 10),
)

_DEFAULT_MESH_SHAPE = (1, 4)
_TASK_OP_SUPPORT_COUNT = {
    "t2tt": 20000,
    "s2tt": 30000,
    "t2st": 30000,
    "s2st": 30000,
    "asr": 30000,
}


def _task_params():
    return [pytest.param(t, gs, mnt, id=t) for (t, gs, mnt) in _TASKS]


def _mesh_device_param() -> tuple[int, int]:
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in {"P150": (1, 1), "BH-QB": (1, 4)}:
        return {"P150": (1, 1), "BH-QB": (1, 4)}[mesh_env]
    if "TT_MESH_WIDTH" in os.environ:
        return (1, int(os.environ["TT_MESH_WIDTH"]))
    return _DEFAULT_MESH_SHAPE


def _mesh_num_devices(mesh_shape: tuple[int, int]) -> int:
    return max(1, int(mesh_shape[0]) * int(mesh_shape[1]))


def _mesh_id(mesh_shape: tuple[int, int]) -> str:
    return f"{int(mesh_shape[0])}x{int(mesh_shape[1])}"


def _inner_command(task: str) -> str:
    env_parts = []
    for key in ("MESH_DEVICE", "TT_MESH_WIDTH"):
        value = os.environ.get(key)
        if value is not None:
            env_parts.append(f"{key}={value}")
    prefix = " ".join(env_parts)
    command = f"pytest --timeout=0 {_FWD_TEST}::test_{task} -sv"
    return f"{prefix} {command}" if prefix else command


def _read_timings_side_channel(task: str) -> dict:
    path = _TIMINGS_PATH_FMT.format(task=task)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Timings side-channel missing at {path}")
        return {}


# Do not open or touch ttnn in this process — tracy spawns the inner test as a subprocess.
@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("task,generate_speech,max_new_tokens", _task_params())
def test_perf_device_bare_metal_seamless(task: str, generate_speech: bool, max_new_tokens: int):
    """Per-task device-bound perf via tracy on the eager (no-trace) forward."""
    batch_size = 1
    mesh_shape = _mesh_device_param()
    num_devices = _mesh_num_devices(mesh_shape)
    mesh_id = _mesh_id(mesh_shape)
    subdir = f"ttnn_seamless_m4t_v2_large_{task}_{mesh_id}"
    command = _inner_command(task)
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    duration_cols = [c + " DURATION [ns]" for c in cols]
    samples_cols = [c + " SAMPLES/S" for c in cols]

    sample_path = _SAMPLES_PATH_FMT.format(task=task)
    timings_path = _TIMINGS_PATH_FMT.format(task=task)
    for path in (sample_path, timings_path):
        if os.path.exists(path):
            os.remove(path)

    clear_profiler_runtime_artifacts()
    run_device_profiler(
        command,
        subdir,
        check_test_return_code=False,
        device_analysis_types=["device_kernel_duration"],
        op_support_count=_TASK_OP_SUPPORT_COUNT[task] * num_devices,
    )

    raw = post_process_ops_log(subdir, duration_cols)
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

    timings = _read_timings_side_channel(task)
    output_tokens = int(timings.get("output_tokens", 0) or 0)
    output_samples = int(timings.get("output_samples", 0) or 0)
    steady_ms_per_tok = float(timings.get("steady_decode_ms_per_tok", 0.0) or 0.0)
    decode_tok_s_u = float(timings.get("decode_tok_s_u", 0.0) or 0.0)
    ttft_ms = float(timings.get("ttft_ms", 0.0) or 0.0)
    encoder_ms = float(timings.get("encoder_ms", 0.0) or 0.0)
    e2e_ms = float(timings.get("e2e_ms", 0.0) or 0.0)
    t2u_ms = float(timings.get("t2u_ms", 0.0) or 0.0)
    vocoder_ms = float(timings.get("vocoder_ms", 0.0) or 0.0)
    rtf = float(timings.get("rtf", 0.0) or 0.0)

    post_processed_results["STEADY_DECODE_MS_PER_TOK"] = steady_ms_per_tok
    post_processed_results["DECODE_TOK_S_U"] = decode_tok_s_u
    post_processed_results["TTFT_MS"] = ttft_ms
    post_processed_results["ENCODER_MS"] = encoder_ms
    post_processed_results["E2E_MS"] = e2e_ms
    post_processed_results["OUTPUT_TOKENS"] = output_tokens
    post_processed_results["T2U_MS"] = t2u_ms
    post_processed_results["VOCODER_MS"] = vocoder_ms
    post_processed_results["RTF"] = rtf

    if generate_speech:
        try:
            with open(sample_path) as f:
                num_samples = int(f.read().strip())
        except FileNotFoundError:
            num_samples = output_samples
        device_vocoder_sps = (num_samples / kernel_seconds) if kernel_seconds > 0 else 0.0
        post_processed_results["AVG VOCODER SAMPLES/S (kernel)"] = device_vocoder_sps
        post_processed_results["NUM_AUDIO_SAMPLES"] = num_samples
        headline = (
            f"decode {decode_tok_s_u:.2f} t/s/u ({steady_ms_per_tok:.1f} ms/tok steady), "
            f"vocoder kernel {device_vocoder_sps:.0f} samples/s, RTF {rtf:.2f}x"
        )
        workload_str = f"{num_samples} audio samples, {output_tokens} text tokens"
    else:
        token_workload = output_tokens if output_tokens > 0 else max_new_tokens
        device_decode_tps = (token_workload / kernel_seconds) if kernel_seconds > 0 else 0.0
        post_processed_results["AVG TOKENS/S (kernel)"] = device_decode_tps
        post_processed_results["MAX_NEW_TOKENS"] = max_new_tokens
        headline = f"decode {decode_tok_s_u:.2f} t/s/u ({steady_ms_per_tok:.1f} ms/tok steady)"
        workload_str = f"{output_tokens} output tokens (budget {max_new_tokens})"

    # ``prep_device_perf_report`` formats values via ``value.is_integer()``, so keep all numeric
    # metrics as floats even when the underlying side-channel values are integer counts.
    post_processed_results = {name: float(value) for name, value in post_processed_results.items()}

    logger.info(f"\nTest: {command}\n{json.dumps(post_processed_results, indent=4)}")
    print(f"\n{'='*60}")
    print(f"Seamless M4T v2 Large Device Performance ({task.upper()}, {mesh_id})")
    print(f"{'='*60}")
    print(
        f"  TT-aligned (wall): {headline}  "
        f"(TTFT {ttft_ms:.1f} ms, encoder {encoder_ms:.1f} ms, e2e {e2e_ms:.1f} ms)"
    )
    print(
        f"  Tracy kernel floor: {kernel_ns / 1e6:.2f} ms per-device  "
        f"({workload_str}, TP={num_devices}, eager no-trace)"
    )
    if generate_speech and t2u_ms > 0:
        print(f"  Speech synth (wall): T2U {t2u_ms:.1f} ms, vocoder {vocoder_ms:.1f} ms")
    print(f"{'='*60}\n")

    prep_device_perf_report(
        model_name=f"ttnn_seamless_m4t_v2_large_batch{batch_size}_{task}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments=(
            f"seamless_m4t_v2_large_{task}_{mesh_id}_TP{num_devices}_eager_" f"decode_tok_s_u_{decode_tok_s_u:.1f}"
        ),
    )
