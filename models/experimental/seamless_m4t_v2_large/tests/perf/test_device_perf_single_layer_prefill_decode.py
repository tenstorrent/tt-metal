# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dump device perf report for Seamless M4T v2 single-layer prefill+decode profiling.

Runs the 1-layer prefill+decode profile workload under Tracy and writes
``device_perf_*.csv`` plus a partial benchmark JSON via ``prep_device_perf_report``.
No golden perf assertion — this script is for collecting / inspecting reports.

The profile workload matches ``test_profile_single_layer_prefill_decode.py`` (layer 0 only,
prefill 128 decoder tokens, cross-attn encoder timeline 128, decode at position 128).
Each measured iteration profiles **decoder forward only** (prefill + decode
``TTSeamlessM4Tv2Decoder.forward`` calls) inside the signpost window.

This script intentionally avoids importing ``ttnn`` so it does not compete for the UMD
``CHIP_IN_USE_*_PCIe`` lock with another pytest parent on multi-chip hosts.

Mesh: auto ``MeshShape(1, 4)`` when four devices are present, else ``MeshShape(1, 1)``.
Override with ``MESH_DEVICE=P150`` or ``MESH_DEVICE=BH-QB``.

Run::

    python models/experimental/seamless_m4t_v2_large/tests/perf/test_device_perf_single_layer_prefill_decode.py

After the run, analyze ``generated/profiler/seamless_m4t_v2_L1_prefill_decode/reports/*/ops_perf_results_*.csv``::

    tt-perf-report <ops_perf_results.csv> --start-signpost start --end-signpost stop
"""

from __future__ import annotations

import glob
import os


def _detect_num_devices() -> int:
    return len([p for p in glob.glob("/dev/tenstorrent/*") if os.path.basename(p).isdigit()])


def _mesh_device_param() -> tuple[int, int]:
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in {"P150": (1, 1), "BH-QB": (1, 4)}:
        return {"P150": (1, 1), "BH-QB": (1, 4)}[mesh_env]
    if "TT_MESH_WIDTH" in os.environ:
        return (1, int(os.environ["TT_MESH_WIDTH"]))
    return (1, 4) if _detect_num_devices() >= 4 else (1, 1)


def _mesh_num_devices(mesh_shape: tuple[int, int]) -> int:
    return max(1, int(mesh_shape[0]) * int(mesh_shape[1]))


def _inner_command() -> str:
    profile_test = (
        "models/experimental/seamless_m4t_v2_large/tests/perf/"
        "test_profile_single_layer_prefill_decode.py::test_profile_single_layer_prefill_decode"
    )
    # Tracy passes this string to ``python -m <module>`` — do NOT prefix ``VAR=val`` here
    # (that becomes ``ImportError: No module named VAR=val``). ``MESH_DEVICE`` / ``TT_MESH_WIDTH``
    # are read from the inherited process environment by the inner pytest.
    return f"pytest --timeout=3600 {profile_test} -sv"


def main() -> int:
    from loguru import logger
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_samples_per_s, post_process_ops_log, run_device_profiler

    from models.perf.device_perf_utils import prep_device_perf_report

    mesh_shape = _mesh_device_param()
    num_devices = _mesh_num_devices(mesh_shape)
    mesh_id = f"{mesh_shape[0]}x{mesh_shape[1]}"
    logger.info(
        f"Mesh auto-detect: MeshShape{mesh_shape} ({num_devices} device(s), "
        f"{_detect_num_devices()} chip node(s) under /dev/tenstorrent)"
    )
    model_name = f"seamless_m4t_v2_L1_prefill128_decode1_{mesh_id}"
    subdir = "seamless_m4t_v2_L1_prefill_decode"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    op_support_count = 8000 * num_devices

    if num_devices < 1:
        logger.error(f"no Tenstorrent devices found (detected {_detect_num_devices()} via /dev/tenstorrent)")
        return 2

    command = _inner_command()
    batch_size = 1
    duration_cols = [col + " DURATION [ns]" for col in cols]
    samples_cols = [col + " SAMPLES/S" for col in cols]

    clear_profiler_runtime_artifacts()
    run_device_profiler(
        command,
        subdir,
        check_test_return_code=False,
        device_analysis_types=["device_kernel_duration"],
        op_support_count=op_support_count,
    )

    raw = post_process_ops_log(subdir, duration_cols, has_signposts=True)
    post_processed_results = {}
    for s_col, d_col in zip(samples_cols, duration_cols):
        per_device_ns = raw[d_col] / num_devices
        post_processed_results[f"AVG {s_col}"] = get_samples_per_s(per_device_ns, batch_size)
        post_processed_results[f"MIN {s_col}"] = get_samples_per_s(per_device_ns, batch_size)
        post_processed_results[f"MAX {s_col}"] = get_samples_per_s(per_device_ns, batch_size)
        post_processed_results[f"AVG {d_col}"] = per_device_ns
        post_processed_results[f"MIN {d_col}"] = per_device_ns
        post_processed_results[f"MAX {d_col}"] = per_device_ns

    logger.info(f"Device perf results for {model_name}:\n{post_processed_results}")

    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments="prefill128_decode1_layer0_enc128",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
