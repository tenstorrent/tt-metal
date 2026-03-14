# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from loguru import logger
from pathlib import Path
from unittest.mock import patch

from tracy.process_model_log import run_device_profiler as tracy_run_device_profiler
from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf
import models.perf.device_perf_utils as device_perf_utils


def _run_device_profiler_with_high_op_support(*args, **kwargs):
    kwargs.setdefault("op_support_count", 20000)
    return tracy_run_device_profiler(*args, **kwargs)


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_maskformer_swin_b():
    os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "20000")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    # Ensure the profiler subprocess uses the repo build of TTNN (must be Tracy-enabled).
    repo_root = Path(__file__).resolve().parents[5]
    ttnn_py_path = str(repo_root / "ttnn")
    os.environ["PYTHONPATH"] = ttnn_py_path + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

    batch_size = 1
    subdir = "maskformer_swin_base_coco"
    num_iterations = 1
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    artifact_root = Path("generated/maskformer_swin/perf_test")
    perf_json = artifact_root / "perf.json"
    perf_header_json = artifact_root / "perf_header.json"
    demo_outputs = artifact_root / "demo_outputs"

    command = (
        "models.experimental.maskformer_swin.demo.runner "
        "--image models/sample_data/demo.jpeg "
        "--weights facebook/maskformer-swin-base-coco "
        "--height 320 --width 320 "
        "--optimization-stage stage3 "
        "--tt-repeats 1 "
        f"--output-dir {demo_outputs} "
        f"--dump-perf {perf_json} "
        f"--dump-perf-header {perf_header_json}"
    )
    with patch.object(device_perf_utils, "run_device_profiler", side_effect=_run_device_profiler_with_high_op_support):
        post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)

    # No fixed perf target yet (bring-up baseline); report only.
    expected_results = {}
    logger.info(f"{post_processed_results}")

    assert perf_json.exists(), f"Expected perf JSON at {perf_json}"
    assert perf_header_json.exists(), f"Expected perf header JSON at {perf_header_json}"
    assert (demo_outputs / "semantic_overlay.png").exists(), "Expected semantic overlay artifact from TT-only demo run"

    prep_device_perf_report(
        model_name=f"ttnn_maskformer_swin_b_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="baseline",
    )
