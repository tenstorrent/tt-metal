# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from pathlib import Path

from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_maskformer_swin_b():
    batch_size = 1
    subdir = "maskformer_swin_base_coco"
    num_iterations = 1
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    artifact_root = Path("generated/maskformer_swin/perf_test")
    perf_json = artifact_root / "perf.json"
    perf_header_json = artifact_root / "perf_header.json"
    demo_outputs = artifact_root / "demo_outputs"

    command = (
        "python -m models.experimental.maskformer_swin.demo.runner "
        "--image models/sample_data/demo.jpeg "
        "--weights facebook/maskformer-swin-base-coco "
        "--height 320 --width 320 "
        "--optimization-stage stage3 "
        "--tt-repeats 1 "
        f"--output-dir {demo_outputs} "
        f"--dump-perf {perf_json} "
        f"--dump-perf-header {perf_header_json}"
    )
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
