# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import glob
import os

import pytest
from loguru import logger

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

# Device perf runs the model under Tracy in a subprocess. The parent must not open UMD / hold CHIP_IN_USE.
pytestmark = pytest.mark.no_reset_default_device

_MESH_DEVICE_LOCAL_CHIPS = {
    "N150": 1,
    "P150": 1,
    "N300": 2,
    "P300": 2,
    "T3K": 8,
    "TG": 32,
}


def _detect_pcie_node_count_without_opening() -> int:
    return sum(os.path.basename(path).isdigit() for path in glob.glob("/dev/tenstorrent/*"))


def _effective_local_chip_count_for_dp_skip() -> int:
    mesh = os.environ.get("MESH_DEVICE", "").strip().upper()
    if mesh and mesh in _MESH_DEVICE_LOCAL_CHIPS:
        return _MESH_DEVICE_LOCAL_CHIPS[mesh]
    n = _detect_pcie_node_count_without_opening()
    if n == 4:
        return 8
    return n


def _looks_like_multi_rank_host() -> bool:
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            return True
    except ValueError:
        pass
    try:
        if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
            return True
    except ValueError:
        pass
    return False


@pytest.mark.parametrize(
    "name_suffix,batch_size,resolution,expected_perf,test_filter,min_mesh_devices",
    [
        ["b1_640", 1, 640, 236.95, "test_yolov8s_640", 1],
        ["b2_640", 2, 640, 232.37, "dp_batch2 and n300_1x2", 2],
        ["b4_640", 4, 640, 230, "dp_batch4 and wh_1x4", 4],
        ["b8_640", 8, 640, 230, "dp_batch8 and t3k_1x8", 8],
    ],
    ids=["b1_640", "b2_640", "b4_640", "b8_640"],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8s(name_suffix, batch_size, resolution, expected_perf, test_filter, min_mesh_devices):
    if min_mesh_devices > 1 and not _looks_like_multi_rank_host():
        effective_chips = _effective_local_chip_count_for_dp_skip()
        if effective_chips < min_mesh_devices:
            pytest.skip(
                f"{name_suffix} needs a mesh with at least {min_mesh_devices} chips (DP uses one batch slice per chip). "
                f"Skip logic sees {effective_chips} chip(s) "
                f"(MESH_DEVICE={os.environ.get('MESH_DEVICE', '')!r}, "
                f"/dev/tenstorrent nodes={_detect_pcie_node_count_without_opening()}). "
                "Parent must not call ttnn.get_num_devices() here (CHIP_IN_USE vs Tracy child)."
            )

    subdir = "ttnn_yolov8s"
    num_iterations = 1
    margin = 0.05
    op_support_count = 6000
    command = "pytest models/demos/yolov8s/tests/pcc/test_yolov8s.py " f'-k "{test_filter}"'
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        op_support_count=op_support_count,
    )
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov8s_{name_suffix}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"resolution={resolution}",
    )
