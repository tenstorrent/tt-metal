# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end performance test for YOLO11 Pose Estimation

Tests full inference pipeline performance including:
- Model initialization
- Input preprocessing
- TTNN inference on TT hardware
- Output postprocessing
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.runner.performant_runner_pose import YOLOv11PosePerformantRunner

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_yolov11_pose_inference(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
):
    """
    Run YOLO11 Pose inference with performant runner and measure performance

    Args:
        device: TT device
        batch_size_per_device: Batch size per device
        act_dtype: Activation data type
        weight_dtype: Weight data type
        resolution: Input resolution (height, width)
    """
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    logger.info(f"Initializing YOLO11 Pose Performant Runner for batch_size={batch_size}...")

    # Create performant runner (includes trace capture)
    performant_runner = YOLOv11PosePerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        outputs_mesh_composer=outputs_mesh_composer,
    )

    # Create input tensor
    input_shape = (batch_size, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    logger.info("Performant runner ready. Starting performance measurement...")
    logger.info("Running 100 iterations with trace execution...")

    if use_signpost:
        signpost(header="start")

    t0 = time.time()
    for i in range(100):
        _ = performant_runner.run(torch_input_tensor=torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    if use_signpost:
        signpost(header="stop")

    # Release trace
    performant_runner.release()

    # Calculate metrics
    total_time = t1 - t0
    inference_time_avg = total_time / 100
    fps = batch_size / inference_time_avg

    logger.info("=" * 70)
    logger.info(f"YOLO11 Pose Performance Results (with Performant Runner):")
    logger.info(f"  Model: ttnn_yolov11_pose")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  Total time (100 iters): {total_time:.2f} sec")
    logger.info(f"  Avg inference time: {inference_time_avg:.6f} sec")
    logger.info(f"  FPS (frames per second): {fps:.2f}")
    logger.info(f"  Throughput: {fps:.2f} images/sec")
    logger.info("=" * 70)


@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant_pose(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
    reset_seeds,
):
    """
    Test end-to-end performance of YOLO11 Pose model

    Runs 100 iterations and measures:
    - Average inference time
    - FPS (frames per second)
    - Throughput
    """
    run_yolov11_pose_inference(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution,
    )


@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant_pose_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
    reset_seeds,
):
    """
    Test end-to-end performance with data parallelism (multi-device)

    Runs on mesh device with multiple chips for higher throughput.
    """
    run_yolov11_pose_inference(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution,
    )
