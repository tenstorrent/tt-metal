# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov11.reference import yolov11
from models.demos.yolov11.tt import ttnn_yolov11
from models.demos.yolov11.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, enable_persistent_kernel_cache, is_wormhole_b0


def get_expected_times(name):
    base = {"yolov11": (110.9, 0.05)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [(1)])
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor"])
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        # True,
        False
    ],
    ids=[
        # "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_yolov11(device, input_tensor, batch_size, use_weights_from_ultralytics, min_channels=8):
    disable_persistent_kernel_cache()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=input_tensor.shape[0],
        input_channels=input_tensor.shape[1],
        input_height=input_tensor.shape[2],
        input_width=input_tensor.shape[3],
        is_sub_module=False,
    )
    if use_weights_from_ultralytics:
        torch_model = YOLO("yolo11n.pt")
        state_dict = {k.replace("model.", "", 1): v for k, v in torch_model.state_dict().items()}

    torch_model = yolov11.YoloV11()
    torch_model.eval()
    if use_weights_from_ultralytics:
        torch_model.load_state_dict(state_dict)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    model = ttnn_yolov11.TtnnYoloV11(device, parameters)
    durations = []
    n, c, h, w = ttnn_input.shape
    if c == 3:  # for sharding config of padded input
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    for i in range(2):
        ttnn_input_sharded = ttnn_input.to(device, input_mem_config)
        start = time.time()
        ttnn_model_output = model(ttnn_input_sharded)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov11")

    prep_perf_report(
        model_name="models/demos/yolov11",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 198],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov11(batch_size, expected_perf):
    subdir = "ttnn_yolov11"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest tests/ttnn/integration_tests/yolov11/test_ttnn_yolov11.py::test_yolov11"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov11{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
