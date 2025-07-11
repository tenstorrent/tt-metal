# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import torch
import pytest
from loguru import logger
from models.utility_functions import is_wormhole_b0
from models.perf.perf_utils import prep_perf_report
from models.experimental.yolov12x.reference import yolov12x
from models.experimental.yolov12x.tt.yolov12x import YoloV12x
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.experimental.yolov12x.tt.model_preprocessing import (
    create_yolov12x_model_parameters,
)
from ultralytics import YOLO


def get_expected_times(name):
    base = {"yolov12x": (140.9, 0.09)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [(1)])
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor"])
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        True,
        # False
    ],
    ids=[
        "pretrained_weight_true",
        # "pretrained_weight_false",
    ],
)
def test_yolov12x(device, input_tensor, batch_size, use_weights_from_ultralytics, min_channels=16):
    disable_persistent_kernel_cache()
    torch_input = torch.randn(1, 3, 640, 640)
    n, c, h, w = torch_input.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )

    if use_weights_from_ultralytics:
        torch_model = YOLO("yolo12x.pt")
        state_dict = torch_model.state_dict()

    torch_model = yolov12x.YoloV12x()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)

    torch_model.eval()
    parameters = create_yolov12x_model_parameters(torch_model, torch_input, device=device)
    model = YoloV12x(device, parameters)
    durations = []
    for i in range(2):
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn_input = ttnn_input.to(device, input_mem_config)
        start = time.time()
        ttnn_model_output = model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov12x")

    prep_perf_report(
        model_name="models/experimental/yolov12x",
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
        [1, 14],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov12x(batch_size, expected_perf):
    subdir = "ttnn_yolov12x"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest tests/ttnn/integration_tests/yolov12x/test_ttnn_yolov12x.py"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov12x{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
