# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.experimental.yolov8s.tt.ttnn_yolov8s import TtYolov8sModel
from models.experimental.yolov8s.tt.tt_yolov8s_utils import custom_preprocessor
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, profiler


def get_expected_times(name):
    base = {"yolov8s": (128.267, 0.54)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor"])
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
def test_yolov8s(device, input_tensor, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()
    profiler.clear()
    batch_size = input_tensor.shape[0]
    inp_h, inp_w = input_tensor.shape[2], input_tensor.shape[3]
    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8s.pt")
        torch_model = torch_model.model
        torch_model.eval()
        state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)
    ttnn_model = TtYolov8sModel(device=device, parameters=parameters, res=(inp_h, inp_w))

    # input_tensor = torch.nn.functional.pad(input_tensor, (0, 13, 0, 0, 0, 0, 0, 0), value=0)
    # ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    n, c, h, w = input_tensor.shape
    if c == 3:
        c = 8
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn_input.to(device, input_mem_config)

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output_tensor = ttnn_model(ttnn_input)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        print("hellooo")
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_output_tensor = ttnn_model(ttnn_input)

        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {batch_size} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

    expected_compile_time, expected_inference_time = get_expected_times("yolov8s")

    prep_perf_report(
        model_name="models/demos/yolov8s",
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
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit Yolov8s perf test")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 225],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov8s(batch_size, expected_perf):
    subdir = "ttnn_yolov8s"
    num_iterations = 1
    margin = 0.03
    command = f"pytest tests/ttnn/integration_tests/yolov8s/test_yolov8s.py::test_yolov8s_640[use_weights_from_ultralytics=True-input_tensor1-0]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov8s{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
