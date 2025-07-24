# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.experimental.vanilla_unet.ttnn.ttnn_unet import TtUnet
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import profiler
from models.experimental.vanilla_unet.test.pcc.test_ttnn_unet import create_custom_preprocessor
from models.experimental.vanilla_unet.load_model_utils import load_torch_model


def get_expected_times(name):
    base = {"vanilla_unet": (64.44, 0.019)}
    return base[name]


@pytest.mark.parametrize("device_params", [{"l1_small_size": (7 * 8192) + 1730}], indirect=True, ids=["0"])
def test_vanilla_unet(device, reset_seeds, model_location_generator):
    torch.manual_seed(0)

    reference_model = load_torch_model(model_location_generator)

    torch_input_tensor = torch.randn(1, 3, 480, 640)
    batch_size = torch_input_tensor.shape[0]
    torch_output_tensor = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    ttnn_model = TtUnet(device=device, parameters=parameters, model=reference_model)

    n, c, h, w = torch_input_tensor.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, 640, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input_host = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output = ttnn_model(device, ttnn_input_host)
    ttnn.deallocate(ttnn_output)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        ttnn_input_host = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_output = ttnn_model(device, ttnn_input_host)
        ttnn.deallocate(ttnn_output)
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

    expected_compile_time, expected_inference_time = get_expected_times("vanilla_unet")

    prep_perf_report(
        model_name="models/experimental/vanilla_unet",
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
    logger.info("Exit vanilla_unet perf test")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 46.7],
    ],
)
def test_perf_device_vanilla_unet(batch_size, expected_perf):
    subdir = "ttnn_vanilla_unet"
    num_iterations = 1
    margin = 0.03
    command = f"pytest models/experimental/vanilla_unet/test/pcc/test_ttnn_unet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_vanilla_unet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
