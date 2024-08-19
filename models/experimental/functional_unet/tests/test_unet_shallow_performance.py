# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from tt_metal.tools.profiler.process_model_log import get_samples_per_s
from models.utility_functions import skip_for_grayskull
from models.experimental.functional_unet.unet_utils import create_unet_models, create_unet_input_tensors
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@skip_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("loop", [4])
def test_unet_wh_e2e_performance(device, perf_mode, batch, groups, loop, reset_seeds):
    with torch.no_grad():
        expected_compile_time = 25.0
        expected_inference_time = 0.15  # s

        torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(device, batch, groups)
        _, ttnn_model = create_unet_models(device, groups, torch_input_tensor)

        start = time.time()
        ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)
        end = time.time()
        inference_and_compile_time = end - start

        start = time.time()
        for _ in range(loop):
            ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)
        end = time.time()
        average_inference_time = (end - start) / loop

        prep_perf_report(
            model_name="unet_shallow",
            batch_size=batch,
            inference_and_compile_time=inference_and_compile_time,
            inference_time=average_inference_time,
            expected_compile_time=expected_compile_time,
            expected_inference_time=expected_inference_time,
            comments="",
            inference_time_cpu=0.0,
        )

        logger.info(f"Perf mode: {perf_mode}")
        logger.info(f"Batch: {batch}")
        logger.info(f"Groups: {groups}")
        logger.info(f"Loop: {loop}")
        logger.info(f"Compile time: {(inference_and_compile_time - average_inference_time):.2f} s")
        logger.info(f"Expected compile time: {expected_compile_time:.2f} s")
        logger.info(f"Average inference time: {(1000.0 * average_inference_time):.1f} ms")
        logger.info(f"Expected inference time: {(1000.0 * expected_inference_time):.1f} ms")


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
def test_unet_device_performance(device, perf_mode, batch, groups, reset_seeds):
    with torch.no_grad():
        torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(device, batch, groups)
        torch_model, ttnn_model = create_unet_models(device, groups, torch_input_tensor)

        ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)


@skip_for_grayskull()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("margin", [0.03])
@pytest.mark.parametrize("expected_inference_time_ms", [92.0])
@pytest.mark.parametrize("batch", [2])
def test_unet_wh_device_performance(margin, expected_inference_time_ms, batch):
    subdir = "unet_shallow"
    command = f"pytest models/experimental/functional_unet/tests/test_unet_shallow_performance.py::test_unet_device_performance[1-{batch}-True-device_params0]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    # Convert expected perf (ms) to samples/s
    expected_device_fw_duration_ns = expected_inference_time_ms * 1e6  # ms to ns
    expected_total_device_fw_samples = get_samples_per_s(expected_device_fw_duration_ns, batch)

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_total_device_fw_samples}

    post_processed_results = run_device_perf(command, subdir, 1, cols, batch)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"unet-shallow_batch-{batch}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
