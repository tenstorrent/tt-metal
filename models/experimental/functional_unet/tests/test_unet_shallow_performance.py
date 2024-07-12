# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import argparse

import torch
import torch.nn as nn
from loguru import logger

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_grayskull,
    is_x2_harvested,
    enable_persistent_kernel_cache,
)

from models.experimental.functional_unet.unet_utils import create_unet_models, create_unet_input_tensors

import time
import tt_lib as ttl
import os
from tt_lib import profiler

import ttnn
from models.perf.perf_utils import prep_perf_report

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@skip_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("loop", [10])
def test_unet_model_performance(device, perf_mode, batch, groups, loop):
    with torch.no_grad():
        # Perf report numbers needed
        inference_and_compile_time = 1  # need to figure out
        expected_compile_time = 0.5
        expected_inference_time = 0.1
        average_inference_time = 1

        torch.manual_seed(0)

        # Create initial parameters
        torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(device, batch, groups)
        torch_model, ttnn_model = create_unet_models(device, groups, torch_input_tensor)

        # Run torch golden result
        torch_output_tensor = torch_model(torch_input_tensor)

        # enable_persistent_kernel_cache()

        start_time = time.time()
        start = None
        for i in range(loop):
            if i == 0:
                start = time.perf_counter()
            profiler.tracy_frame()

            # Run ttnn output result
            output_tensor = ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)

        # Post processing
        end_time = time.time()
        stop = time.perf_counter()
        total_time = stop - start
        total_frame_count = batch * loop
        average_inference_time = (end_time - start_time) / loop
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
        logger.info(f"Elapsed host time (sec): {total_time}")
        logger.info(f"Frames processed: {total_frame_count}")
        logger.info(f"Host perf (fps): {total_frame_count / total_time}")
        logger.info(f"Inference and compile time: {inference_and_compile_time}")
        logger.info(f"Average inference time: {average_inference_time}")
        logger.info(f"Expected compile time: {expected_compile_time}")
        logger.info(f"Expected inference time: {expected_inference_time}")


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
def test_unet_device_performance_helper(device, perf_mode, batch, groups):
    with torch.no_grad():
        torch.manual_seed(0)

        # Create initial parameters
        torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(device, batch, groups)
        torch_model, ttnn_model = create_unet_models(device, groups, torch_input_tensor)

        # Run torch golden result
        torch_output_tensor = torch_model(torch_input_tensor)

        # Run ttnn output result
        output_tensor = ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)


@skip_for_grayskull()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("num_iterations", [3])
@pytest.mark.parametrize("margin", [2])
@pytest.mark.parametrize("expected_perf", [22])
@pytest.mark.parametrize("batch", [2])
def test_unet_device_performance(num_iterations, margin, expected_perf, batch):
    subdir = "ttnn_unet_shallow"
    command = f"pytest models/experimental/functional_unet/tests/test_unet_shallow_performance.py::test_unet_device_performance_helper"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_unet_{batch}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
