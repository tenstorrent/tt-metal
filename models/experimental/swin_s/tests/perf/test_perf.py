# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from torchvision import models
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, profiler
from models.experimental.swin_s.reference.swin_transformer import SwinTransformer

# from models.experimental.swin_v2.tt.model_preprocessing import create_swinv2_model_parameters, preprocess_attn_mask
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer import (
    create_custom_mesh_preprocessor,
    preprocess_attn_mask,
)
from models.experimental.swin_s.tt.tt_swin_transformer import TtSwinTransformer


def load_torch_model():
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()

    model.load_state_dict(state_dict)
    model.eval()
    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
    )

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    return torch_model


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, expected_compile_time, expected_inference_time",
    [
        ((1, 3, 512, 512), 91, 0.19),
    ],
)
def test_swin_s(
    device,
    input_shape,
    expected_compile_time,
    expected_inference_time,
    model_location_generator,
):
    disable_persistent_kernel_cache()
    profiler.clear()

    torch_model = load_torch_model()
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    batch_size = input_shape[0]
    resolution = input_shape[1:3]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_mesh_preprocessor(), device=device
    )

    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, device=device)
    attn_mask_tuple = preprocess_attn_mask([1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device)

    ttnn_model = TtSwinTransformer(
        device,
        parameters,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        attn_mask_tuple=attn_mask_tuple,
    )

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output_tensor = ttnn_model(ttnn_input)
    ttnn.deallocate(ttnn_output_tensor)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(
        f"Model with input resolution {resolution} compiled with warmup run in {(inference_and_compile_time):.2f} s"
    )

    iterations = 16

    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, device=device)
        ttnn_output_tensor = ttnn_model(ttnn_input)
        ttnn.deallocate(ttnn_output_tensor)
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation of resolution {resolution} took {compile_time:.1f} s")
    logger.info(
        f"Inference time on last iterations for resolution: {resolution} was completed in {(inference_time * 1000.0):.2f} ms"
    )
    logger.info(
        f"Mean inference time for {batch_size} (batch), resolution {resolution} images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

    prep_perf_report(
        model_name="Swin_s",
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
    "batch_size, model_name, expected_perf",
    [
        (1, "Swin_s", 5.3),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_swin_s(batch_size, model_name, expected_perf):
    subdir = model_name
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer.py::test_swin_s_transformer[pretrained_weight_true-0]"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
