# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import time

from torchvision import models
from loguru import logger
from models.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_vgg.tt import ttnn_vgg_current
from models.utility_functions import (
    torch_random,
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times(functional_vgg):
    return {
        ttnn_vgg_current.ttnn_vgg16: (15.0, 9.2),
        ttnn_vgg_current.ttnn_vgg11: (12, 0.85),
    }[functional_vgg]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity", ((1, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),)
)
def test_vgg16(
    device,
    imagenet_sample_input,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
):
    disable_persistent_kernel_cache()
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = imagenet_sample_input.to(torch.bfloat16)
    torch_input_tensor_nchw = torch_input_tensor = imagenet_sample_input.to(torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=ttnn_vgg_current.custom_preprocessor,
    )

    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    conv_config = ttnn.Conv2dConfig(
        dtype=model_config["ACTIVATIONS_DTYPE"],
        weights_dtype=model_config["WEIGHTS_DTYPE"],
        math_fidelity=model_config["MATH_FIDELITY"],
        activation="relu",
        deallocate_activation=True,
        input_channels_alignment=16,
        act_block_h_override=0,
        transpose_shards=True,
    )

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    durations = []
    for i in range(2):
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        start = time.time()
        ttnn_output = ttnn_vgg_current.ttnn_vgg16(
            device, tt_input_tensor, parameters, batch_size, model_config, conv_config
        )
        output = ttnn.from_device(ttnn_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("vgg16")
    prep_perf_report(
        model_name="tt_vgg16",
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
    logger.info(f"Tokens per second: {1 / inference_time * batch_size}")
