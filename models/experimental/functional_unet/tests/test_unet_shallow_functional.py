# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull

from models.experimental.functional_unet.unet_utils import create_unet_models, create_unet_input_tensors


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
def test_unet_pcc(device, perf_mode, batch, groups, reset_seeds, use_program_cache):
    logger.info(f"Running UNet test with batch_size={batch}, groups={groups}")
    with torch.no_grad():
        torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(device, batch, groups)
        logger.info(f"Created UNet input tensors: {list(torch_input_tensor.shape)}, {list(ttnn_input_tensor.shape)}")

        start = time.time()
        torch_model, ttnn_model = create_unet_models(device, groups, torch_input_tensor)
        logger.info(f"Initialized UNet and its reference model in {(time.time() - start):.2f} s")

        start = time.time()
        torch_output_tensor = torch_model(torch_input_tensor)
        logger.info(f"Finished UNet reference model inference in {1000.0 * (time.time() - start):.1f} ms")

        start = time.time()
        output_tensor = ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)
        logger.info(f"Finished UNet model inference in {1000.0 * (time.time() - start):.1f} ms")

        # Reshape output to match torch output
        input_shape = torch_input_tensor.shape
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = output_tensor[:, :, :, :1]
        output_tensor = output_tensor.reshape(input_shape[0], input_shape[2], input_shape[3], -1)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
