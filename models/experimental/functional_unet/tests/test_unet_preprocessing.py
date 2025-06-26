# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import ttnn

from loguru import logger

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
)

from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import verify_with_pcc, UNET_L1_SMALL_REGION_SIZE


def nearest_16(x):
    return math.ceil(x / 16) * 16


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_preprocessing(batch, groups, device, reset_seeds):
    input_memory_config = unet_shallow_ttnn.UNet.input_sharded_memory_config
    torch_input, ttnn_input = create_unet_input_tensors(
        batch, groups, channel_order="first", pad=False, fold=True, device=device, memory_config=input_memory_config
    )
    logger.info(f"Created input tensor with shape {list(ttnn_input.shape)}")

    def golden_fn(x):
        N, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)

    torch_output_tensor = golden_fn(torch_input)

    ttnn_output_tensor = unet_shallow_ttnn.preprocess_unet_input_tensor(ttnn_input)  # 1, 1, NHW, C (padded up to 16)
    logger.info(f"Preprocessing input tensor yielded the following shape: {list(ttnn_output_tensor.shape)}")

    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
    assert list(torch_output_tensor.shape) == list(
        torch_output_tensor.shape
    ), "Expected torch and TTNN input shapes to match"
    verify_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.99999)
