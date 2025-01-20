# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import ttnn
import torch

from loguru import logger

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)

from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import verify_with_pcc, UNET_FULL_MODEL_PCC


def nearest_16(x):
    return math.ceil(x / 16) * 16


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [2, 4, 8])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_unet_preprocessing(batch, groups, device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups, channel_order="first", pad=False, fold=False)
    logger.info(f"Created input tensor with shape {list(ttnn_input.shape)}")

    assert list(torch_input.shape) == list(ttnn_input.shape), "Expected torch and TTNN input shapes to match"

    min_channels = 16

    def golden_fn(x):
        N, C, H, W = x.shape
        if C < min_channels:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, min_channels - C), mode="constant", value=0)
        return x.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, max(C, min_channels))

    torch_output_tensor = golden_fn(torch_input)

    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        [ttnn_input.shape[0], nearest_16(ttnn_input.shape[1]), ttnn_input.shape[2], ttnn_input.shape[3]],
        ttnn.CoreGrid(x=8, y=6),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device, memory_config=input_sharded_memory_config)

    ttnn_output_tensor = unet_shallow_ttnn.preprocess_unet_input_tensor(ttnn_input)  # 1, 1, NHW, C (padded up to 16)
    logger.info(f"Preprocessing input tensor yielded the following shape: {list(ttnn_output_tensor.shape)}")

    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
    assert list(torch_output_tensor.shape) == list(
        torch_output_tensor.shape
    ), "Expected torch and TTNN input shapes to match"
    verify_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.99999)
