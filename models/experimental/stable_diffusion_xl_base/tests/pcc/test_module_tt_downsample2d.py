# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_downsample2d import TtDownsample2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    to_channel_last_ttnn,
    from_channel_last_ttnn,
)
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize("input_shape, down_block_id", [((1, 320, 128, 128), 0), ((1, 640, 64, 64), 1)])
@pytest.mark.parametrize("stride", [(2, 2)])
@pytest.mark.parametrize("padding", [(1, 1)])
@pytest.mark.parametrize("dilation", [(1, 1)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_downsample2d(
    device,
    input_shape,
    down_block_id,
    stride,
    padding,
    dilation,
    reset_seeds,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="unet"
    )
    unet.eval()
    state_dict = unet.state_dict()

    torch_downsample = unet.down_blocks[down_block_id].downsamplers[0]
    groups = 1

    model_config = ModelOptimisations()
    tt_downsample = TtDownsample2D(
        device,
        state_dict,
        f"down_blocks.{down_block_id}.downsamplers.0",
        stride,
        padding,
        dilation,
        groups,
        model_config=model_config,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_downsample(torch_input_tensor)

    ttnn_input_tensor = to_channel_last_ttnn(
        torch_input_tensor, ttnn.bfloat16, device, ttnn.DRAM_MEMORY_CONFIG, ttnn.TILE_LAYOUT
    )

    ttnn_output_tensor, output_shape = tt_downsample.forward(ttnn_input_tensor, input_shape)

    output_tensor = from_channel_last_ttnn(
        ttnn_output_tensor, [input_shape[0], output_shape[1], output_shape[2], output_shape[0]]
    )

    del unet, tt_downsample
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"PCC is {pcc_message}")
