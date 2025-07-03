# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, temb_shape, down_block_id, resnet_id, conv_shortcut, split_in, block, pcc",
    [
        ((1, 320, 128, 128), (1, 1280), 0, 0, False, 1, "down_blocks", 0.999),
        ((1, 320, 64, 64), (1, 1280), 1, 0, True, 1, "down_blocks", 0.999),
        ((1, 640, 64, 64), (1, 1280), 1, 1, False, 1, "down_blocks", 0.997),
        ((1, 640, 32, 32), (1, 1280), 2, 0, True, 1, "down_blocks", 0.999),
        ((1, 1280, 32, 32), (1, 1280), 2, 1, False, 1, "down_blocks", 0.997),
        ((1, 960, 128, 128), (1, 1280), 2, 0, True, 2, "up_blocks", 0.998),
        ((1, 640, 128, 128), (1, 1280), 2, 1, True, 2, "up_blocks", 0.998),
        ((1, 2560, 32, 32), (1, 1280), 0, 0, True, 1, "up_blocks", 0.996),
        ((1, 1920, 32, 32), (1, 1280), 0, 2, True, 1, "up_blocks", 0.995),
        ((1, 1920, 64, 64), (1, 1280), 1, 0, True, 1, "up_blocks", 0.999),
        ((1, 1280, 64, 64), (1, 1280), 1, 1, True, 1, "up_blocks", 0.998),
        ((1, 960, 64, 64), (1, 1280), 1, 2, True, 1, "up_blocks", 0.998),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_resnetblock2d(
    device,
    temb_shape,
    input_shape,
    down_block_id,
    resnet_id,
    conv_shortcut,
    split_in,
    block,
    pcc,
    reset_seeds,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="unet"
    )
    unet.eval()
    state_dict = unet.state_dict()

    if block == "down_blocks":
        torch_resnet = unet.down_blocks[down_block_id].resnets[resnet_id]
    elif block == "up_blocks":
        torch_resnet = unet.up_blocks[down_block_id].resnets[resnet_id]
    else:
        assert "Incorrect block name"

    model_config = ModelOptimisations()
    tt_resnet = TtResnetBlock2D(
        device,
        state_dict,
        f"{block}.{down_block_id}.resnets.{resnet_id}",
        model_config,
        conv_shortcut,
        split_in,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_resnet(torch_input_tensor, torch_temb_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_temb_tensor = ttnn.from_torch(
        torch_temb_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_output_tensor, output_shape = tt_resnet.forward(ttnn_input_tensor, ttnn_temb_tensor, [B, C, H, W])

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")
