# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.quasar.tt import qsr
from models.demos.stable_diffusion_xl_base.quasar.tests.sdxl_quasar_test_utils import load_torch_unet
from models.common.utility_functions import torch_random
from models.demos.stable_diffusion_xl_base.quasar.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.quasar.tt.tt_resnetblock2d import TtResnetBlock2D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, temb_shape, down_block_id, resnet_id, conv_shortcut, block, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 320, 128, 128), (1, 1280), 0, 0, False, "down_blocks", 0.999),
        ((1024, 1024), (1, 320, 64, 64), (1, 1280), 1, 0, True, "down_blocks", 0.999),
        ((1024, 1024), (1, 640, 64, 64), (1, 1280), 1, 1, False, "down_blocks", 0.999),
        ((1024, 1024), (1, 640, 32, 32), (1, 1280), 2, 0, True, "down_blocks", 0.999),
        ((1024, 1024), (1, 1280, 32, 32), (1, 1280), 2, 1, False, "down_blocks", 0.999),
        ((1024, 1024), (1, 2560, 32, 32), (1, 1280), 0, 0, True, "up_blocks", 0.999),
        ((1024, 1024), (1, 1920, 32, 32), (1, 1280), 0, 2, True, "up_blocks", 0.999),
        ((1024, 1024), (1, 1920, 64, 64), (1, 1280), 1, 0, True, "up_blocks", 0.999),
        ((1024, 1024), (1, 1280, 64, 64), (1, 1280), 1, 1, True, "up_blocks", 0.999),
        ((1024, 1024), (1, 960, 64, 64), (1, 1280), 1, 2, True, "up_blocks", 0.999),
        ((1024, 1024), (1, 960, 128, 128), (1, 1280), 2, 0, True, "up_blocks", 0.998),
        ((1024, 1024), (1, 640, 128, 128), (1, 1280), 2, 1, True, "up_blocks", 0.998),
    ],
)
def test_resnetblock2d(
    device,
    image_resolution,
    temb_shape,
    input_shape,
    down_block_id,
    resnet_id,
    conv_shortcut,
    block,
    pcc,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    unet = load_torch_unet(sdxl_base_unet_location, is_ci_env, is_ci_v2_env)
    state_dict = unet.state_dict()

    if block == "down_blocks":
        torch_resnet = unet.down_blocks[down_block_id].resnets[resnet_id]
    elif block == "up_blocks":
        torch_resnet = unet.up_blocks[down_block_id].resnets[resnet_id]
    else:
        assert "Incorrect block name"

    model_config = load_model_optimisations(image_resolution)
    tt_resnet = TtResnetBlock2D(
        device,
        state_dict,
        f"{block}.{down_block_id}.resnets.{resnet_id}",
        model_config,
        conv_shortcut,
        debug_mode=debug_mode,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_resnet(torch_input_tensor, torch_temb_tensor)

    ttnn_input_tensor, [B, C, H, W] = to_device_nhwc(torch_input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn_temb_tensor = qsr.from_torch(
        torch.nn.functional.silu(torch_temb_tensor),
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
