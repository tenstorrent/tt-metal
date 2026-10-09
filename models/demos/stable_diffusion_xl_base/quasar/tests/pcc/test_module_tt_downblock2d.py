# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.quasar.tt import qsr
from models.demos.stable_diffusion_xl_base.quasar.tests.sdxl_quasar_test_utils import load_torch_unet, to_device_nhwc
from models.common.utility_functions import torch_random
from models.demos.stable_diffusion_xl_base.quasar.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.quasar.tt.tt_downblock2d import TtDownBlock2D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, temb_shape, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 320, 128, 128), (1, 1280), 0.999),
    ],
)
def test_downblock2d(
    device,
    image_resolution,
    input_shape,
    temb_shape,
    pcc,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    unet = load_torch_unet(sdxl_base_unet_location, is_ci_env, is_ci_v2_env)
    state_dict = unet.state_dict()

    torch_downblock = unet.down_blocks[0]

    model_config = load_model_optimisations(image_resolution)
    tt_downblock = TtDownBlock2D(
        device, state_dict, f"down_blocks.0", model_config=model_config, has_downsample=True, debug_mode=debug_mode
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor, _ = torch_downblock(torch_input_tensor, torch_temb_tensor)

    ttnn_input_tensor, [B, C, H, W] = to_device_nhwc(torch_input_tensor, device)

    ttnn_temb_tensor = qsr.from_torch(
        torch.nn.functional.silu(torch_temb_tensor),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_output_tensor, output_shape, _ = tt_downblock.forward(ttnn_input_tensor, [B, C, H, W], ttnn_temb_tensor)

    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet, tt_downblock
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")
