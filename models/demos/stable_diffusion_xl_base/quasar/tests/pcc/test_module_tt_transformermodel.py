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
from models.demos.stable_diffusion_xl_base.quasar.tt.tt_transformermodel import TtTransformer2DModel
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, encoder_shape, down_block_id, query_dim, num_attn_heads, out_dim, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 640, 64, 64), (1, 77, 2048), 1, 640, 10, 640, 0.998),
        ((1024, 1024), (1, 1280, 32, 32), (1, 77, 2048), 2, 1280, 20, 1280, 0.997),
    ],
)
def test_transformermodel(
    device,
    image_resolution,
    input_shape,
    encoder_shape,
    down_block_id,
    query_dim,
    num_attn_heads,
    out_dim,
    pcc,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    unet = load_torch_unet(sdxl_base_unet_location, is_ci_env, is_ci_v2_env)
    state_dict = unet.state_dict()

    torch_transformerblock = unet.down_blocks[down_block_id].attentions[0]
    model_config = load_model_optimisations(image_resolution)
    tt_transformerblock = TtTransformer2DModel(
        device,
        state_dict,
        f"down_blocks.{down_block_id}.attentions.0",
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_transformerblock(torch_input_tensor, encoder_hidden_states=torch_encoder_tensor).sample

    ttnn_encoder_tensor = qsr.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_input_tensor, [B, C, H, W] = to_device_nhwc(torch_input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn_output_tensor = tt_transformerblock.forward(ttnn_input_tensor, [B, C, H, W], None, ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, H, W, C)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")
