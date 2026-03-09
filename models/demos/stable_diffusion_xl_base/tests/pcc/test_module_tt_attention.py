# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from diffusers import UNet2DConditionModel
from loguru import logger

import ttnn
from models.common.utility_functions import torch_random, is_blackhole
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_attention import TtAttention
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, encoder_shape, down_block_id, attn_id, query_dim, num_attn_heads, out_dim",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 4096, 640), None, 1, 1, 640, 10, 640),
        ((1024, 1024), (1, 4096, 640), (1, 77, 2048), 1, 2, 640, 10, 640),
        ((1024, 1024), (1, 1024, 1280), None, 2, 1, 1280, 20, 1280),
        ((1024, 1024), (1, 1024, 1280), (1, 77, 2048), 2, 2, 1280, 20, 1280),
        # 512x512 image resolution
        ((512, 512), (1, 1024, 640), None, 1, 1, 640, 10, 640),
        ((512, 512), (1, 1024, 640), (1, 77, 2048), 1, 2, 640, 10, 640),
        ((512, 512), (1, 256, 1280), None, 2, 1, 1280, 20, 1280),
        ((512, 512), (1, 256, 1280), (1, 77, 2048), 2, 2, 1280, 20, 1280),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_attention(
    device,
    image_resolution,
    input_shape,
    encoder_shape,
    down_block_id,
    attn_id,
    query_dim,
    num_attn_heads,
    out_dim,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_base_unet_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "unet",
    )
    unet.eval()
    state_dict = unet.state_dict()

    if attn_id == 1:
        torch_attention = unet.down_blocks[down_block_id].attentions[0].transformer_blocks[0].attn1
    else:
        torch_attention = unet.down_blocks[down_block_id].attentions[0].transformer_blocks[0].attn2
    model_config = load_model_optimisations(image_resolution)
    tt_attention = TtAttention(
        device,
        state_dict,
        f"down_blocks.{down_block_id}.attentions.0.transformer_blocks.0.attn{attn_id}",
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = (
        torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32) if encoder_shape is not None else None
    )

    torch_output_tensor = torch_attention(torch_input_tensor, torch_encoder_tensor).unsqueeze(0)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_encoder_tensor = (
        ttnn.from_torch(
            torch_encoder_tensor,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if encoder_shape is not None
        else None
    )
    ttnn_output_tensor = tt_attention.forward(ttnn_input_tensor, None, ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del unet, tt_attention
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"PCC is: {pcc_message}")
