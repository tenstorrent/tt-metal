# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.tt_attention import TtAttention
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, encoder_shape, attn_id, down_block_id, query_dim, num_attn_heads, out_dim",
    [
        ((1, 4096, 640), None, 1, 1, 640, 10, 640),
        ((1, 4096, 640), (1, 77, 2048), 2, 1, 640, 10, 640),
        ((1, 1024, 1280), None, 1, 2, 1280, 20, 1280),
        ((1, 1024, 1280), (1, 77, 2048), 2, 2, 1280, 20, 1280),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_attention(
    device,
    input_shape,
    encoder_shape,
    attn_id,
    down_block_id,
    query_dim,
    num_attn_heads,
    out_dim,
    reset_seeds,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="unet"
    )
    unet.eval()
    state_dict = unet.state_dict()

    if attn_id == 1:
        torch_attention = unet.down_blocks[down_block_id].attentions[0].transformer_blocks[0].attn1
    else:
        torch_attention = unet.down_blocks[down_block_id].attentions[0].transformer_blocks[0].attn2
    model_config = ModelOptimisations()
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
        torch_input_tensor,
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
