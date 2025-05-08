# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_transformerblock import TtBasicTransformerBlock
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape, encoder_shape, down_block_id, block_id, query_dim, num_attn_heads, out_dim, pcc",
    [
        ((1, 4096, 640), (1, 77, 2048), 1, 0, 640, 10, 640, 0.999),
        ((1, 4096, 640), (1, 77, 2048), 1, 1, 640, 10, 640, 0.999),
        ((1, 1024, 1280), (1, 77, 2048), 2, 0, 1280, 20, 1280, 0.999),
        ((1, 1024, 1280), (1, 77, 2048), 2, 1, 1280, 20, 1280, 0.998),
    ],
)
@pytest.mark.parametrize("transformer_weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_transformerblock(
    device,
    input_shape,
    encoder_shape,
    down_block_id,
    block_id,
    query_dim,
    num_attn_heads,
    out_dim,
    use_program_cache,
    reset_seeds,
    transformer_weights_dtype,
    pcc,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="unet"
    )
    # unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    torch_transformerblock = unet.down_blocks[down_block_id].attentions[0].transformer_blocks[block_id]
    tt_transformerblock = TtBasicTransformerBlock(
        device,
        state_dict,
        f"down_blocks.{down_block_id}.attentions.0.transformer_blocks.{block_id}",
        query_dim,
        num_attn_heads,
        out_dim,
        weights_dtype=transformer_weights_dtype,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_transformerblock(torch_input_tensor, None, torch_encoder_tensor).unsqueeze(0)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_encoder_tensor = ttnn.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_transformerblock.forward(ttnn_input_tensor, None, ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")
