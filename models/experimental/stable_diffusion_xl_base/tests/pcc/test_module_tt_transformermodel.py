# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.tt_transformermodel import TtTransformer2DModel
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, encoder_shape, down_block_id, query_dim, num_attn_heads, out_dim, pcc",
    [
        ((1, 640, 64, 64), (1, 77, 2048), 1, 640, 10, 640, 0.998),
        ((1, 1280, 32, 32), (1, 77, 2048), 2, 1280, 20, 1280, 0.996),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_transformermodel(
    device,
    input_shape,
    encoder_shape,
    down_block_id,
    query_dim,
    num_attn_heads,
    out_dim,
    pcc,
    is_ci_env,
    reset_seeds,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
        local_files_only=is_ci_env,
    )
    unet.eval()
    state_dict = unet.state_dict()

    torch_transformerblock = unet.down_blocks[down_block_id].attentions[0]
    model_config = ModelOptimisations()
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

    ttnn_encoder_tensor = ttnn.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

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

    ttnn_output_tensor = tt_transformerblock.forward(ttnn_input_tensor, [B, C, H, W], None, ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, H, W, C)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")
