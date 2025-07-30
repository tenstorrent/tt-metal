# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_crossattndownblock2d import TtCrossAttnDownBlock2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, temb_shape, encoder_shape, query_dim, num_attn_heads, out_dim, down_block_id, pcc",
    [
        ((1, 320, 64, 64), (1, 1280), (1, 77, 2048), 640, 10, 640, 1, 0.996),
        ((1, 640, 32, 32), (1, 1280), (1, 77, 2048), 1280, 20, 1280, 2, 0.993),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_crossattndown(
    device,
    input_shape,
    temb_shape,
    encoder_shape,
    query_dim,
    num_attn_heads,
    out_dim,
    down_block_id,
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

    torch_crosattn = unet.down_blocks[down_block_id]

    model_config = ModelOptimisations()
    tt_crosattn = TtCrossAttnDownBlock2D(
        device,
        state_dict,
        f"down_blocks.{down_block_id}",
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
        down_block_id == 1,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor, _ = torch_crosattn(
        torch_input_tensor, temb=torch_temb_tensor, encoder_hidden_states=torch_encoder_tensor
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

    ttnn_temb_tensor = ttnn.from_torch(
        torch_temb_tensor,
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
    ttnn_output_tensor, output_shape, _ = tt_crosattn.forward(
        ttnn_input_tensor, [B, C, H, W], temb=ttnn_temb_tensor, encoder_hidden_states=ttnn_encoder_tensor
    )

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet, tt_crosattn
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")
