# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from diffusers import UNet2DConditionModel
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, torch_random
from models.demos.stable_diffusion_xl_base.refiner.tt.model_configs import load_refiner_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_transformerblock import TtBasicTransformerBlock
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, encoder_shape, down_block_id, block_id, query_dim, num_attn_heads, out_dim, pcc, block_type",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 4096, 768), (1, 77, 1280), 1, 0, 768, 12, 768, 0.999, "down_blocks"),
        ((1024, 1024), (1, 4096, 768), (1, 77, 1280), 1, 1, 768, 12, 768, 0.997, "down_blocks"),
        ((1024, 1024), (1, 1024, 1536), (1, 77, 1280), 2, 0, 1536, 24, 1536, 0.998, "down_blocks"),
        ((1024, 1024), (1, 1024, 1536), (1, 77, 1280), 2, 1, 1536, 24, 1536, 0.997, "down_blocks"),
        ((1024, 1024), (1, 256, 1536), (1, 77, 1280), -1, 0, 1536, 24, 1536, 0.998, "mid_block"),
        ((1024, 1024), (1, 256, 1536), (1, 77, 1280), -1, 1, 1536, 24, 1536, 0.997, "mid_block"),
        # 512x512 image resolution
        ((512, 512), (1, 1024, 768), (1, 77, 1280), 1, 0, 768, 12, 768, 0.999, "down_blocks"),
        ((512, 512), (1, 1024, 768), (1, 77, 1280), 1, 1, 768, 12, 768, 0.997, "down_blocks"),
        ((512, 512), (1, 256, 1536), (1, 77, 1280), 2, 0, 1536, 24, 1536, 0.998, "down_blocks"),
        ((512, 512), (1, 256, 1536), (1, 77, 1280), 2, 1, 1536, 24, 1536, 0.997, "down_blocks"),
        ((512, 512), (1, 64, 1536), (1, 77, 1280), -1, 0, 1536, 24, 1536, 0.997, "mid_block"),
        ((512, 512), (1, 64, 1536), (1, 77, 1280), -1, 1, 1536, 24, 1536, 0.997, "mid_block"),
    ],
)
def test_transformerblock(
    device,
    image_resolution,
    input_shape,
    encoder_shape,
    down_block_id,
    block_id,
    query_dim,
    num_attn_heads,
    out_dim,
    pcc,
    block_type,
    is_ci_env,
    is_ci_v2_env,
    sdxl_refiner_unet_location,
    reset_seeds,
):
    if is_blackhole():
        pytest.skip("Not supported on Blackhole")
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_refiner_unet_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "unet",
    )
    unet.eval()
    state_dict = unet.state_dict()

    if block_type == "down_blocks":
        torch_transformerblock = unet.down_blocks[down_block_id].attentions[0].transformer_blocks[block_id]
        block_type = f"down_blocks.{down_block_id}"
    elif block_type == "mid_block":
        torch_transformerblock = unet.mid_block.attentions[0].transformer_blocks[block_id]
    else:
        raise ValueError(f"Unknown block_type: {block_type}")

    model_config = load_refiner_model_optimisations(image_resolution)
    tt_transformerblock = TtBasicTransformerBlock(
        device,
        state_dict,
        f"{block_type}.attentions.0.transformer_blocks.{block_id}",
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_transformerblock(torch_input_tensor, None, torch_encoder_tensor).unsqueeze(0)

    ttnn_encoder_tensor = ttnn.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_transformerblock.forward(ttnn_input_tensor, None, ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")
