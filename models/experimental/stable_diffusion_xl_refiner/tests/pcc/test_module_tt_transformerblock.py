# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_refiner.tt.tt_transformerblock import TtBasicTransformerBlock
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_refiner.tests.test_common import SDXL_REFINER_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, encoder_shape, block_id, transformer_block_id, pcc, block_type",
    [
        ((1, 1024, 1536), (1, 77, 1280), 2, 0, 0.999, "down_blocks"),
        ((1, 4096, 768), (1, 77, 1280), 1, 0, 0.999, "down_blocks"),
        ((1, 256, 1536), (1, 77, 1280), 0, 0, 0.999, "mid_block"),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_REFINER_L1_SMALL_SIZE}], indirect=True)
def test_transformerblock_refiner(
    device,
    input_shape,
    encoder_shape,
    block_id,
    transformer_block_id,
    pcc,
    block_type,
    is_ci_env,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
        local_files_only=is_ci_env,
    )
    unet.eval()
    state_dict = unet.state_dict()

    if block_type == "down_blocks":
        torch_transformerblock = unet.down_blocks[block_id].attentions[0].transformer_blocks[transformer_block_id]
        module_path = f"down_blocks.{block_id}.attentions.0.transformer_blocks.{transformer_block_id}"
    elif block_type == "mid_block":
        torch_transformerblock = unet.mid_block.attentions[0].transformer_blocks[transformer_block_id]
        module_path = f"mid_block.attentions.0.transformer_blocks.{transformer_block_id}"

    tt_transformerblock = TtBasicTransformerBlock(
        device,
        state_dict,
        module_path,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_transformerblock(torch_input_tensor, None, torch_encoder_tensor).unsqueeze(0)

    ttnn_encoder_tensor = ttnn.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_transformerblock.forward(ttnn_input_tensor, ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")
