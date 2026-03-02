# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.experimental.stable_diffusion_xl_base.tt.tt_feedforward import TtFeedForward
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random, is_blackhole


@pytest.mark.parametrize(
    "image_resolution, input_shape, block_id, transformer_block_id, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1024, 1280), 2, 0, 0.997),
        ((1024, 1024), (4096, 640), 1, 0, 0.999),
        # 512x512 image resolution - skip on Blackhole
        pytest.param(
            (512, 512),
            (256, 1280),
            2,
            0,
            0.997,
            marks=pytest.mark.skipif(is_blackhole(), reason="512x512 not supported on Blackhole"),
        ),
        pytest.param(
            (512, 512),
            (1024, 640),
            1,
            0,
            0.999,
            marks=pytest.mark.skipif(is_blackhole(), reason="512x512 not supported on Blackhole"),
        ),
    ],
)
def test_feedforward(
    device,
    image_resolution,
    input_shape,
    block_id,
    transformer_block_id,
    pcc,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_base_unet_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env,
        subfolder=None if is_ci_v2_env else "unet",
    )
    unet.eval()
    state_dict = unet.state_dict()

    torch_ff = unet.down_blocks[block_id].attentions[0].transformer_blocks[transformer_block_id].ff

    model_config = load_model_optimisations(image_resolution)
    tt_ff = TtFeedForward(
        device,
        state_dict,
        f"down_blocks.{block_id}.attentions.0.transformer_blocks.{transformer_block_id}.ff",
        model_config,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_ff(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_ff.forward(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor).squeeze()

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")
