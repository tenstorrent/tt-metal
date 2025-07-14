# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.tt_feedforward import TtFeedForward
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_CI_WEIGHTS_PATH


@pytest.mark.parametrize(
    "input_shape, block_id, transformer_block_id, pcc",
    [
        ((1024, 1280), 2, 0, 0.997),
        ((4096, 640), 1, 0, 0.999),
        ((4096, 640), 1, 1, 0.998),
    ],
)
def test_feedforward(device, input_shape, block_id, transformer_block_id, pcc, is_ci_env, reset_seeds):
    if is_ci_env:
        os.environ["HF_HOME"] = SDXL_CI_WEIGHTS_PATH
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
        local_files_only=is_ci_env,
    )
    unet.eval()
    state_dict = unet.state_dict()

    torch_ff = unet.down_blocks[block_id].attentions[0].transformer_blocks[transformer_block_id].ff

    model_config = ModelOptimisations()
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
