# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_refiner.tt.tt_feedforward import TtFeedForward
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random
from functools import reduce


@pytest.mark.parametrize(
    "input_shape, module_path, pcc",
    [
        ((1024, 1536), "down_blocks.2.attentions.0.transformer_blocks.0.ff", 0.999),
        ((256, 1536), "up_blocks.1.attentions.0.transformer_blocks.0.ff", 0.999),
        ((4096, 768), "down_blocks.1.attentions.0.transformer_blocks.0.ff", 0.999),
    ],
)
def test_feedforward_refiner(device, input_shape, module_path, pcc, is_ci_env, reset_seeds):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
        local_files_only=is_ci_env,
    )
    unet.eval()
    state_dict = unet.state_dict()

    try:
        torch_ff = reduce(
            lambda obj, key: obj[int(key)] if key.isdigit() else getattr(obj, key), module_path.split("."), unet
        )
    except (AttributeError, IndexError, TypeError) as e:
        torch_ff = None

    assert torch_ff is not None, f"{module_path} is not a valid UNet module"

    tt_ff = TtFeedForward(
        device,
        state_dict,
        module_path,
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
