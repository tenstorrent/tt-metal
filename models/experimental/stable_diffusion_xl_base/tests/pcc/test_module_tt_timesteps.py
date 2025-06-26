# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_timesteps import TtTimesteps
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from loguru import logger


@pytest.mark.parametrize(
    "input_shape, module_path, num_channels", [((1,), "time_proj", 320), ((6,), "add_time_proj", 256)]
)
def test_timesteps(device, input_shape, module_path, num_channels, reset_seeds):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="unet"
    )
    # unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    torch_timesteps = eval("unet." + module_path)
    assert torch_timesteps is not None, f"{module_path} is not a valid UNet module"

    tt_timesteps = TtTimesteps(device, num_channels, True, 0, 1)

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_timesteps(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_timesteps.forward(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"PCC is: {pcc_message}")
