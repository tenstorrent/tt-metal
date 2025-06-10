# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_midblock2d import TtUNetMidBlock2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 512, 128, 128),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vae_midblock(device, input_shape, use_program_cache, reset_seeds):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="vae"
    )
    vae.eval()
    state_dict = vae.state_dict()

    torch_midblock = vae.decoder.mid_block

    model_config = ModelOptimisations()
    tt_midblock = TtUNetMidBlock2D(device, state_dict, "decoder.mid_block", model_config=model_config)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_midblock(torch_input_tensor, temb=None)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_output_tensor, output_shape = tt_midblock.forward(ttnn_input_tensor, [B, C, H, W])
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del vae
    gc.collect()

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
