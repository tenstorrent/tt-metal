# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_vae import TtVAEDecoder
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 4, 128, 128),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 16384}], indirect=True)
def test_vae(device, input_shape, reset_seeds):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="vae"
    )
    vae.eval()
    state_dict = vae.state_dict()

    torch_vae = vae.decoder
    tt_vae = TtVAEDecoder(device, state_dict)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_vae(torch_input_tensor)

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

    output_tensor = tt_vae.forward(ttnn_input_tensor, [B, C, H, W])

    del vae
    gc.collect()

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
