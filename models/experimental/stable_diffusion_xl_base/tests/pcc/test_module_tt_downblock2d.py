# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.ttnn_impl.tt_downblock2d import TtDownBlock2D
from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape, temb_shape",
    [
        ((1, 320, 128, 128), (1, 1280)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_downblock2d(device, temb_shape, input_shape, use_program_cache):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16"
    )
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    torch_downblock = unet.down_blocks[0]
    tt_downblock = TtDownBlock2D(device, state_dict, f"down_blocks.0")

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor, _ = torch_downblock(torch_input_tensor, torch_temb_tensor)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_temb_tensor = ttnn.from_torch(torch_temb_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_output_tensor, output_shape = tt_downblock.forward(ttnn_input_tensor, ttnn_temb_tensor, [B, C, H, W])
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor, 0.994)
