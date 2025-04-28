# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_upblock2d import TtUpBlock2D
from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape, temb_shape, residuals, block_id",
    [
        (
            (1, 640, 128, 128),
            (1, 1280),
            ((1, 320, 128, 128), (1, 320, 128, 128), (1, 320, 128, 128)),
            2,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 3 * 16384}], indirect=True)
def test_crossattnup(
    device,
    input_shape,
    temb_shape,
    residuals,
    block_id,
    use_program_cache,
    reset_seeds,
):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16"
    )
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    torch_crosattn = unet.up_blocks[block_id]
    tt_crosattn = TtUpBlock2D(device, state_dict, f"up_blocks.{block_id}")
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)

    torch_residual_tensors = ()
    for r in residuals:
        residual = torch_random(r, -0.1, 0.1, dtype=torch.float32)
        torch_residual_tensors = torch_residual_tensors + (residual,)

    torch_output_tensor = torch_crosattn(torch_input_tensor, torch_residual_tensors, temb=torch_temb_tensor)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_residual_tensors = ()
    for torch_residual in torch_residual_tensors:
        ttnn_residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        Br, Cr, Hr, Wr = list(ttnn_residual.shape)
        ttnn_residual = ttnn.permute(ttnn_residual, (0, 2, 3, 1))
        ttnn_residual = ttnn.reshape(ttnn_residual, (Br, 1, Hr * Wr, Cr))
        ttnn_residual_tensors = ttnn_residual_tensors + (ttnn_residual,)

    ttnn_temb_tensor = ttnn.from_torch(torch_temb_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_output_tensor, output_shape = tt_crosattn.forward(
        ttnn_input_tensor,
        ttnn_residual_tensors,
        input_shape=[B, C, H, W],
        temb=ttnn_temb_tensor,
    )
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor, 0.96)
