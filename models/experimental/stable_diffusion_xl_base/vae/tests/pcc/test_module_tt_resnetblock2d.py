# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape, block_id, resnet_id, conv_shortcut, block, pcc",
    [
        ((1, 512, 128, 128), 0, 0, False, "mid_block", 0.999),
        ((1, 512, 128, 128), 0, 0, False, "up_blocks", 0.999),
        ((1, 512, 256, 256), 1, 0, False, "up_blocks", 0.999),
        ((1, 512, 256, 256), 2, 0, True, "up_blocks", 0.999),
        ((1, 256, 256, 256), 2, 1, False, "up_blocks", 0.999),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2 * 16384}], indirect=True)
def test_vae_resnetblock2d(
    device, input_shape, block_id, resnet_id, conv_shortcut, block, pcc, use_program_cache, reset_seeds
):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="vae"
    )
    vae.eval()
    state_dict = vae.state_dict()

    if block == "up_blocks":
        torch_resnet = vae.decoder.up_blocks[block_id].resnets[resnet_id]
        block = f"{block}.{block_id}"
    else:
        torch_resnet = vae.decoder.mid_block.resnets[resnet_id]

    model_config = ModelOptimisations()
    tt_resnet = TtResnetBlock2D(device, state_dict, f"decoder.{block}.resnets.{resnet_id}", model_config, conv_shortcut)

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_resnet(torch_input_tensor, None)

    B, C, H, W = input_shape
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_tensor = torch_input_tensor.reshape(B, 1, H * W, C)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output_tensor, output_shape = tt_resnet.forward(ttnn_input_tensor, [B, C, H, W])
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del vae
    gc.collect()

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)
