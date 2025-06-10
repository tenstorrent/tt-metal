# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_upblock2d import TtUpDecoderBlock2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape, block_id, pcc",
    [
        ((1, 512, 128, 128), 0, 0.999),
        ((1, 512, 256, 256), 1, 0.993),
        ((1, 512, 512, 512), 2, 0.998),
        ((1, 256, 1024, 1024), 3, 0.999),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 16384}], indirect=True)
def test_vae_upblock(device, input_shape, block_id, pcc, reset_seeds):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="vae"
    )
    vae.eval()
    state_dict = vae.state_dict()

    torch_upblock = vae.decoder.up_blocks[block_id]

    model_config = ModelOptimisations()
    tt_upblock = TtUpDecoderBlock2D(
        device,
        state_dict,
        f"decoder.up_blocks.{block_id}",
        model_config,
        has_upsample=block_id < 3,
        conv_shortcut=block_id > 1,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_upblock(torch_input_tensor, temb=None)

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
    import tracy

    tracy.signpost("Compilation pass")
    ttnn_output_tensor, output_shape = tt_upblock.forward(ttnn_input_tensor, [B, C, H, W])
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del vae
    gc.collect()

    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    print(pcc_passed, pcc_message)
