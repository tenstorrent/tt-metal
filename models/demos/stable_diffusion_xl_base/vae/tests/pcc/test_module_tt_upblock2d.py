# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from diffusers import AutoencoderKL

import ttnn
from models.common.utility_functions import is_blackhole, torch_random
from models.demos.stable_diffusion_xl_base.vae.tt.model_configs import load_vae_model_optimisations
from models.demos.stable_diffusion_xl_base.vae.tt.tt_upblock2d import TtUpDecoderBlock2D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, block_id, pcc",
    [
        # NOTE: https://github.com/tenstorrent/tt-metal/issues/39225
        # PCC lowered for some input until the mentioned issue with DRAM GN is resolved
        # TODO: once DRAM GN is fixed, update the PCC values accordingly and remove skips
        # 1024x1024 image resolution
        ((1024, 1024), (1, 512, 128, 128), 0, 0.999),  # skipped; PCC=0.74
        ((1024, 1024), (1, 512, 256, 256), 1, 0.995),  # skipped; PCC=0.62
        ((1024, 1024), (1, 512, 512, 512), 2, 0.998),  # skipped; PCC=0.59
        ((1024, 1024), (1, 256, 1024, 1024), 3, 0.999 if not is_blackhole() else 0.99),
        # 512x512 image resolution
        ((512, 512), (1, 512, 64, 64), 0, 0.999),
        ((512, 512), (1, 512, 128, 128), 1, 0.995),
        ((512, 512), (1, 512, 256, 256), 2, 0.998),
        ((512, 512), (1, 256, 512, 512), 3, 0.999),
    ],
)
def test_vae_upblock(
    device,
    image_resolution,
    input_shape,
    block_id,
    pcc,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    reset_seeds,
):
    if is_blackhole():
        if image_resolution == (512, 512):
            pytest.skip("512x512 resolution not supported on Blackhole")

        if input_shape != (1, 256, 1024, 1024):
            pytest.skip("Skipping on Blackhole due to PCC issue with DRAM group_norm")
    vae = AutoencoderKL.from_pretrained(
        sdxl_base_vae_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "vae",
    )
    vae.eval()
    state_dict = vae.state_dict()

    torch_upblock = vae.decoder.up_blocks[block_id]

    model_config = load_vae_model_optimisations(image_resolution)
    tt_upblock = TtUpDecoderBlock2D(
        device,
        state_dict,
        f"decoder.up_blocks.{block_id}",
        model_config,
        has_upsample=block_id < 3,
        conv_shortcut=block_id > 1,
        debug_mode=debug_mode,
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
