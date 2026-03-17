# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from diffusers import AutoencoderKL
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, torch_random
from models.demos.stable_diffusion_xl_base.vae.tt.model_configs import load_vae_model_optimisations
from models.demos.stable_diffusion_xl_base.vae.tt.tt_attention import TtAttention
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, encoder_shape",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 512, 128, 128), None),
        # 512x512 image resolution
        ((512, 512), (1, 512, 64, 64), None),
    ],
)
@pytest.mark.parametrize(
    "block_name, pcc",
    [
        ("encoder", 0.999),
        ("decoder", 0.997),
    ],
)
def test_vae_attention(
    device,
    image_resolution,
    input_shape,
    encoder_shape,
    block_name,
    pcc,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    reset_seeds,
):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")
    vae = AutoencoderKL.from_pretrained(
        sdxl_base_vae_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "vae",
    )
    vae.eval()
    state_dict = vae.state_dict()

    model_config = load_vae_model_optimisations(image_resolution)
    if block_name == "encoder":
        torch_attention = vae.encoder.mid_block.attentions[0]
        tt_attention = TtAttention(
            device, state_dict, "encoder.mid_block.attentions.0", model_config, 512, 1, 512, None, 512
        )
    else:
        torch_attention = vae.decoder.mid_block.attentions[0]
        tt_attention = TtAttention(
            device, state_dict, "decoder.mid_block.attentions.0", model_config, 512, 1, 512, None, 512
        )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = (
        torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32) if encoder_shape is not None else None
    )

    torch_output_tensor = torch_attention(torch_input_tensor, torch_encoder_tensor)

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

    ttnn_encoder_tensor = (
        ttnn.from_torch(
            torch_encoder_tensor,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if encoder_shape is not None
        else None
    )
    ttnn_output_tensor = tt_attention.forward(ttnn_input_tensor, [B, C, H, W], ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, H, W, C)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del vae, tt_attention
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")
