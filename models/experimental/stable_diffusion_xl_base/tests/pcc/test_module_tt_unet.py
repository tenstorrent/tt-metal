# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


def prepare_ttnn_tensors(
    device, torch_input_tensor, torch_timestep_tensor, torch_temb_tensor, torch_encoder_tensor, torch_time_ids
):
    torch.manual_seed(2025)
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

    ttnn_timestep_tensor = ttnn.from_torch(
        torch_timestep_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_encoder_tensor = ttnn.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_text_embeds = ttnn.from_torch(
        torch_temb_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_time_ids = ttnn.from_torch(
        torch_time_ids,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_added_cond_kwargs = {
        "text_embeds": ttnn_text_embeds,
        "time_ids": ttnn_time_ids,
    }

    return ttnn_input_tensor, [B, C, H, W], ttnn_timestep_tensor, ttnn_encoder_tensor, ttnn_added_cond_kwargs


@pytest.mark.parametrize(
    "input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape",
    [
        ((1, 4, 128, 128), (1,), (1, 77, 2048), (1, 1280), (1, 6)),
    ],
)
@pytest.mark.parametrize("conv_weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("transformer_weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 16384}], indirect=True)
def test_unet(
    device,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    use_program_cache,
    reset_seeds,
    conv_weights_dtype,
    transformer_weights_dtype,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="unet"
    )
    # unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    torch_unet = unet
    tt_unet = TtUNet2DConditionModel(
        device,
        state_dict,
        "unet",
        transformer_weights_dtype=transformer_weights_dtype,
        conv_weights_dtype=conv_weights_dtype,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_timestep_tensor = torch_random(timestep_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)
    torch_time_ids = torch.tensor([1024, 1024, 0, 0, 1024, 1024])

    added_cond_kwargs = {
        "text_embeds": torch_temb_tensor,
        "time_ids": torch_time_ids,
    }

    torch_output_tensor = torch_unet(
        torch_input_tensor,
        timestep=torch_timestep_tensor,
        encoder_hidden_states=torch_encoder_tensor,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    (
        ttnn_input_tensor,
        [B, C, H, W],
        ttnn_timestep_tensor,
        ttnn_encoder_tensor,
        ttnn_added_cond_kwargs,
    ) = prepare_ttnn_tensors(
        device, torch_input_tensor, torch_timestep_tensor, torch_temb_tensor, torch_encoder_tensor, torch_time_ids
    )

    # import tracy

    # tracy.signpost("Compilation pass")
    _, _ = tt_unet.forward(
        ttnn_input_tensor,
        [B, C, H, W],
        timestep=ttnn_timestep_tensor,
        encoder_hidden_states=ttnn_encoder_tensor,
        added_cond_kwargs=ttnn_added_cond_kwargs,
    )

    (
        ttnn_input_tensor,
        [B, C, H, W],
        ttnn_timestep_tensor,
        ttnn_encoder_tensor,
        ttnn_added_cond_kwargs,
    ) = prepare_ttnn_tensors(
        device, torch_input_tensor, torch_timestep_tensor, torch_temb_tensor, torch_encoder_tensor, torch_time_ids
    )

    # tracy.signpost("Second pass")
    ttnn_output_tensor, output_shape = tt_unet.forward(
        ttnn_input_tensor,
        [B, C, H, W],
        timestep=ttnn_timestep_tensor,
        encoder_hidden_states=ttnn_encoder_tensor,
        added_cond_kwargs=ttnn_added_cond_kwargs,
    )

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.988)
    logger.info(f"PCC is: {pcc_message}")
