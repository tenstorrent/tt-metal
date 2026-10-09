# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.quasar.tt import qsr
from models.demos.stable_diffusion_xl_base.quasar.tests.sdxl_quasar_test_utils import load_torch_unet
from models.common.utility_functions import torch_random
from models.demos.stable_diffusion_xl_base.quasar.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.quasar.tt.tt_unet import TtUNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc


def prepare_ttnn_tensors(
    device, torch_input_tensor, torch_timestep_tensor, torch_temb_tensor, torch_encoder_tensor, torch_time_ids
):
    torch.manual_seed(2025)

    ttnn_timestep_tensor = qsr.from_torch(
        torch_timestep_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_encoder_tensor = qsr.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_text_embeds = qsr.from_torch(
        torch_temb_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_time_ids = qsr.from_torch(
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

    ttnn_input_tensor, [B, C, H, W] = to_device_nhwc(torch_input_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    return ttnn_input_tensor, [B, C, H, W], ttnn_timestep_tensor, ttnn_encoder_tensor, ttnn_added_cond_kwargs


def run_unet_model(
    device,
    image_resolution,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    pcc,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    sdxl_inpainting_unet_location,
    iterations=1,
):
    # Select model location based on input channels
    if input_shape[1] == 4:
        model_location = sdxl_base_unet_location
    else:
        model_location = sdxl_inpainting_unet_location

    unet = load_torch_unet(model_location, is_ci_env, is_ci_v2_env)
    state_dict = unet.state_dict()

    torch_unet = unet

    model_config = load_model_optimisations(image_resolution)
    tt_unet = TtUNet2DConditionModel(
        device,
        state_dict,
        "unet",
        model_config=model_config,
        debug_mode=debug_mode,
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
    ttnn_output_tensor, output_shape = tt_unet.forward(
        ttnn_input_tensor,
        [B, C, H, W],
        timestep=ttnn_timestep_tensor,
        encoder_hidden_states=ttnn_encoder_tensor,
        time_ids=ttnn_added_cond_kwargs["time_ids"],
        text_embeds=ttnn_added_cond_kwargs["text_embeds"],
    )

    output_tensor = ttnn.to_torch(ttnn_output_tensor.cpu())
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    ttnn.deallocate(ttnn_input_tensor)
    ttnn.deallocate(ttnn_output_tensor)
    ttnn.deallocate(ttnn_timestep_tensor)
    ttnn.deallocate(ttnn_encoder_tensor)
    ttnn.deallocate(ttnn_added_cond_kwargs["text_embeds"])
    ttnn.deallocate(ttnn_added_cond_kwargs["time_ids"])

    ttnn.ReadDeviceProfiler(device)

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC of first iteration is: {pcc_message}")

    for _ in range(iterations - 1):
        (
            ttnn_input_tensor,
            [B, C, H, W],
            ttnn_timestep_tensor,
            ttnn_encoder_tensor,
            ttnn_added_cond_kwargs,
        ) = prepare_ttnn_tensors(
            device, torch_input_tensor, torch_timestep_tensor, torch_temb_tensor, torch_encoder_tensor, torch_time_ids
        )
        ttnn_output_tensor, output_shape = tt_unet.forward(
            ttnn_input_tensor,
            [B, C, H, W],
            timestep=ttnn_timestep_tensor,
            encoder_hidden_states=ttnn_encoder_tensor,
            time_ids=ttnn_added_cond_kwargs["time_ids"],
            text_embeds=ttnn_added_cond_kwargs["text_embeds"],
        )
        ttnn.deallocate(ttnn_input_tensor)
        ttnn.deallocate(ttnn_output_tensor)
        ttnn.deallocate(ttnn_timestep_tensor)
        ttnn.deallocate(ttnn_encoder_tensor)
        ttnn.deallocate(ttnn_added_cond_kwargs["text_embeds"])
        ttnn.deallocate(ttnn_added_cond_kwargs["time_ids"])

        ttnn.ReadDeviceProfiler(device)

    del unet
    gc.collect()


@pytest.mark.parametrize(
    "image_resolution, input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 4, 128, 128), (1,), (1, 77, 2048), (1, 1280), (1, 6), 0.9968),
        ((1024, 1024), (1, 9, 128, 128), (1,), (1, 77, 2048), (1, 1280), (1, 6), 0.9968),
    ],
)
def test_unet(
    device,
    image_resolution,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    pcc,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    sdxl_inpainting_unet_location,
    reset_seeds,
):
    run_unet_model(
        device,
        image_resolution,
        input_shape,
        timestep_shape,
        encoder_shape,
        temb_shape,
        time_ids_shape,
        pcc,
        debug_mode,
        is_ci_env,
        is_ci_v2_env,
        sdxl_base_unet_location,
        sdxl_inpainting_unet_location,
    )
