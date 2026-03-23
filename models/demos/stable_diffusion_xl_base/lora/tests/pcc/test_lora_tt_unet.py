# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from diffusers import DiffusionPipeline
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0, torch_random
from models.demos.stable_diffusion_xl_base.lora.tt_lora_weights_manager import TtLoRAWeightsManager
from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc


def _get_diffusers_pipeline(is_ci_env):
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )
    return pipeline


def prepare_ttnn_tensors(
    device, torch_input_tensor, torch_timestep_tensor, torch_temb_tensor, torch_encoder_tensor, torch_time_ids
):
    torch.manual_seed(2025)

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
        layout=ttnn.ROW_MAJOR_LAYOUT,
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
    lora_path,
    iterations=1,
):
    assert not (is_ci_v2_env and input_shape[1] != 4), "Currently only vanilla SDXL UNet is supported in CI v2"
    pipeline = _get_diffusers_pipeline(is_ci_env)
    pipeline.unet.eval()

    pipeline_for_tt = _get_diffusers_pipeline(is_ci_env)
    state_dict = pipeline_for_tt.unet.state_dict()

    lora_mgr = TtLoRAWeightsManager(device, pipeline_for_tt)
    model_config = load_model_optimisations(image_resolution)
    tt_unet = TtUNet2DConditionModel(
        device,
        state_dict,
        "unet",
        model_config=model_config,
        debug_mode=debug_mode,
        lora_weights_manager=lora_mgr,
    )

    lora_mgr.load_lora_weights(lora_path)
    lora_mgr.fuse_lora(lora_scale=1.0)
    pipeline.load_lora_weights(lora_path)
    pipeline.fuse_lora(lora_scale=1.0)

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_timestep_tensor = torch_random(timestep_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)
    torch_time_ids = torch.tensor([1024, 1024, 0, 0, 1024, 1024])

    added_cond_kwargs = {
        "text_embeds": torch_temb_tensor,
        "time_ids": torch_time_ids,
    }

    torch_output_tensor = pipeline.unet(
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

    del pipeline, pipeline_for_tt, tt_unet, lora_mgr
    gc.collect()


@pytest.mark.parametrize(
    "image_resolution, input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape, pcc",
    [
        ((1024, 1024), (1, 4, 128, 128), (1,), (1, 77, 2048), (1, 1280), (1, 6), 0.9969 if is_wormhole_b0() else 0.996),
        ((512, 512), (1, 4, 64, 64), (1,), (1, 77, 2048), (1, 1280), (1, 6), 0.9958 if is_wormhole_b0() else 0.995),
        # TODO: Add test for 9x128x128 input shape if needed (inpainting)
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
    reset_seeds,
    lora_path,
):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")

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
        lora_path,
    )
