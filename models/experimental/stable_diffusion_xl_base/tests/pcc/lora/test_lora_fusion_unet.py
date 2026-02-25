# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.lora.lora_weights_manager import TtLoRAWeightsManager
from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE

LORA_PATH = "lora_weights/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"


def _get_base_state_dict_and_pipeline_with_lora(is_ci_env):
    """Load pipeline, get base UNet state_dict, load LoRA (do not fuse yet). Returns (base_state_dict, pipeline)."""
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )
    pipeline.unet.eval()
    base_state_dict = {k: v.clone() for k, v in pipeline.unet.state_dict().items()}
    pipeline.load_lora_weights(LORA_PATH)
    return base_state_dict, pipeline


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
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    model_location_generator,
    iterations=1,
):
    assert not (is_ci_v2_env and input_shape[1] != 4), "Currently only vanilla SDXL UNet is supported in CI v2"
    base_state_dict, pipeline = _get_base_state_dict_and_pipeline_with_lora(is_ci_env)
    torch_unet = pipeline.unet

    lora_mgr = TtLoRAWeightsManager(device, pipeline)
    model_config = ModelOptimisations()
    tt_unet = TtUNet2DConditionModel(
        device,
        base_state_dict,
        "unet",
        model_config=model_config,
        debug_mode=debug_mode,
        lora_weights_manager=lora_mgr,
    )
    lora_mgr.fuse_lora_weights(lora_scale=1.0)
    pipeline.fuse_lora()

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

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.997)
    logger.info(f"LoRA UNet PCC of first iteration is: {pcc_message}")

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

    del pipeline, tt_unet, lora_mgr
    gc.collect()


@pytest.mark.parametrize(
    "input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape",
    [
        ((1, 4, 128, 128), (1,), (1, 77, 2048), (1, 1280), (1, 6)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_unet(
    device,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    model_location_generator,
    reset_seeds,
):
    run_unet_model(
        device,
        input_shape,
        timestep_shape,
        encoder_shape,
        temb_shape,
        time_ids_shape,
        debug_mode,
        is_ci_env,
        is_ci_v2_env,
        model_location_generator,
    )
