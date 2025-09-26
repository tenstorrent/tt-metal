# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_scheduler(num_inference_steps=1):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


def constant_prop_time_embeddings(timesteps, batch_size, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(batch_size)
    t_emb = time_proj(timesteps)
    return t_emb


def unsqueeze_all_params_to_4d(params):
    if isinstance(params, dict):
        for key in params.keys():
            params[key] = unsqueeze_all_params_to_4d(params[key])
    elif isinstance(params, ttnn.ttnn.model_preprocessing.ParameterList):
        for i in range(len(params)):
            params[i] = unsqueeze_all_params_to_4d(params[i])
    elif isinstance(params, ttnn.Tensor):
        params = ttnn.unsqueeze_to_4D(params)

    return params


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": SD_L1_SMALL_SIZE}], ids=["device_params=l1_small_size_24576"], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (2, 4, 64, 64),
    ],
)
@pytest.mark.parametrize("num_inference_steps", [1])
def test_unet_2d_condition_model_512x512(
    device, batch_size, in_channels, input_height, input_width, num_inference_steps
):
    ttnn.CONFIG.throw_exception_on_fallback = True
    # setup pytorch model
    torch.manual_seed(0)
    model_name = "CompVis/stable-diffusion-v1-4"
    load_from_disk = False
    if not load_from_disk:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

        model = pipe.unet
        model.eval()
        config = model.config
        torch.save(model, "unet.pt")
        torch.save(config, "unet_config.pt")
    else:
        model = torch.load("unet.pt")
        config = torch.load("unet_config.pt")

    # Create scheduler with configurable inference steps
    scheduler = create_scheduler(num_inference_steps)

    parameters = preprocess_model_parameters(
        model_name=model_name, initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    # unsqueeze weight tensors to 4D for generating perf dump
    parameters = unsqueeze_all_params_to_4d(parameters)

    encoder_hidden_states_shape = [1, 2, 77, 768]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]

    # Initialize latents for both torch and ttnn
    latents_torch = torch.randn(hidden_states_shape)
    latents_ttnn = latents_torch.clone()

    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    # Prepare encoder hidden states for ttnn
    encoder_hidden_states_padded = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states_ttnn = ttnn.from_torch(
        encoder_hidden_states_padded, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    encoder_hidden_states_ttnn = ttnn.to_device(encoder_hidden_states_ttnn, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Initialize ttnn model
    ttnn_model = UNet2D(device, parameters, batch_size, input_height, input_width)

    use_signpost = True
    try:
        from tracy import signpost
    except ModuleNotFoundError:
        use_signpost = False

    # Iterate through all timesteps
    for i, timestep in enumerate(tqdm(scheduler.timesteps)):
        if use_signpost:
            signpost(header="start")

        # Run torch model
        torch_output = model(
            latents_torch, timestep=timestep, encoder_hidden_states=encoder_hidden_states.squeeze(0)
        ).sample

        # Prepare timestep for ttnn
        ttnn_timestep = constant_prop_time_embeddings(timestep, batch_size, model.time_proj)
        ttnn_timestep = ttnn_timestep.unsqueeze(0).unsqueeze(0)
        ttnn_timestep = ttnn_timestep.permute(2, 0, 1, 3)  # pre-permute temb
        ttnn_timestep = ttnn.from_torch(ttnn_timestep, ttnn.bfloat16)
        ttnn_timestep = ttnn.to_device(ttnn_timestep, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn_timestep = ttnn.to_layout(ttnn_timestep, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

        # Prepare latents for ttnn
        latents_ttnn_tensor = ttnn.from_torch(latents_ttnn, ttnn.bfloat16)
        latents_ttnn_tensor = ttnn.to_device(latents_ttnn_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        latents_ttnn_tensor = ttnn.to_layout(latents_ttnn_tensor, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        # Run ttnn model
        ttnn_output = ttnn_model(
            latents_ttnn_tensor,
            timestep=ttnn_timestep,
            encoder_hidden_states=encoder_hidden_states_ttnn,
            class_labels=class_labels,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            config=config,
        )

        if use_signpost:
            signpost(header="stop")

        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        # Apply scheduler step to both outputs for next iteration
        if i < len(scheduler.timesteps) - 1:  # Don't update on last iteration
            latents_torch = scheduler.step(torch_output, timestep, latents_torch, return_dict=False)[0]
            latents_ttnn = scheduler.step(ttnn_output_torch, timestep, latents_ttnn, return_dict=False)[0]

    # Compare final outputs
    assert_with_pcc(torch_output, ttnn_output_torch, 0.995)
