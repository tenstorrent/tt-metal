# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from models.utility_functions import skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)

scheduler.set_timesteps(1)


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


@skip_for_blackhole("Failing on harvested BH, see #21088")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], ids=["device_params=l1_small_size_24576"], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (2, 4, 64, 64),
    ],
)
def test_unet_2d_condition_model_512x512(device, batch_size, in_channels, input_height, input_width, use_program_cache):
    # setup envvar if testing on N300
    wh_arch_yaml_org = None
    if device.core_grid.y == 7:
        if ("WH_ARCH_YAML" not in os.environ) or (
            os.environ["WH_ARCH_YAML"] != "wormhole_b0_80_arch_eth_dispatch.yaml"
        ):
            pytest.skip("SD unet2d only works for 8x8 grid size")

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

    parameters = preprocess_model_parameters(
        model_name=model_name, initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    # unsqueeze weight tensors to 4D for generating perf dump
    parameters = unsqueeze_all_params_to_4d(parameters)

    timestep_shape = [1, 1, 2, 320]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]

    input = torch.randn(hidden_states_shape)
    timestep = [i for i in tqdm(scheduler.timesteps)][0]
    ttnn_timestep = constant_prop_time_embeddings(timestep, batch_size, model.time_proj)
    ttnn_timestep = ttnn_timestep.unsqueeze(0).unsqueeze(0)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = model(input, timestep=timestep, encoder_hidden_states=encoder_hidden_states.squeeze(0)).sample
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_timestep = ttnn_timestep.permute(2, 0, 1, 3)  # pre-permute temb
    ttnn_timestep = ttnn.from_torch(ttnn_timestep, ttnn.bfloat16)
    ttnn_timestep = ttnn.to_device(ttnn_timestep, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_timestep = ttnn.to_layout(ttnn_timestep, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    model = UNet2D(device, parameters, batch_size, input_height, input_width)

    first_iter = time.time()
    use_signpost = True
    try:
        from tracy import signpost
    except ModuleNotFoundError:
        use_signpost = False
    if use_signpost:
        signpost(header="start")
    ttnn_output_ = model(
        input,
        timestep=ttnn_timestep,
        encoder_hidden_states=encoder_hidden_states,
        class_labels=class_labels,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        config=config,
    )
    if use_signpost:
        signpost(header="stop")
    first_iter = time.time() - first_iter
    print(f"First iteration took {first_iter} seconds")

    second_iter = time.time()
    ttnn_output = model(
        input,
        timestep=ttnn_timestep,
        encoder_hidden_states=encoder_hidden_states,
        class_labels=class_labels,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        config=config,
    )
    second_iter = time.time() - second_iter
    print(f"Second iteration took {second_iter} seconds")

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.996)
