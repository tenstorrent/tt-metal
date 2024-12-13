# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from diffusers import StableDiffusionPipeline
import pytest
from tqdm.auto import tqdm
import time

from models.utility_functions import (
    skip_for_grayskull,
    is_wormhole_b0,
    comp_pcc,
    is_blackhole,
)
from diffusers import LMSDiscreteScheduler
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from ttnn import unsqueeze_to_4D
from models.demos.wormhole.stable_diffusion_dp.tests.custom_preprocessing import create_custom_mesh_preprocessor

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
        params = unsqueeze_to_4D(params)

    return params


def get_mesh_mappers(device):
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    if is_mesh_device:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = (
            None  # ttnn.ReplicateTensorToMesh(device) causes unnecessary replication/takes more time on the first pass
        )
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


@skip_for_grayskull()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], ids=["device_params=l1_small_size_24576"], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (2, 4, 64, 64),
    ],
)
# @pytest.mark.skip(reason="#15931: Failing, skip for now")
def test_unet_2d_condition_model_512x512(device, batch_size, in_channels, input_height, input_width):
    device.enable_program_cache()
    print(f"device.enabled_program_cache")

    # setup envvar if testing on N300
    wh_arch_yaml_org = None
    # if device.core_grid.y == 7:
    #     if ("WH_ARCH_YAML" not in os.environ) or (
    #         os.environ["WH_ARCH_YAML"] != "wormhole_b0_80_arch_eth_dispatch.yaml"
    #     ):
    #         pytest.skip("SD unet2d only works for 8x8 grid size")
    print(f" setup env var on n300")
    ttnn.CONFIG.throw_exception_on_fallback = True
    # setup pytorch model
    torch.manual_seed(0)
    print(f"torch manual seed")
    model_name = "CompVis/stable-diffusion-v1-4"
    load_from_disk = False
    print(f" load from disk :  {load_from_disk}")
    if not load_from_disk:
        print(f" if load from disk :  {load_from_disk}")

        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
        print(f" if load from disk : pipe")

        model = pipe.unet
        print(f" if load from disk : pipe.unet")
        model.eval()
        print(f" if load from disk : pipe.unet.eval")

        config = model.config
        print(f" if load from disk : pipe.unet.config")
        torch.save(model, "unet.pt")
        torch.save(config, "unet_config.pt")
        print(f" if load from disk : pipe.unet - model and config saved")

    else:
        print(f" else load from disk : unet.py")
        model = torch.load("unet.pt")
        print(f" else load from disk : unet.pt config")
        config = torch.load("unet_config.pt")

    print(f"mesh_mappers")
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    print(f"mesh_mappers weight: {weights_mesh_mapper}")
    num_devices = device.get_num_devices() if inputs_mesh_mapper else 1
    print(f"num_devices: {num_devices}")
    batch_size = batch_size * num_devices
    print(f"batch_size: {batch_size}")

    print(f"preprocessing parameters start")
    if weights_mesh_mapper:
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)):
            parameters = preprocess_model_parameters(
                model_name=model_name,
                initialize_model=lambda: model,
                custom_preprocessor=custom_preprocessor,
                device=device,
            )
    else:
        parameters = preprocess_model_parameters(
            model_name=model_name,
            initialize_model=lambda: model,
            custom_preprocessor=custom_preprocessor,
            device=device,
        )
    print(f"preprocessing parameters end")

    # unsqueeze weight tensors to 4D for generating perf dump
    parameters = unsqueeze_all_params_to_4d(parameters)
    print(f"unsqueeze_all_params_to_4d")

    timestep_shape = [1, 1, 2, 320]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]

    print(
        f"timestep_shape: {timestep_shape} encoder_hidden_states_shape: {encoder_hidden_states_shape} class_labels:{class_labels} hidden_states_shape: {hidden_states_shape}"
    )
    input = torch.randn(hidden_states_shape)
    timestep = [i for i in tqdm(scheduler.timesteps)][0]
    ttnn_timestep = constant_prop_time_embeddings(timestep, batch_size, model.time_proj)
    print(f"ttnn_timestep b4 unsqueezing: {ttnn_timestep.shape}")
    ttnn_timestep = ttnn_timestep.unsqueeze(0).unsqueeze(0)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
    print(
        f"-- input: {input.shape} timestep: {timestep.shape} encoder_hidden_states:{encoder_hidden_states.squeeze(0).shape} ttnn_timestep: {ttnn_timestep.shape}"
    )

    print(f" ------ TORCH MODEL : start")
    torch_output = model(input, timestep=timestep, encoder_hidden_states=encoder_hidden_states.squeeze(0)).sample
    print(f" ------ TORCH MODEL : end")

    print(f" torch_output")

    input = ttnn.from_torch(
        input,
        ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
    )
    ttnn_timestep = ttnn_timestep.permute(2, 0, 1, 3)  # pre-permute temb
    ttnn_timestep = ttnn.from_torch(
        ttnn_timestep,
        ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )

    print(
        f"ttnn_input: {input.shape} encoder_hidden_states: {encoder_hidden_states.shape} ttnn_timestep: {ttnn_timestep.shape}"
    )
    reader_patterns_cache = {}
    model = UNet2D(device, parameters, batch_size, input_height, input_width, reader_patterns_cache)

    print(f"first iter -")
    first_iter = time.time()
    use_signpost = True
    try:
        from tracy import signpost
    except ModuleNotFoundError:
        use_signpost = False
    if use_signpost:
        signpost(header="start")

    print(f" ------ TTNN MODEL : start")
    print(f" frst : input: {input.shape}")
    print(f" frst : ttnn_timestep: {ttnn_timestep.shape}")
    print(f" frst : encoder_hidden_states: {encoder_hidden_states.shape}")
    # print(f" frst : class_labels: {class_labels.shape}")
    # print(f" frst : attention_mask: {attention_mask.shape if attention_mask is not None else None}")
    # print(f" frst : cross_attention_kwargs: {cross_attention_kwargs}")
    # print(f" frst : return_dict: {return_dict}")

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
    print(f" ------ TTNN MODEL : end")

    if use_signpost:
        signpost(header="stop")
    first_iter = time.time() - first_iter
    print(f"First iteration took {first_iter} seconds")

    print(f"scnd iter -")
    second_iter = time.time()
    print(f" ------ TTNN MODEL : start")
    print(f" scnd : input: {input.shape}")
    print(f" scnd : ttnn_timestep: {ttnn_timestep.shape}")
    print(f" scnd : encoder_hidden_states: {encoder_hidden_states.shape}")
    # print(f" scnd : class_labels: {class_labels.shape}")
    # print(f" scnd : attention_mask: {attention_mask.shape if attention_mask is not None else None}")
    # print(f" scnd : cross_attention_kwargs: {cross_attention_kwargs}")
    # print(f" scnd : return_dict: {return_dict}")

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
    print(f" ------ TTNN MODEL : end")

    second_iter = time.time() - second_iter
    print(f"Second iteration took {second_iter} seconds")
    ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer)

    # times = []
    # for i in range(50):
    #     start = time.time()
    #     ttnn_output = model(
    #         input,
    #         timestep=ttnn_timestep,
    #         encoder_hidden_states=encoder_hidden_states,
    #         class_labels=class_labels,
    #         attention_mask=attention_mask,
    #         cross_attention_kwargs=cross_attention_kwargs,
    #         return_dict=return_dict,
    #         config=config,
    #     )
    #     ttnn_output = ttnn_to_torch(ttnn_output)
    #     passing, output = comp_pcc(torch_output, ttnn_output, pcc=0.99)
    #     print(output)
    #     end = time.time()
    #     times.append(end - start)
    #     print(f"Current iteration took {end - start} seconds")
    # total_time = 0
    # for iter in times:
    #     total_time += iter
    #     print(iter)
    # print(f"Time taken for 50 iterations: {total_time}")
    # print(f"Samples per second: {50 / total_time}")

    print(f"done with runs pcc check")
    passing, output = comp_pcc(torch_output, ttnn_output, pcc=0.981)
    print(output)
    assert passing

    print("EXIT UNET-2D TEST")
