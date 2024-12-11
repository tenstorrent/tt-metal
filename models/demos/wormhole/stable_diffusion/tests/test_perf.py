# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import json
import torch
import pytest
import numpy as np
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset
from scipy import integrate

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    LMSDiscreteScheduler,
)
from models.utility_functions import (
    skip_for_grayskull,
)
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from ttnn import unsqueeze_to_4D
from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.wormhole.stable_diffusion.sd_helper_funcs import TtLMSDiscreteScheduler
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)

from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import profiler, enable_persistent_kernel_cache, skip_for_grayskull


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
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


@skip_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, expected_compile_time, expected_inference_time",
    [
        (2, 4, 3600, 0.14),  # Issue 7816 Inference time
    ],
)
def test_stable_diffusion_perf(device, batch_size, num_inference_steps, expected_compile_time, expected_inference_time):
    device.enable_program_cache()
    # disable_persistent_kernel_cache()

    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"
    # Clear global profiler state before starting measurements
    profiler.clear()

    # setup envvar if testing on N300
    wh_arch_yaml_org = None
    if device.core_grid.y == 7:
        if ("WH_ARCH_YAML" not in os.environ) or (
            os.environ["WH_ARCH_YAML"] != "wormhole_b0_80_arch_eth_dispatch.yaml"
        ):
            pytest.skip("SD unet2d only works for 8x8 grid size")

    # setup the configs
    ttnn.CONFIG.throw_exception_on_fallback = True
    in_channels = 4
    input_height = 64
    input_width = 64

    # setup pytorch model
    torch.manual_seed(0)
    model_name = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()
    config = model.config

    # setup scheduler
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, device=device
    )
    ttnn_scheduler.set_timesteps(4)

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

    input_pt = torch.randn(hidden_states_shape)

    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    input = ttnn.from_torch(input_pt, ttnn.bfloat16)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, input, model.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()
    torch_output = model(input_pt, timestep=time_step[0], encoder_hidden_states=encoder_hidden_states.squeeze(0)).sample

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # define model
    reader_patterns_cache = {}
    model = UNet2D(device, parameters, batch_size, input_height, input_width, reader_patterns_cache)

    # run inference iterations
    for i in range(num_inference_steps):
        profiler.start(f"model_run_for_inference_{i}")
        ttnn_output = model(
            input,
            timestep=_tlist[i],
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            config=config,
        )
        ttnn_output = ttnn_to_torch(ttnn_output)
        profiler.end(f"model_run_for_inference_{i}")

    # printout the perf
    profiler.print()
    comment = f"diffusiondb_512x512"
    first_iter_time = profiler.get("model_run_for_inference_0")
    second_iter_time = profiler.get(f"model_run_for_inference_{num_inference_steps-1}")

    logger.info("Call prep-perf-report")
    prep_perf_report(
        model_name=f"StableDiffusion",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )
    assert (
        second_iter_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {second_iter_time}"
    logger.info("Exit SD perf test")


@skip_for_grayskull()
@pytest.mark.skip(reason="#9945: Skip for now since this breaks on WH because of di/dt")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "expected_perf",
    ((10.31),),
)
def test_stable_diffusion_device_perf(expected_perf):
    subdir = "ttnn_stable_diffusion"
    margin = 0.01
    batch = 1
    iterations = 1
    command = f"pytest tests/ttnn/integration_tests/stable_diffusion/test_unet_2d_condition_model.py::test_unet_2d_condition_model_512x512[batch_size=2-in_channels=4-input_height=64-input_width=64-device_params=l1_small_size_24576]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    # back-up the value of WH_ARCH_YAML if exist
    wh_arch_yaml_backup = None
    if "WH_ARCH_YAML" in os.environ:
        wh_arch_yaml_backup = os.environ["WH_ARCH_YAML"]

    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    post_processed_results = run_device_perf(command, subdir, iterations, cols, batch, has_signposts=True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"stable_diffusion_{batch}batch",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

    # set WH_ARCH_YAML back to the original value
    if wh_arch_yaml_backup is not None:
        os.environ["WH_ARCH_YAML"] = wh_arch_yaml_backup
    else:
        del os.environ["WH_ARCH_YAML"]


@skip_for_grayskull()
@pytest.mark.skip(reason="#9945: Skip for now since this breaks on WH because of di/dt")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "expected_perf",
    ((10.4),),
)
def test_stable_diffusion_device_perf_new_conv(expected_perf):
    subdir = "ttnn_stable_diffusion"
    margin = 0.01
    batch = 1
    iterations = 1
    command = f"pytest tests/ttnn/integration_tests/stable_diffusion/test_unet_2d_condition_model_new_conv.py::test_unet_2d_condition_model_512x512[batch_size=2-in_channels=4-input_height=64-input_width=64-device_params=l1_small_size_24576]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    # back-up the value of WH_ARCH_YAML if exist
    wh_arch_yaml_backup = None
    if "WH_ARCH_YAML" in os.environ:
        wh_arch_yaml_backup = os.environ["WH_ARCH_YAML"]

    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    post_processed_results = run_device_perf(command, subdir, iterations, cols, batch, has_signposts=True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"stable_diffusion_{batch}batch",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

    # set WH_ARCH_YAML back to the original value
    if wh_arch_yaml_backup is not None:
        os.environ["WH_ARCH_YAML"] = wh_arch_yaml_backup
    else:
        del os.environ["WH_ARCH_YAML"]
