# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.sd_helper_funcs import run
from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae import Vae
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import is_blackhole, is_wormhole_b0, profiler
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import unsqueeze_to_4D


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 15659008}], indirect=True)
def test_stable_diffusion_unet_trace(device):
    assert is_wormhole_b0() or is_blackhole(), "SD 1.4 runs on Wormhole B0 or Blackhole"

    if is_wormhole_b0():
        os.environ["SLOW_MATMULS"] = "1"

    profiler.clear()
    torch.manual_seed(0)

    model_name = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    torch_model = pipe.unet
    torch_model.eval()
    config = torch_model.config

    # Setup scheduler
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, device=device
    )
    ttnn_scheduler.set_timesteps(4)

    parameters = preprocess_model_parameters(
        model_name=model_name,
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters = unsqueeze_all_params_to_4d(parameters)

    batch_size = 2
    in_channels = 4
    input_height = 64
    input_width = 64
    encoder_hidden_states_shape = [1, 2, 77, 768]
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True

    # Run torch model
    torch_input = torch.randn(hidden_states_shape)
    torch_encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
    time_step = ttnn_scheduler.timesteps.tolist()
    torch_output = torch_model(
        torch_input, timestep=time_step[0], encoder_hidden_states=torch_encoder_hidden_states.squeeze(0)
    ).sample

    # Set up ttnn inputs
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    torch_encoder_hidden_states = torch.nn.functional.pad(torch_encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(torch_encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_input, torch_model.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    ttnn_model = UNet2D(device, parameters, batch_size, input_height, input_width)

    encoder_hidden_states_device = ttnn.allocate_tensor_on_device(
        encoder_hidden_states.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # COMPILE
    ttnn.copy_host_to_device_tensor(encoder_hidden_states, encoder_hidden_states_device, cq_id=0)
    output_tensor = ttnn.from_device(
        ttnn_model(
            ttnn_input,
            timestep=_tlist[0],
            encoder_hidden_states=encoder_hidden_states_device,
            class_labels=class_labels,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            config=config,
        ),
        blocking=True,
    )

    # CAPTURE
    ttnn.copy_host_to_device_tensor(encoder_hidden_states, encoder_hidden_states_device, cq_id=0)
    output_tensor.deallocate(True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(
        ttnn_input,
        timestep=_tlist[0],
        encoder_hidden_states=encoder_hidden_states_device,
        class_labels=class_labels,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        config=config,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # TRACE
    ttnn.synchronize_device(device)
    profiler.start(f"model_run_for_inference_{0}")

    ttnn.copy_host_to_device_tensor(encoder_hidden_states, encoder_hidden_states_device, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    host_output_tensor = output_tensor.cpu(blocking=False)
    ttnn.synchronize_device(device)

    profiler.end(f"model_run_for_inference_{0}")
    ttnn.release_trace(device, tid)

    assert_with_pcc(torch_output, ttnn.to_torch(host_output_tensor), 0.996)

    inference_time = profiler.get(f"model_run_for_inference_{0}")
    expected_inference_time = 0.113 if is_wormhole_b0() else 0.072

    assert (
        inference_time <= expected_inference_time
    ), f"Inference time with trace and 2 cqs is {inference_time}s, while expected time is {expected_inference_time}s"

    num_model_iterations_per_image = 51
    fps = 1 / (inference_time * num_model_iterations_per_image)
    print(f"SD1.4 is running at {fps} FPS")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 8192, "trace_region_size": 6348800}], indirect=True)
def test_stable_diffusion_vae_trace(device):
    if is_wormhole_b0():
        os.environ["SLOW_MATMULS"] = "1"

    profiler.clear()
    torch.manual_seed(0)

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    ttnn_model = Vae(torch_vae=vae, device=device)

    input_channels = 4
    input_height = 64
    input_width = 64
    out_channels = 3
    output_height = 512
    output_width = 512

    input_shape = [1, input_channels, input_height, input_width]
    ttnn_input_device = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    # Run torch model
    torch_input = torch.randn(input_shape)
    torch_output = vae.decode(torch_input).sample

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def wrapper(ttnn_input_device):
        ttnn_nhwc = ttnn.permute(ttnn_input_device, [0, 2, 3, 1])
        ttnn_output = ttnn_model.decode(ttnn_nhwc)
        ttnn_output = ttnn.reshape(ttnn_output, [1, output_height, output_width, out_channels])
        ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
        return ttnn_output

    # COMPILE
    ttnn.copy_host_to_device_tensor(ttnn_input, ttnn_input_device)
    ttnn_output = wrapper(ttnn_input_device)

    ttnn_output.deallocate(True)

    # CAPTURE
    ttnn.copy_host_to_device_tensor(ttnn_input, ttnn_input_device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn_output = wrapper(ttnn_input_device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    # EXECUTE TRACE
    profiler.start(f"vae_run_for_inference_{0}")
    ttnn.copy_host_to_device_tensor(ttnn_input, ttnn_input_device)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn_out = ttnn.to_torch(ttnn_output)
    ttnn.synchronize_device(device)
    profiler.end(f"vae_run_for_inference_{0}")
    ttnn.release_trace(device, tid)

    pcc = 0.985
    if is_blackhole():
        pcc = 0.923
    assert_with_pcc(torch_output, ttnn_out, pcc)

    inference_time = profiler.get(f"vae_run_for_inference_{0}")
    expected_inference_time = 0.749 if is_wormhole_b0() else 0.474

    assert (
        inference_time <= expected_inference_time
    ), f"Inference time with trace is {inference_time}s, while expected time is {expected_inference_time}s"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 21 * 4096, "trace_region_size": 789321728}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, expected_compile_time, expected_inference_time",
    [
        (1, 50, 3600, 6.31),  # Issue 7816 Inference time
    ],
)
def test_stable_diffusion_perf(device, batch_size, num_inference_steps, expected_compile_time, expected_inference_time):
    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"
    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"
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
    guidance_scale = 7.5  # Scale for classifier-free guidance

    # setup pytorch model
    torch.manual_seed(0)
    model_name = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    vae = pipe.vae
    model.eval()
    config = model.config

    # setup scheduler
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        device=device,
        skip_prk_steps=True,
        steps_offset=1,
    )

    # setup vae
    tt_vae = Vae(torch_vae=vae, device=device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    parameters = preprocess_model_parameters(
        model_name=model_name, initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    # unsqueeze weight tensors to 4D for generating perf dump
    parameters = unsqueeze_all_params_to_4d(parameters)

    hidden_states_shape = [batch_size, model.config.in_channels, input_height, input_width]
    input_pt = torch.randn(hidden_states_shape)
    input = ttnn.from_torch(input_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    _tlist = []
    ttnn_latent_model_input = ttnn.concat([input, input], dim=0)
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, model.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()

    encoder_hidden_states_shape = [2, 77, 768]
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # define model
    model = UNet2D(device, parameters, 2, input_height, input_width)

    ttnn_text_embeddings_device = ttnn.allocate_tensor_on_device(
        ttnn.Shape([2, 96, 768]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # COMPILE
    ttnn_scheduler.set_timesteps(num_inference_steps)
    ttnn.copy_host_to_device_tensor(encoder_hidden_states, ttnn_text_embeddings_device, cq_id=0)
    output = ttnn.from_device(
        run(
            model,
            config,
            tt_vae,
            input,
            ttnn_text_embeddings_device,
            _tlist,
            time_step,
            guidance_scale,
            ttnn_scheduler,
            is_blackhole(),
        )
    )

    # CAPTURE
    ttnn_scheduler.set_timesteps(num_inference_steps)
    profiler.end(f"model_run_for_inference_0{0}")
    ttnn.copy_host_to_device_tensor(encoder_hidden_states, ttnn_text_embeddings_device, cq_id=0)
    output.deallocate(True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output = run(
        model,
        config,
        tt_vae,
        input,
        ttnn_text_embeddings_device,
        _tlist,
        time_step,
        guidance_scale,
        ttnn_scheduler,
        is_blackhole(),
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    profiler.end(f"model_run_for_inference_0{0}")

    profiler.start(f"trace_model_run_for_inference_{0}")
    ttnn.copy_host_to_device_tensor(encoder_hidden_states, ttnn_text_embeddings_device, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    image = ttnn.to_torch(output.cpu(blocking=True))
    ttnn.synchronize_device(device)
    profiler.end(f"trace_model_run_for_inference_{0}")
    ttnn.release_trace(device, tid)

    profiler.start(f"postprocess_for_inference_{0}")
    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().float().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    profiler.end(f"postprocess_for_inference_{0}")

    # printout the perf
    profiler.print()
    comment = f"diffusiondb_512x512"
    first_iter_time = profiler.get("model_run_for_inference_0") + profiler.get("postprocess_for_inference_0")
    second_iter_time = profiler.get(f"trace_model_run_for_inference_0") + profiler.get("postprocess_for_inference_0")

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


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "expected_kernel_samples_per_second",
    ((9.5),),
)
def test_stable_diffusion_device_perf(expected_kernel_samples_per_second):
    subdir = "ttnn_stable_diffusion"
    margin = 0.03
    batch = 1
    iterations = 1
    command = f"pytest models/demos/wormhole/stable_diffusion/tests/test_unet_2d_condition_model.py::test_unet_2d_condition_model_512x512[2-4-64-64-device_params=l1_small_size_24576]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_kernel_samples_per_second}

    if is_wormhole_b0():
        # back-up the value of WH_ARCH_YAML if exist
        wh_arch_yaml_backup = None
        if "WH_ARCH_YAML" in os.environ:
            wh_arch_yaml_backup = os.environ["WH_ARCH_YAML"]
        os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
        os.environ["SLOW_MATMULS"] = "1"

    post_processed_results = run_device_perf(command, subdir, iterations, cols, batch, has_signposts=True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"stable_diffusion_{batch}batch",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

    if is_wormhole_b0():
        # set WH_ARCH_YAML back to the original value
        if wh_arch_yaml_backup is not None:
            os.environ["WH_ARCH_YAML"] = wh_arch_yaml_backup
        else:
            del os.environ["WH_ARCH_YAML"]
