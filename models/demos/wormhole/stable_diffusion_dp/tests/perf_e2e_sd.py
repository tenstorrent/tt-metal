# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger
from transformers import AutoImageProcessor

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    PNDMScheduler,
)

from models.utility_functions import profiler
from models.perf.perf_utils import prep_perf_report
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion_dp.tests.sd_test_infra import (
    create_test_infra,
    unsqueeze_all_params_to_4d,
    constant_prop_time_embeddings,
)

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


def dump_device_profiler(device):
    if isinstance(device, ttnn.Device):
        ttnn.DumpDeviceProfiler(device)
    else:
        for dev in device.get_device_ids():
            ttnn.DumpDeviceProfiler(device.get_device(dev))


ttnn.dump_device_profiler = dump_device_profiler

ttnn.buffer_address = buffer_address
comments = "Stable_Diffusion"


def run_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, tt_inputs)
    print(f" perf_e2e_sd : run_model : compile start")
    profiler.start("compile")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    print(f" perf_e2e_sd : run_model : compile end")

    ttnn.dump_device_profiler(device)

    # print(f" perf_e2e_sd : run_model : cache start")
    # profiler.start("cache")
    # test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # _ = ttnn.from_device(test_infra.run(), blocking=True)
    # profiler.end("cache")
    # print(f" perf_e2e_sd : run_model : cache end")

    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        print(f" perf_e2e_sd : run_model : test_infra inputs to host")
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        print(f" perf_e2e_sd : run_model : test_infra run")
        _ = ttnn.from_device(test_infra.run(), blocking=True)
        # ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)

    if use_signpost:
        signpost(header="start")

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        print(f" perf_e2e_sd : run_model : run : test_infra inputs to host")
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        print(f" perf_e2e_sd : run_model : run : test_infra run")
        outputs.append(ttnn.from_device(test_infra.run(), blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")

    if use_signpost:
        signpost(header="stop")

    ttnn.dump_device_profiler(device)


def run_2cq_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device, tt_inputs)

    tt_inputs_host_dram = tt_inputs_host.to(device, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)

    ttnn.record_event(0, op_event)

    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
        ttnn.record_event(0, op_event)
        _ = ttnn.from_device(test_infra.run(), blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)
    if use_signpost:
        signpost(header="start")

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
        ttnn.record_event(0, op_event)
        outputs.append(ttnn.from_device(test_infra.run(), blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")

    if use_signpost:
        signpost(header="stop")

    ttnn.dump_device_profiler(device)


def run_trace_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, tt_inputs)
    print(f" --- input_mem_config: {input_mem_config}")
    profiler.start("compile")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    shape = test_infra.input_tensor.shape
    dtype = test_infra.input_tensor.dtype
    layout = test_infra.input_tensor.layout
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)
    test_infra.output_tensor.deallocate(force=True)

    profiler.start("cache")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    print(f"After Cache assigning trace_input_addr : ")
    print(f" trace_input_addr: {trace_input_addr}")
    print(f" ttnn.buffer_address(test_infra.input_tensor): {ttnn.buffer_address(test_infra.input_tensor)}")
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    print(f"tt_inputs_host_dram : ")
    print(f"shape: {shape}")
    print(f"dtype: {dtype}")
    print(f"layout: {layout}")
    print(f"device: {device}")
    print(f"input_mem_config: {input_mem_config}")
    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    tt_inputs_host_dram = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        device,
        input_mem_config,
    )
    print(f"Ending trace Capture : ")
    print(f" trace_input_addr: {trace_input_addr}")
    print(f" ttnn.buffer_address(tt_inputs_host_dram): {ttnn.buffer_address(tt_inputs_host_dram)}")

    assert trace_input_addr == ttnn.buffer_address(tt_inputs_host_dram)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        _ = ttnn.from_device(tt_output_res, blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)

    if use_signpost:
        signpost(header="start")

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(ttnn.from_device(tt_output_res, blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")

    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)

    ttnn.release_trace(device, tid)


def run_trace_2cq_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device, tt_inputs)
    tt_inputs_host_dram = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)

    ttnn.record_event(0, op_event)

    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    shape = test_infra.input_tensor.shape
    dtype = test_infra.input_tensor.dtype
    layout = test_infra.input_tensor.layout
    ttnn.record_event(0, op_event)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    # Deallocate the previous output tensor here to make allocation match capture setup
    # This allows us to allocate the input tensor after at the same address
    test_infra.output_tensor.deallocate(force=True)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        device,
        input_mem_config,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(input_tensor)
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        input_tensor = ttnn.reshard(tt_inputs_host_dram, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        input_tensor = ttnn.reshard(tt_inputs_host_dram, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")

    if use_signpost:
        signpost(header="stop")

    ttnn.dump_device_profiler(device)

    ttnn.release_trace(device, tid)


def run_perf_sd(
    device_batch_size,
    num_inference_steps,
    expected_inference_time,
    expected_compile_time,
    input_shape,
    device,
    model_version,
):
    profiler.clear()
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    num_devices = device.get_num_devices() if is_mesh_device else 1
    batch_size = device_batch_size * num_devices
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()
    config = model.config

    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, device=device
    )
    ttnn_scheduler.set_timesteps(4)

    encoder_hidden_states_shape = [1, 2, 77, 768]
    batch_size, in_channels, input_height, input_width = input_shape
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    input_pt = torch.randn(hidden_states_shape)

    torch_encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
    input = ttnn.from_torch(
        input_pt, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, input, model.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()
    test_infra = create_test_infra(
        device,
        device_batch_size,
        input_shape,
        num_inference_steps,
    )
    print(f" perf_e2e_sd : test_infra ")
    ttnn.synchronize_devices(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        print(f" perf_e2e_sd : cpu_key start ")
        torch_output = model(
            input_pt, timestep=time_step[0], encoder_hidden_states=torch_encoder_hidden_states.squeeze(0)
        ).sample
        print(f" perf_e2e_sd : cpu_key end ")
        profiler.end(cpu_key)

        if "sd_trace_2cqs" in model_version:
            run_trace_2cq_model(device, input_pt, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "sd_2cqs" in model_version:
            run_2cq_model(device, input_pt, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "sd_trace" in model_version:
            run_trace_model(device, input_pt, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "sd" in model_version:
            print(f" perf_e2e_sd : ttnn run_model start ")
            run_model(device, input_pt, test_infra, num_warmup_iterations, num_measurement_iterations)
            print(f" perf_e2e_sd : ttnn run_model end ")
        else:
            assert False, f"Model version to run {model_version} not found"

    print(f" perf_e2e_sd : first_iter_time compile+cache ")
    first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")
    print(f" perf_e2e_sd : inference_time_avg ")
    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - 2 * inference_time_avg
    print(f" perf_e2e_sd : prep_perf_report ")

    prep_perf_report(
        model_name=f"ttnn_{model_version}_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(
        f"{model_name} {comments} inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"{model_name} compile time: {compile_time}")
