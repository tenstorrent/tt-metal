# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from loguru import logger

import ttnn

from models.demos.vit.tt import ttnn_optimized_sharded_vit_wh
from models.utility_functions import is_blackhole

from models.utility_functions import (
    disable_persistent_kernel_cache,
    profiler,
)

from models.demos.wormhole.vit.demo.vit_test_infra import create_test_infra

from models.perf.perf_utils import prep_perf_report

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

#################
import os

os.environ["TTNN_CONFIG_OVERRIDES"] = '{"enable_fast_runtime_mode": true}'
#################


def get_expected_times(functional_vit):
    return {
        ttnn_optimized_sharded_vit_wh: (11, 0.02),
    }[functional_vit]


####


def run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations):
    ops_parallel_config = {}
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    # Initialize the op event so we can write
    op_event = ttnn.record_event(device, 0)

    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    spec = test_infra.input_tensor.spec
    op_event = ttnn.record_event(device, 0)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.DumpDeviceProfiler(device)

    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    # Deallocate the previous output tensor here to make allocation match capture setup
    # This allows us to allocate the input tensor after at the same address
    test_infra.output_tensor.deallocate(force=True)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.DumpDeviceProfiler(device)

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)

    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = test_infra.input_tensor.buffer_address()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    reshard_out = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == reshard_out.buffer_address()
    ttnn.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        reshard_out = ttnn.reshard(tt_image_res, input_mem_config, reshard_out)
        op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.DumpDeviceProfiler(device)

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        reshard_out = ttnn.reshard(tt_image_res, input_mem_config, reshard_out)
        op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.DumpDeviceProfiler(device)

    ttnn.release_trace(device, tid)


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1700000}], indirect=True
)
def test_vit(device, use_program_cache):
    torch.manual_seed(0)

    profiler.clear()
    disable_persistent_kernel_cache()

    batch_size = 8

    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"

    test_infra = create_test_infra(
        device,
        batch_size,
        # final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn.synchronize_device(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations)

    #####
    #####
    first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    compile_time = first_iter_time - 2 * inference_time_avg
    prep_perf_report(
        model_name=f"ttnn_vit_base_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=0,
        expected_inference_time=0,
        comments="",
        inference_time_cpu=0.0,
    )

    model_name = f"ttnn_vit_base_batch_size{batch_size}"
    comments = ""
    logger.info(f"{model_name} {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"{model_name} compile time: {compile_time}")
    logger.info(f"Samples per second: {1 / inference_time_avg * batch_size}")
