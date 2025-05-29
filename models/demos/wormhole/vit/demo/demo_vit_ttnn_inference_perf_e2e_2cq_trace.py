# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.vit.tests.vit_test_infra import create_test_infra
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, is_blackhole, profiler

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    output_sharded_mem_config_DRAM = test_infra.setup_dram_sharded_output(
        device, (1, 8, 224, 1152, None, None)
    )  # scandalous
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    # Initialize the op event so we can write
    first_op_event = ttnn.record_event(device, 0)
    read_event = ttnn.record_event(device, 1)

    # JIT
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    first_op_event = ttnn.record_event(device, 0)
    test_infra.run()
    output_tensor_dram = ttnn.reshard(test_infra.output_tensor, output_sharded_mem_config_DRAM)
    last_op_event = ttnn.record_event(device, 0)

    # Capture trace
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    first_op_event = ttnn.record_event(device, 0)

    spec = test_infra.input_tensor.spec
    input_trace_addr = test_infra.input_tensor.buffer_address()
    test_infra.output_tensor.deallocate(force=True)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    input_l1_tensor = ttnn.allocate_tensor_on_device(spec, device)
    assert input_trace_addr == input_l1_tensor.buffer_address()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    output_tensor_dram = ttnn.reshard(test_infra.output_tensor, output_sharded_mem_config_DRAM, output_tensor_dram)

    # Warmup run
    outputs = []
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(0, write_event)
        input_l1_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_l1_tensor)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_tensor_dram = ttnn.reshard(test_infra.output_tensor, output_sharded_mem_config_DRAM, output_tensor_dram)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        outputs.append(ttnn.from_device(output_tensor_dram, blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)

    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(0, write_event)
        input_l1_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_l1_tensor)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_tensor_dram = ttnn.reshard(test_infra.output_tensor, output_sharded_mem_config_DRAM, output_tensor_dram)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        outputs.append(ttnn.from_device(output_tensor_dram, blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.DumpDeviceProfiler(device)

    ttnn.release_trace(device, trace_id)


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
    )

    ttnn.synchronize_device(device)

    num_warmup_iterations = 100
    num_measurement_iterations = 1000

    run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations)

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
        inference_time_cpu=0,
    )

    model_name = f"ttnn_vit_base_batch_size_{batch_size}"
    comments = ""
    logger.info(f"{model_name} {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"Samples per second: {1 / inference_time_avg * batch_size}")
