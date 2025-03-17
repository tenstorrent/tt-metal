# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
import pdb

import ttnn.torch_tracer
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0
from models.demos.segformer.tests.segformer_test_infra import create_test_infra


def buffer_address(tensor):
    addr = []
    for t in ttnn.get_device_tensors(tensor):
        addr.append(t.buffer_address())
    return addr


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time",
    [
        [1, ttnn.bfloat16, ttnn.bfloat16, 35, 1.5],
    ],
)
def test_perf_segformer_trace_2cq(
    device, batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time
):
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    )

    tt_inputs_host, padded_shape, l1_config_input = test_infra.setup_l1_sharded_input(device)
    dram_config_input = ttnn.DRAM_MEMORY_CONFIG
    dram_config_output = ttnn.DRAM_MEMORY_CONFIG
    tt_inputs_dram = ttnn.allocate_tensor_on_device(
        tt_inputs_host.shape, tt_inputs_host.dtype, tt_inputs_host.layout, device, dram_config_input
    )

    ## MODEL COMPILATION

    # initialize events on both CQs
    # first_op_event = ttnn.record_event(device, 0)
    # read_event = ttnn.record_event(device, 1)

    # write host input to DRAM
    # we do padding on the device side to avoid a variety of issues
    jit_start = time.time()
    first_op_event = ttnn.record_event(device, 0)
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_dram, cq_id=1)
    tt_inputs_dram = ttnn.pad(tt_inputs_dram, padded_shape, [0, 0, 0, 0], 0)
    write_event = ttnn.record_event(device, 1)

    # reshard input to L1
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_dram, l1_config_input)
    first_op_event = ttnn.record_event(device, 0)

    # run model
    read_event = ttnn.record_event(device, 1)
    test_infra.run()
    ttnn.wait_for_event(0, read_event)
    tt_outputs_dram = ttnn.to_memory_config(test_infra.output_tensor.logits, dram_config_output)
    last_op_event = ttnn.record_event(device, 0)
    jit_end = time.time()

    # deallocate and reallocate tt_inputs dram due to padding issue
    # this defeats the purpose of persistent DRAM input but its the only way for now
    tt_inputs_dram.deallocate(force=True)
    tt_inputs_dram = ttnn.allocate_tensor_on_device(
        tt_inputs_host.shape, tt_inputs_host.dtype, tt_inputs_host.layout, device, dram_config_input
    )

    ## TRACE CAPTURE

    ttnn.wait_for_event(1, last_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_dram, cq_id=1)
    tt_inputs_dram = ttnn.pad(tt_inputs_dram, padded_shape, [0, 0, 0, 0], 0)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_dram, l1_config_input)
    first_op_event = ttnn.record_event(device, 0)

    input_trace_addr = buffer_address(test_infra.input_tensor)
    input_spec = test_infra.input_tensor.spec

    # deallocate to preserve input tensor address in L1
    test_infra.output_tensor.logits.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()

    test_infra.input_tensor = ttnn.allocate_tensor_on_device(input_spec, device)
    assert input_trace_addr == test_infra.input_tensor.buffer_address()

    ttnn.end_trace_capture(device, cq_id=0)

    # potentially unneeded, reenable if copy to host fails below
    # tt_inputs_dram.deallocate(force=True)
    # tt_inputs_dram = ttnn.allocate_tensor_on_device(
    #    tt_inputs_host.shape,
    #    tt_inputs_host.dtype,
    #    tt_inputs_host.layout,
    #    device,
    #    dram_config_input)

    ## EXECUTE TRACE
    tt_outputs_host = []

    # luckily we can reuse it here for all loop iterations
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_dram, cq_id=1)
    tt_inputs_dram = ttnn.pad(tt_inputs_dram, padded_shape, [0, 0, 0, 0], 0)
    write_event = ttnn.record_event(device, 1)

    for iter in range(0, 2):
        ttnn.wait_for_event(0, write_event)
        test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_dram, l1_config_input)
        first_op_event = ttnn.record_event(device, 0)

        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        tt_outputs_dram = ttnn.to_memory_config(test_infra.output_tensor.logits, dram_config_output)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_dram, cq_id=1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(0, last_op_event)
        tt_outputs_host.append(tt_outputs_dram.cpu(blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)

    ## BENCHMARK TRACE
    start = time.time()
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_dram, l1_config_input)
    first_op_event = ttnn.record_event(device, 0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.wait_for_event(0, read_event)

    tt_outputs_dram = ttnn.to_memory_config(test_infra.output_tensor.logits, dram_config_output)
    last_op_event = ttnn.record_event(device, 0)

    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_dram, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    ttnn.wait_for_event(0, last_op_event)
    tt_outputs_host.append(tt_outputs_dram.cpu(blocking=False, cq_id=1))
    read_event = ttnn.record_event(device, 1)
    end = time.time()

    ttnn.synchronize_device(device)

    prep_perf_report(
        model_name="segformer_e2e",
        batch_size=batch_size,
        inference_and_compile_time=end_jit - start_jit,
        inference_time=end - start,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="trace_2cq",
    )
