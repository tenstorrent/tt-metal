# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import (
    is_wormhole_b0,
)
from models.demos.wormhole.stable_diffusion.tests.sd_test_infra import create_test_infra

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


# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def run_sd_inference(
    device,
    device_batch_size,
    num_inference_steps,
    input_shape,
):
    test_infra = create_test_infra(device, device_batch_size, input_shape, num_inference_steps)
    batch_size, in_channels, input_height, input_width = input_shape
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    input_pt = torch.randn(hidden_states_shape)

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, input_pt)

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    if use_signpost:
        signpost(header="start")

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()

    if use_signpost:
        signpost(header="stop")

    test_infra.validate()


def run_sd_trace_inference(device, device_batch_size, input_shape, num_inference_steps):
    test_infra = create_test_infra(device, device_batch_size, input_shape, num_inference_steps)
    batch_size, in_channels, input_height, input_width = input_shape
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    input_pt = torch.randn(hidden_states_shape)

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, input_pt)

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    shape = test_infra.input_tensor.shape
    dtype = test_infra.input_tensor.dtype
    layout = test_infra.input_tensor.layout
    test_infra.run()
    test_infra.validate()
    test_infra.output_tensor.deallocate(force=True)

    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    tt_inputs_host_dram = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        device,
        input_mem_config,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(tt_inputs_host_dram)
    ttnn.dump_device_profiler(device)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")

    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()

    ttnn.release_trace(device, tid)


def run_sd_2cqs_inference(device, device_batch_size, input_shape, num_inference_steps):
    test_infra = create_test_infra(device, device_batch_size, input_shape, num_inference_steps)
    batch_size, in_channels, input_height, input_width = input_shape
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    input_pt = torch.randn(hidden_states_shape)

    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device, input_pt)
    tt_inputs_host_dram = tt_inputs_host.to(device, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # Optimized run
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    outputs = []
    for iter in range(0, 2):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_imagtt_inputs_host_drame_res, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
        ttnn.record_event(0, op_event)
        outputs.append(ttnn.from_device(test_infra.run(), blocking=False))

    ttnn.synchronize_devices(device)
    if use_signpost:
        signpost(header="stop")
    for output in outputs:
        test_infra.validate(output)


def run_sd_trace_2cqs_inference(
    device,
    device_batch_size,
    num_inference_steps,
    input_shape,
):
    test_infra = create_test_infra(device, device_batch_size, input_shape, num_inference_steps)
    batch_size, in_channels, input_height, input_width = input_shape
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    input_pt = torch.randn(hidden_states_shape)

    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device, input_pt)
    tt_inputs_host_dram = tt_inputs_host.to(device, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # Optimized run
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host_dram, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
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

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    outputs = []
    for iter in range(0, 2):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_inputs_host_dram, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        input_tensor = ttnn.reshard(tt_inputs_host_dram, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_devices(device)

    if use_signpost:
        signpost(header="stop")
    for output in outputs:
        test_infra.validate(output)

    ttnn.release_trace(device, tid)
