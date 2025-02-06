# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import (
    is_wormhole_b0,
)
from models.demos.squeezebert.tests.squeezebert_test_infra import create_test_infra

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


def run_squeezebert_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
    device_batch_size = device_batch_size * num_devices
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )

    input_ids, token_type_ids, position_ids, attention_mask, input_mem_config = test_infra.setup_l1_sharded_input(
        device
    )

    # First run configures convs JIT
    test_infra.tt_input_ids = input_ids.to(device, input_mem_config)
    test_infra.tt_token_type_ids = token_type_ids.to(device, input_mem_config)
    test_infra.tt_position_ids = position_ids.to(device, input_mem_config)
    test_infra.tt_attention_mask = attention_mask.to(device, input_mem_config)

    test_infra.run()
    test_infra.validate()

    # Optimized run
    test_infra.tt_input_ids = input_ids.to(device, input_mem_config)
    test_infra.tt_token_type_ids = token_type_ids.to(device, input_mem_config)
    test_infra.tt_position_ids = position_ids.to(device, input_mem_config)
    test_infra.tt_attention_mask = attention_mask.to(device, input_mem_config)

    test_infra.run()
    test_infra.validate()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")

    test_infra.tt_input_ids = input_ids.to(device, input_mem_config)
    test_infra.tt_token_type_ids = token_type_ids.to(device, input_mem_config)
    test_infra.tt_position_ids = position_ids.to(device, input_mem_config)
    test_infra.tt_attention_mask = attention_mask.to(device, input_mem_config)

    test_infra.run()
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()


def run_squeezebert_trace_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )

    (
        tt_input_ids,
        tt_token_type_ids,
        tt_position_ids,
        tt_attention_mask,
        input_mem_config,
    ) = test_infra.setup_l1_sharded_input(device)

    # First run configures convs JIT
    input_ids_spec = test_infra.tt_input_ids.spec
    token_type_spec = test_infra.tt_token_type_ids.spec
    position_ids_spec = test_infra.tt_position_ids.spec
    attn_mask_spec = test_infra.tt_attention_mask.spec

    test_infra.input_ids = tt_input_ids.to(device, input_mem_config)
    test_infra.torch_token_type_ids = tt_token_type_ids.to(device, input_mem_config)
    test_infra.position_ids = tt_position_ids.to(device, input_mem_config)
    test_infra.torch_attention_mask = tt_attention_mask.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()
    test_infra.output_tensor.deallocate(force=True)

    # Optimized run
    test_infra.input_ids = tt_input_ids.to(device, input_mem_config)
    test_infra.torch_token_type_ids = tt_token_type_ids.to(device, input_mem_config)
    test_infra.position_ids = tt_position_ids.to(device, input_mem_config)
    test_infra.torch_attention_mask = tt_attention_mask.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    # Capture
    test_infra.input_ids = tt_input_ids.to(device, input_mem_config)
    test_infra.torch_token_type_ids = tt_token_type_ids.to(device, input_mem_config)
    test_infra.position_ids = tt_position_ids.to(device, input_mem_config)
    test_infra.torch_attention_mask = tt_attention_mask.to(device, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)

    trace_input_addr = ttnn.buffer_address(test_infra.input_ids)
    trace_token_type_addr = ttnn.buffer_address(test_infra.torch_token_type_ids)
    trace_position_ids_addr = ttnn.buffer_address(test_infra.position_ids)
    trace_attention_mask_addr = ttnn.buffer_address(test_infra.torch_attention_mask)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()

    tt_input_dev = ttnn.allocate_tensor_on_device(input_ids_spec, device)
    tt_token_type_dev = ttnn.allocate_tensor_on_device(token_type_spec, device)
    tt_position_ids_dev = ttnn.allocate_tensor_on_device(position_ids_spec, device)
    tt_attn_mask_dev = ttnn.allocate_tensor_on_device(attn_mask_spec, device)

    ttnn.end_trace_capture(device, tid, cq_id=0)

    assert trace_input_addr == ttnn.buffer_address(tt_input_dev)
    assert trace_token_type_addr == ttnn.buffer_address(tt_token_type_dev)
    assert trace_position_ids_addr == ttnn.buffer_address(tt_position_ids_dev)
    assert trace_attention_mask_addr == ttnn.buffer_address(tt_attn_mask_dev)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")

    ttnn.copy_host_to_device_tensor(tt_input_ids, tt_input_dev, 0)
    ttnn.copy_host_to_device_tensor(tt_token_type_ids, tt_token_type_dev, 0)
    ttnn.copy_host_to_device_tensor(tt_position_ids, tt_position_ids_dev, 0)
    ttnn.copy_host_to_device_tensor(tt_attention_mask, tt_attn_mask_dev, 0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()

    ttnn.release_trace(device, tid)


def run_squeezebert_trace_2cqs_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )
    (
        tt_input_ids,
        tt_token_type_ids,
        tt_position_ids,
        tt_attention_mask,
        sharded_mem_config_DRAM,
        input_mem_config,
    ) = self.setup_l1_sharded_input(device, torch_input_tensor)
