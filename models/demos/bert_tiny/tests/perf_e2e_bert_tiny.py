# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import pytest
import ttnn

from models.utility_functions import (
    profiler,
)
from models.demos.bert_tiny.tests.bert_tiny_test_infra import create_test_infra

from models.perf.perf_utils import prep_perf_report

from transformers import BertForQuestionAnswering, BertConfig

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


# TODO: Create ttnn apis for this
ttnn.dump_device_profiler = dump_device_profiler

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def run_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, input_mem_config = test_infra.setup_inputs(device)
    profiler.start("compile")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        _ = ttnn.from_device(test_infra.run(), blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        outputs.append(ttnn.from_device(test_infra.run(), blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)


def run_trace_2cq_model(
    device,
    tt_inputs,
    token_type_ids,
    position_ids,
    attention_mask,
    test_infra,
    num_warmup_iterations,
    num_measurement_iterations,
):
    (
        tt_inputs_host,
        tt_token_type_ids_host,
        tt_position_ids_host,
        tt_attention_mask_host,
    ) = test_infra.setup_inputs(device)
    tt_input_ids = tt_inputs_host.to(device)
    tt_token_type_ids = tt_token_type_ids_host.to(device)
    tt_position_ids = tt_position_ids_host.to(device)
    tt_attention_mask = tt_attention_mask_host.to(device)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = tt_input_ids
    test_infra.token_type_ids = tt_token_type_ids
    test_infra.position_ids = tt_position_ids
    test_infra.attention_mask = tt_attention_mask
    shape = test_infra.input_tensor.shape
    dtype = test_infra.input_tensor.dtype
    layout = test_infra.input_tensor.layout
    ttnn.record_event(0, op_event)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")

    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)

    test_infra.input_tensor = tt_input_ids
    test_infra.token_type_ids = tt_token_type_ids
    test_infra.position_ids = tt_position_ids
    test_infra.attention_mask = tt_attention_mask
    ttnn.record_event(0, op_event)
    # Deallocate the previous output tensor here to make allocation match capture setup
    # This allows us to allocate the input tensor after at the same address
    test_infra.output_tensor.deallocate(force=True)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)

    test_infra.input_tensor = tt_input_ids
    test_infra.token_type_ids = tt_token_type_ids
    test_infra.position_ids = tt_position_ids
    test_infra.attention_mask = tt_attention_mask
    ttnn.record_event(0, op_event)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)

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
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)

    ttnn.release_trace(device, tid)


def run_perf_bert_tiny(
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    sequence_size,
    device,
    model_version,
):
    profiler.clear()
    if device_batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")

    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    num_devices = device.get_num_devices() if is_mesh_device else 1
    batch_size = device_batch_size * num_devices
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "mrm8488/bert-tiny-finetuned-squadv2"

    config = BertConfig.from_pretrained(model_name)
    torch_bert_tiny = BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size)

    comments = f"Bert-Tiny_{batch_size}_sequence_size_{sequence_size}"

    test_infra = create_test_infra(
        device,
        device_batch_size,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        config=config,
        sequence_size=sequence_size,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.synchronize_devices(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_bert_tiny(
            torch_input_ids,
            token_type_ids=torch_token_type_ids,
            position_ids=torch_position_ids,
            attention_mask=torch_attention_mask,
        )
        profiler.end(cpu_key)

        run_trace_2cq_model(
            device,
            torch_input_ids,
            torch_token_type_ids,
            torch_position_ids,
            torch_attention_mask,
            test_infra,
            num_warmup_iterations,
            num_measurement_iterations,
        )

    first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - 2 * inference_time_avg

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
