# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import transformers

from loguru import logger
from models.utility_functions import profiler
from models.perf.perf_utils import prep_perf_report
from models.demos.bert.tt import ttnn_optimized_bert as ttnn_roberta
from models.demos.roberta.tests.roberta_test_infra import create_test_infra
from models.demos.roberta.tests.roberta_test_infra import create_position_ids_from_input_ids

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

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

ttnn.buffer_address = buffer_address


def run_trace_2cq_model(
    device,
    input_ids,
    torch_attention_mask,
    torch_token_type_ids,
    torch_position_ids,
    test_infra,
    num_warmup_iterations,
    num_measurement_iterations,
):
    tt_inputs_host, mem_config_DRAM, input_mem_config = test_infra.setup_dram_input(device)
    tt_input_ids_host, tt_token_type_ids_host, tt_position_ids_host, tt_attention_mask_host = tt_inputs_host

    tt_input_ids_device = tt_input_ids_host.to(device, mem_config_DRAM)
    tt_attention_mask_device = tt_attention_mask_host.to(device)
    tt_token_type_ids_device = tt_token_type_ids_host.to(device)
    tt_position_ids_device = tt_position_ids_host.to(device)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids_device, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask_device, 1)
    ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids_device, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids_device, 1)
    ttnn.record_event(1, write_event)

    ttnn.wait_for_event(0, write_event)
    test_infra.input_ids = ttnn.to_memory_config(tt_input_ids_device, memory_config=input_mem_config)
    test_infra.token_type_ids = ttnn.to_memory_config(tt_token_type_ids_device, memory_config=input_mem_config)
    test_infra.position_ids = ttnn.to_memory_config(tt_position_ids_device, memory_config=input_mem_config)
    # test_infra.attention_mask = ttnn.to_memory_config(tt_attention_mask_device, memory_config=input_mem_config)

    shape = test_infra.input_ids.shape
    dtype = test_infra.input_ids.dtype
    layout = test_infra.input_ids.layout
    ttnn.record_event(0, op_event)

    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids_device, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask_device, 1)
    ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids_device, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids_device, 1)
    ttnn.record_event(1, write_event)

    ttnn.wait_for_event(0, write_event)
    test_infra.input_ids = ttnn.to_memory_config(tt_input_ids_device, memory_config=input_mem_config)
    test_infra.token_type_ids = ttnn.to_memory_config(tt_token_type_ids_device, memory_config=input_mem_config)
    test_infra.position_ids = ttnn.to_memory_config(tt_position_ids_device, memory_config=input_mem_config)
    # test_infra.attention_mask = ttnn.to_memory_config(tt_attention_mask_device, memory_config=input_mem_config)
    ttnn.record_event(0, op_event)
    # Deallocate the previous output tensor here to make allocation match capture setup
    # This allows us to allocate the input tensor after at the same address
    test_infra.output_tensor.deallocate(force=True)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids_device, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask_device, 1)
    ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids_device, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids_device, 1)
    ttnn.record_event(1, write_event)

    ttnn.wait_for_event(0, write_event)
    test_infra.input_ids = ttnn.to_memory_config(tt_input_ids_device, memory_config=input_mem_config)
    test_infra.token_type_ids = ttnn.to_memory_config(tt_token_type_ids_device, memory_config=input_mem_config)
    test_infra.position_ids = ttnn.to_memory_config(tt_position_ids_device, memory_config=input_mem_config)
    # test_infra.attention_mask = ttnn.to_memory_config(tt_attention_mask_device, memory_config=input_mem_config)
    ttnn.record_event(0, op_event)

    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_ids)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    input_ids = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        device,
        input_mem_config,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids_device, 1)
        ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask_device, 1)
        ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids_device, 1)
        ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids_device, 1)
        ttnn.record_event(1, write_event)

        ttnn.wait_for_event(0, write_event)
        test_infra.input_ids = ttnn.to_memory_config(tt_input_ids_device, memory_config=input_mem_config)
        test_infra.token_type_ids = ttnn.to_memory_config(tt_token_type_ids_device, memory_config=input_mem_config)
        test_infra.position_ids = ttnn.to_memory_config(tt_position_ids_device, memory_config=input_mem_config)
        # test_infra.attention_mask = ttnn.to_memory_config(tt_attention_mask_device, memory_config=input_mem_config)
        # input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
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
        ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids_device, 1)
        ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask_device, 1)
        ttnn.copy_host_to_device_tensor(tt_token_type_ids_host, tt_token_type_ids_device, 1)
        ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids_device, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        test_infra.input_ids = ttnn.to_memory_config(tt_input_ids_device, memory_config=input_mem_config)
        test_infra.token_type_ids = ttnn.to_memory_config(tt_token_type_ids_device, memory_config=input_mem_config)
        test_infra.position_ids = ttnn.to_memory_config(tt_position_ids_device, memory_config=input_mem_config)
        # test_infra.attention_mask = ttnn.to_memory_config(tt_attention_mask_device, memory_config=input_mem_config)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)

    ttnn.release_trace(device, tid)


def run_perf_roberta(
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    sequence_size,
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
    model_name = "deepset/roberta-large-squad2"

    torch_roberta = transformers.RobertaModel.from_pretrained(model_name)
    torch_roberta.eval()
    torch_roberta.to(torch.bfloat16)
    config = torch_roberta.config
    config.use_dram = True

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(batch_size, sequence_size)
    torch_position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=config.pad_token_id)

    test_infra = create_test_infra(
        device,
        device_batch_size,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_version=model_version,
        config=config,
        sequence_size=sequence_size,
    )
    ttnn.synchronize_devices(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_roberta(
            input_ids=input_ids,
            attention_mask=torch_attention_mask,
            token_type_ids=torch_token_type_ids,
            position_ids=torch_position_ids,
        ).last_hidden_state
        profiler.end(cpu_key)

        run_trace_2cq_model(
            device,
            input_ids,
            torch_attention_mask,
            torch_token_type_ids,
            torch_position_ids,
            test_infra,
            num_warmup_iterations,
            num_measurement_iterations,
        )

        first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")
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
