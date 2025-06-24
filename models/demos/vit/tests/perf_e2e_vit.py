# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from loguru import logger
from transformers import AutoImageProcessor

import ttnn
from models.demos.vit.tests.vit_test_infra import create_test_infra
from models.demos.wormhole.vit.demo.vit_helper_funcs import get_batch, get_data_loader
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import profiler

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def dump_device_profiler(device):
    ttnn.DumpDeviceProfiler(device)


# TODO: Create ttnn apis for this
ttnn.dump_device_profiler = dump_device_profiler

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def run_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, tt_inputs)
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

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        outputs.append(ttnn.from_device(test_infra.run(), blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)


def run_2cq_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
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
    op_event = ttnn.record_event(device, 0)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
        op_event = ttnn.record_event(device, 0)
        _ = ttnn.from_device(test_infra.run(), blocking=True)
        ttnn.dump_device_profiler(device)

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
        test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
        op_event = ttnn.record_event(device, 0)
        outputs.append(ttnn.from_device(test_infra.run(), blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)


def run_trace_model(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, tt_inputs)
    # Compile
    profiler.start("compile")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    spec = test_infra.input_tensor.spec
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)
    test_infra.output_tensor.deallocate(force=True)

    profiler.start("cache")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = test_infra.input_tensor.buffer_address()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == tt_image_res.buffer_address()
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        _ = ttnn.from_device(tt_output_res, blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(ttnn.from_device(tt_output_res, blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)

    ttnn.release_trace(device, tid)


def run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
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
    output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
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
    output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)

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

        output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
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

        output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
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


def run_perf_vit(
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    device,
    model_version,
):
    profiler.clear()
    if device_batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")

    num_devices = device.get_num_devices()
    batch_size = device_batch_size * num_devices
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "google/vit-base-patch16-224"

    # Print mesh device information
    logger.info(f"=== Mesh Device Information ===")
    logger.info(f"Number of devices: {num_devices}")
    logger.info(f"Device batch size: {device_batch_size}")
    logger.info(f"Total batch size: {batch_size}")
    if hasattr(device, "shape"):
        logger.info(f"Mesh shape: {device.shape}")
    logger.info(f"Device type: {type(device)}")

    # Use ViT dataset loading approach
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Get ImageNet data using ViT helper functions
    iterations = 2  # Number of iterations for data loading
    data_loader = get_data_loader("ImageNet_data", batch_size, iterations)
    inputs, labels = get_batch(data_loader, image_processor)

    inputs = inputs.bfloat16()
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    # Print input tensor information
    logger.info(f"=== Input Tensor Information (ViT Dataset) ===")
    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Input dtype: {inputs.dtype}")
    logger.info(f"Input size (MB): {inputs.numel() * inputs.element_size() / (1024*1024):.2f}")
    logger.info(f"Total elements: {inputs.numel():,}")
    logger.info(f"Images per device: {inputs.shape[0] // num_devices}")
    logger.info(f"Number of labels: {len(labels)}")
    logger.info(f"Sample labels: {labels[:5] if len(labels) >= 5 else labels}")
    logger.info(f"==================================")

    # Load PyTorch model for reference
    torch_vit = transformers.ViTForImageClassification.from_pretrained(model_name)
    torch_vit.eval()
    torch_vit.to(torch.bfloat16)

    test_infra = create_test_infra(
        device,
        device_batch_size,
        use_random_input_tensor=True,
    )

    # Print test infrastructure information
    logger.info(f"=== Test Infrastructure Information ===")
    logger.info(f"Test infra device: {test_infra.device}")
    logger.info(f"Test infra batch size: {test_infra.batch_size}")
    logger.info(f"Test infra torch_pixel_values shape: {test_infra.torch_pixel_values.shape}")
    logger.info(f"Test infra config patch size: {test_infra.config.patch_size}")
    logger.info(f"Test infra inputs_mesh_mapper: {test_infra.inputs_mesh_mapper}")
    logger.info(f"Test infra weights_mesh_mapper: {test_infra.weights_mesh_mapper}")
    logger.info(f"Test infra output_mesh_composer: {test_infra.output_mesh_composer}")
    logger.info(f"========================================")

    ttnn.synchronize_device(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_vit(inputs)
        profiler.end(cpu_key)

        if "vit_trace_2cqs" in model_version:
            run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "vit_2cqs" in model_version:
            run_2cq_model(device, inputs, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "vit_trace" in model_version:
            run_trace_model(device, inputs, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "vit" in model_version:
            run_model(device, inputs, test_infra, num_warmup_iterations, num_measurement_iterations)
        else:
            assert False, f"Model version to run {model_version} not found"

    # Handle profiler measurements differently for trace_2cq model
    if "vit_trace_2cqs" in model_version:
        first_iter_time = 0
        compile_time = 0
    else:
        first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")
        compile_time = first_iter_time - 2 * profiler.get("run") / num_measurement_iterations

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    cpu_time = profiler.get(cpu_key)
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
