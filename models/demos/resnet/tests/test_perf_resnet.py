# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import AutoImageProcessor
import pytest
import ttnn

from models.utility_functions import is_e75, profiler, divup, disable_persistent_kernel_cache, skip_for_wormhole_b0
from models.perf.perf_utils import prep_perf_report

from loguru import logger
from models.demos.resnet.tests.demo_utils import load_resnet50_model
from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def run_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations):
    profiler.start("compile")
    _ = tt_resnet50(tt_inputs).cpu(blocking=True)
    profiler.end("compile")
    ttnn.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        _ = tt_resnet50(tt_inputs).cpu(blocking=True)
        ttnn.DumpDeviceProfiler(device)

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        outputs.append(tt_resnet50(tt_inputs).cpu(blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    ttnn.DumpDeviceProfiler(device)


def run_2cq_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations):
    input_shape = tt_inputs.get_legacy_shape()
    shard_spec = ttnn.ShardSpec(
        tt_resnet50.dram_shard_grid,
        [
            divup(tt_inputs.volume() // input_shape[3], tt_resnet50.n_dram_cores),
            input_shape[3],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec
    )
    tt_image_res = ttnn.allocate_tensor_on_device(
        tt_inputs.shape, tt_inputs.dtype, tt_inputs.layout, device, sharded_mem_config_DRAM
    )
    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    _ = tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=True)
    profiler.end("compile")
    ttnn.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        _ = tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=True)
        ttnn.DumpDeviceProfiler(device)

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        outputs.append(tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    ttnn.DumpDeviceProfiler(device)


def run_trace_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations):
    input_shape = tt_inputs.get_legacy_shape()
    shard_spec = ttnn.ShardSpec(
        tt_resnet50.dram_shard_grid,
        [
            divup(tt_inputs.volume() // input_shape[3], tt_resnet50.n_dram_cores),
            input_shape[3],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec
    )
    tt_image_res = ttnn.allocate_tensor_on_device(
        tt_inputs.shape, tt_inputs.dtype, tt_inputs.layout, device, sharded_mem_config_DRAM
    )
    # Compile
    profiler.start("compile")
    ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res)
    tt_resnet50(tt_image_res).cpu(blocking=True)
    profiler.end("compile")
    ttnn.DumpDeviceProfiler(device)

    # Capture
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = tt_resnet50(tt_image_res)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        _ = tt_output_res.cpu(blocking=True)
        ttnn.DumpDeviceProfiler(device)

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    ttnn.DumpDeviceProfiler(device)


def run_trace_2cq_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations):
    input_shape = tt_inputs.get_legacy_shape()
    shard_spec = ttnn.ShardSpec(
        tt_resnet50.dram_shard_grid,
        [
            divup(tt_inputs.volume() // input_shape[3], tt_resnet50.n_dram_cores),
            input_shape[3],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec
    )
    tt_image_res = ttnn.allocate_tensor_on_device(
        tt_inputs.shape, tt_inputs.dtype, tt_inputs.layout, device, sharded_mem_config_DRAM
    )

    tt_image_res_shape = tt_image_res.get_legacy_shape()
    reshard_shard_spec = ttnn.ShardSpec(
        tt_resnet50.shard_grid,
        [
            tt_image_res_shape[2] // tt_resnet50.first_conv_num_cores_nhw,
            tt_image_res_shape[3],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    reshard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, reshard_shard_spec
    )
    interleaved_dram_mem_config = ttnn.DRAM_MEMORY_CONFIG

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    # Compile
    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res, 1)
    ttnn.record_event(1, write_event)

    ttnn.wait_for_event(0, write_event)
    reshard_out = ttnn.reshard(tt_image_res, reshard_mem_config)
    ttnn.record_event(0, op_event)

    first_out_addr = reshard_out.buffer_address()
    tt_resnet50(reshard_out, final_out_mem_config=interleaved_dram_mem_config).cpu(blocking=True)
    profiler.end("compile")
    ttnn.DumpDeviceProfiler(device)

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res, 1)
    ttnn.record_event(1, write_event)

    ttnn.wait_for_event(0, write_event)
    reshard_out = ttnn.reshard(tt_image_res, reshard_mem_config)
    ttnn.record_event(0, op_event)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = tt_resnet50(reshard_out, final_out_mem_config=interleaved_dram_mem_config)
    reshard_out = ttnn.allocate_tensor_on_device(
        reshard_out.shape, reshard_out.dtype, reshard_out.layout, device, reshard_mem_config
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert first_out_addr == reshard_out.buffer_address()
    ttnn.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res, 1)
        ttnn.record_event(1, write_event)

        ttnn.wait_for_event(0, write_event)
        reshard_out = ttnn.reshard(tt_image_res, reshard_mem_config, reshard_out)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

        _ = tt_output_res.cpu(blocking=True)
        ttnn.DumpDeviceProfiler(device)

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs, tt_image_res, 1)
        ttnn.record_event(1, write_event)

        ttnn.wait_for_event(0, write_event)
        reshard_out = ttnn.reshard(tt_image_res, reshard_mem_config, reshard_out)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_device(device)
    profiler.end(f"run")
    ttnn.DumpDeviceProfiler(device)


def run_perf_resnet(
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
    model_version,
    model_location_generator,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")
    profiler.clear()
    disable_persistent_kernel_cache()
    if batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "microsoft/resnet-50"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = load_resnet50_model(model_location_generator)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    sharded = False
    if batch_size >= 8:
        sharded = True
    tt_resnet50 = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        device=device,
        state_dict=state_dict,
        base_address="",
        fold_batchnorm=True,
        storage_in_dram=False,
        batch_size=batch_size,
        model_config=model_config,
        sharded=sharded,
    )
    ttnn.synchronize_device(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        if "resnet50_trace_2cqs" in model_version:
            run_trace_2cq_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50_2cqs" in model_version:
            run_2cq_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50_trace" in model_version:
            run_trace_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50" in model_version:
            run_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations)
        else:
            assert False, f"Model version to run {model_version} not found"

    first_iter_time = profiler.get(f"compile")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - inference_time_avg
    prep_perf_report(
        model_name=f"{model_version}_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"{model_name} {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"{model_name} compile time: {compile_time}")


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0062, 19),),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        "resnet50",
        model_location_generator,
    )


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    (
        (20, True, 0.0064, 19),
        (20, False, 0.0064, 19),
    ),
    indirect=["enable_async_mode"],
)
def test_perf_trace_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async_mode,
    model_location_generator,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        f"resnet50_trace_{mode}",
        model_location_generator,
    )


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_hw_cqs": 2}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0041, 19),),
)
def test_perf_2cqs_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        "resnet50_2cqs",
        model_location_generator,
    )


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_hw_cqs": 2, "trace_region_size": 1332224}], indirect=True
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0039, 19),),
)
def test_perf_trace_2cqs_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        "resnet50_trace_2cqs",
        model_location_generator,
    )
