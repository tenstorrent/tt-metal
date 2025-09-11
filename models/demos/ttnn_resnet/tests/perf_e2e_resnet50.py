# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import AutoImageProcessor

import ttnn
from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra, load_resnet50_model
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.utility_functions import profiler

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def read_device_profiler(device):
    ttnn.ReadDeviceProfiler(device)


# TODO: Create ttnn apis for this
ttnn.read_device_profiler = read_device_profiler

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def _run_model_pipeline(
    device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations, num_command_queues, trace
):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

    def model_wrapper(l1_input_tensor):
        test_infra.input_tensor = l1_input_tensor
        return test_infra.run()

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=trace, num_command_queues=num_command_queues, all_transfers_on_separate_command_queue=False
        ),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")
    ttnn.read_device_profiler(device)

    host_inputs = [tt_inputs_host] * num_measurement_iterations

    pipeline.preallocate_output_tensors_on_host(
        num_measurement_iterations,
    )

    logger.info(
        f"Starting performance pipline for {num_measurement_iterations} iterations with batch_size={test_infra.batch_size} and num_devices={test_infra.num_devices}"
    )
    if use_signpost:
        signpost(header="start")
    profiler.start(f"run_model_pipeline_{num_command_queues}cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end(f"run_model_pipeline_{num_command_queues}cqs")
    if use_signpost:
        signpost(header="stop")
    ttnn.read_device_profiler(device)

    for i, output in enumerate(outputs):
        passed, pcc_message = test_infra.validate(output)
        logger.info(f"Output {i} validation: {pcc_message}")
        assert passed, f"Output {i} validation failed: {pcc_message}"

    pipeline.cleanup()


def run_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=1,
        trace=False,
    )


def run_trace_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=1,
        trace=True,
    )


def run_2cq_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=2,
        trace=False,
    )


def run_trace_2cq_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=2,
        trace=True,
    )


def run_perf_resnet(
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
    model_version,
    model_location_generator,
):
    profiler.clear()
    if device_batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")

    num_devices = device.get_num_devices()
    batch_size = device_batch_size * num_devices
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "microsoft/resnet-50"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"].bfloat16()
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = load_resnet50_model(model_location_generator)
    torch_resnet50.eval()

    torch_resnet50.to(torch.bfloat16)

    test_infra = create_test_infra(
        device,
        device_batch_size,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )
    ttnn.synchronize_device(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        if "resnet50_trace_2cqs" in model_version:
            run_trace_2cq_model_pipeline(device, inputs, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50_trace" in model_version:
            run_trace_model_pipeline(device, inputs, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50_2cqs" in model_version:
            run_2cq_model_pipeline(device, inputs, test_infra, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50" in model_version:
            run_model_pipeline(device, inputs, test_infra, num_warmup_iterations, num_measurement_iterations)
        else:
            assert False, f"Model version to run {model_version} not found"

    first_iter_time = profiler.get(f"compile")

    # ensuring inference time fluctuations is not noise
    num_cqs = 2 if "2cqs" in model_version else 1
    inference_time_avg = profiler.get(f"run_model_pipeline_{num_cqs}cqs") / num_measurement_iterations

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
