# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from torchvision import models as tvmodels
from transformers import AutoImageProcessor
import pytest
import ttnn

from models.perf.perf_utils import prep_perf_report

from loguru import logger
from models.experimental.resnet.tt.ttnn_functional_resnet50_new_conv_api import resnet50
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
    convert_torch_model_to_ttnn_model,
)
from models.utility_functions import (
    profiler,
    is_e75,
    disable_persistent_kernel_cache,
    skip_for_wormhole_b0,
    pad_and_fold_conv_filters_for_unity_stride,
)

MODEL_NAME = "microsoft/resnet-50"

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
    parameters = {}
    if isinstance(model, tvmodels.resnet.Bottleneck):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)
        parameters["conv1"] = {}
        parameters["conv2"] = {}
        parameters["conv3"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(conv1_weight)
        parameters["conv2"]["weight"] = ttnn.from_torch(conv2_weight)
        parameters["conv3"]["weight"] = ttnn.from_torch(conv3_weight)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(conv1_bias, (1, 1, 1, -1)))
        parameters["conv2"]["bias"] = ttnn.from_torch(torch.reshape(conv2_bias, (1, 1, 1, -1)))
        parameters["conv3"]["bias"] = ttnn.from_torch(torch.reshape(conv3_bias, (1, 1, 1, -1)))
        if model.downsample is not None:
            downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(model.downsample[0], model.downsample[1])
            parameters["downsample"] = {}
            parameters["downsample"]["weight"] = ttnn.from_torch(downsample_weight)
            parameters["downsample"]["bias"] = ttnn.from_torch(torch.reshape(downsample_bias, (1, 1, 1, -1)))
    elif isinstance(model, tvmodels.resnet.ResNet):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv1_weight = pad_and_fold_conv_filters_for_unity_stride(conv1_weight, 2, 2)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(conv1_weight)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(conv1_bias, (1, 1, 1, -1)))
        named_parameters = tuple((name, parameter) for name, parameter in model.named_parameters() if "." not in name)
        for child_name, child in tuple(model.named_children()) + named_parameters:
            if child_name in {"conv1", "bn1"}:
                continue
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=name,
                custom_preprocessor=custom_preprocessor,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )
    return parameters


def run_perf_resnet(
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
):
    disable_persistent_kernel_cache()

    profiler.clear()

    cpu_key = f"ref_key_batchsize{batch_size}"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V1).eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_resnet50, custom_preprocessor=custom_preprocessor, device=None
    )

    tt_resnet50 = resnet50(device=device, parameters=parameters, batch_size=batch_size, model_config=model_config)

    ops_parallel_config = {}

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        warmup_end = 5
        for iter in range(0, warmup_end):
            profiler.start(f"iter_{iter}_key")
            _ = tt_resnet50(
                tt_inputs, device=device, batch_size=batch_size, ops_parallel_config=ops_parallel_config
            ).cpu(blocking=True)
            profiler.end(f"iter_{iter}_key")

        num_warm_iterations = 15
        warm_start = warmup_end
        warm_end = warm_start + num_warm_iterations

        outputs = []
        profiler.start(f"run")
        for iter in range(warm_start, warm_end):
            outputs.append(
                tt_resnet50(
                    tt_inputs, device=device, batch_size=batch_size, ops_parallel_config=ops_parallel_config
                ).cpu(blocking=False)
            )
        ttnn.synchronize_device(device)
        profiler.end(f"run")

        # enable_persistent_kernel_cache()

    first_iter_time = profiler.get(f"iter_{0}_key")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_warm_iterations

    for iter in range(0, 5):
        logger.info(f'iter_{iter}_key: {profiler.get(f"iter_{iter}_key")}')
    logger.info(f'{warm_start} to {warm_end} run: {profiler.get("run")}')

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - inference_time_avg
    prep_perf_report(
        model_name=f"resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"resnet50 {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"resnet50 compile time: {compile_time}")


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_l1_small_size", [24576], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (16, 0.0085, 1),  # Issue 7816 Inference time
        (20, 0.0095, 1),  # Issue 7816 Inference time
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )


def run_perf_resnet_trace(
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
):
    disable_persistent_kernel_cache()
    cpu_key = f"ref_key_batchsize{batch_size}"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V1).eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_resnet50, custom_preprocessor=custom_preprocessor, device=None
    )

    tt_resnet50 = resnet50(device=device, parameters=parameters, batch_size=batch_size, model_config=model_config)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        interleaved_mem_config_DRAM = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        )
        tt_image_res = tt_inputs.to(device, interleaved_mem_config_DRAM)
        # Compile
        profiler.start(f"iter_{0}_key")
        ttnn.write_tensor(tt_inputs, tt_image_res)
        tt_resnet50(tt_image_res).cpu(blocking=True)
        profiler.end(f"iter_{0}_key")
        ttnn.device.DumpDeviceProfiler(device)

        # Capture
        tid = ttnn.device.BeginTraceCapture(device, 0, 1304576)
        tt_output_res = tt_resnet50(tt_image_res)
        ttnn.device.EndTraceCapture(device, 0, tid)
        ttnn.device.DumpDeviceProfiler(device)

        warmup_end = 6
        for iter in range(1, warmup_end):
            profiler.start(f"iter_{iter}_key")
            ttnn.tensor.write_tensor(tt_inputs, tt_image_res)
            ttnn.device.ReplayTrace(device, 0, tid, False)
            _ = tt_output_res.cpu(blocking=True)
            profiler.end(f"iter_{iter}_key")
            ttnn.device.DumpDeviceProfiler(device)

        num_warm_iterations = 15
        warm_start = warmup_end
        warm_end = warm_start + num_warm_iterations

        outputs = []
        profiler.start(f"run")
        for iter in range(warm_start, warm_end):
            ttnn.tensor.write_tensor(tt_inputs, tt_image_res)
            ttnn.device.ReplayTrace(device, 0, tid, False)
            outputs.append(tt_output_res.cpu(blocking=False))
        ttnn.device.Synchronize(device)
        profiler.end(f"run")
        ttnn.device.DumpDeviceProfiler(device)

        # enable_persistent_kernel_cache()

    first_iter_time = profiler.get(f"iter_{0}_key")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_warm_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - inference_time_avg
    prep_perf_report(
        model_name=f"resnet50_trace_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"resnet50 {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"resnet50 compile time: {compile_time}")

    ttnn.device.ReleaseTrace(device, tid)

    assert inference_time_avg < expected_inference_time, f"resnet50 {comments} inference is too slow"
    assert compile_time < expected_compile_time, f"resnet50 {comments} compilation is too slow"


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (16, 0.04, 25),
        (20, 0.04, 25),
    ),
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_perf_trace_bare_metal(
    device,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")
    device.enable_async(enable_async)
    run_perf_resnet_trace(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )
    device.enable_async(False)
