# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn

from loguru import logger
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.efficientdetd0.common import load_torch_model_state
from models.experimental.efficientdetd0.tt.efficientdetd0 import TtEfficientDetBackbone
from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    infer_torch_module_args,
    create_custom_mesh_preprocessor,
)


def create_efficientdet_d0_pipeline_model(ttnn_model):
    """
    Create a pipeline model function for EfficientDet_d0.
    The function receives L1 device tensors and returns device tensors.
    """

    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"

        features, regression, classification = ttnn_model(l1_input_tensor)

        for feature in features:
            if feature.layout != ttnn.ROW_MAJOR_LAYOUT:
                feature = ttnn.to_layout(feature, ttnn.ROW_MAJOR_LAYOUT)
            feature = ttnn.to_memory_config(feature, ttnn.DRAM_MEMORY_CONFIG)

        if regression.layout != ttnn.ROW_MAJOR_LAYOUT:
            regression = ttnn.to_layout(regression, ttnn.ROW_MAJOR_LAYOUT)
        regression = ttnn.to_memory_config(regression, ttnn.DRAM_MEMORY_CONFIG)

        if classification.layout != ttnn.ROW_MAJOR_LAYOUT:
            classification = ttnn.to_layout(classification, ttnn.ROW_MAJOR_LAYOUT)
        classification = ttnn.to_memory_config(classification, ttnn.DRAM_MEMORY_CONFIG)

        return (features, regression, classification)

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 10000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, size, expected_compile_time, expected_throughput_fps",
    [(1, 512, 25.4, 39.3)],
)
@pytest.mark.models_performance_bare_metal
def test_efficientdet_d0_e2e_performant(
    device,
    num_iterations,
    batch_size,
    size,
    expected_compile_time,
    expected_throughput_fps,
    model_location_generator,
):
    """
    Test EfficientDet_d0 end-to-end performance with Pipeline API.
    """
    torch.manual_seed(0)

    num_classes = 90
    dtype = ttnn.bfloat16

    logger.info("Building EfficientDet_d0 model...")
    torch_model = EfficientDetBackbone(
        num_classes=num_classes,
        compound_coef=0,
    ).eval()
    load_torch_model_state(torch_model)

    sample_input = torch.randn(batch_size, 3, size, size)
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    module_args = infer_torch_module_args(model=torch_model, input=sample_input)
    ttnn_model = TtEfficientDetBackbone(
        device=device,
        parameters=parameters,
        module_args=module_args,
        num_classes=num_classes,
    )

    ttnn.synchronize_device(device)

    logger.info("Creating pipeline model...")
    pipeline_model = create_efficientdet_d0_pipeline_model(ttnn_model)

    logger.info("Preparing input tensor...")
    ttnn_input_tensor = ttnn.from_torch(
        sample_input,
        device=None,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # TODO: 2CQ with trace and overlapped input
    logger.info(f"Configuring pipeline...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=False, num_command_queues=1, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()

    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running {num_iterations} inference iterations...")
    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end-start) : .2f} fps")

    total_num_samples = batch_size
    prep_perf_report(
        model_name="efficientdet_d0-notrace-1cq",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_num_samples / expected_throughput_fps,
        comments=f"batch_{batch_size}-size_{size}",
    )

    logger.info("Performance test completed!")
