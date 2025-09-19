# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoImageProcessor

import ttnn
from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.ttnn_resnet.tests.demo_utils import get_batch, get_data_loader
from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

NUM_VALIDATION_IMAGES_IMAGENET = 49920


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, iterations, act_dtype, weight_dtype",
    ((16, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_run_resnet50_trace_2cqs_inference(
    mesh_device,
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
    entire_imagenet_dataset=False,
    expected_accuracy=0.7555288461538462,
):
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    iterations = iterations // mesh_device.get_num_devices()

    if entire_imagenet_dataset:
        iterations = NUM_VALIDATION_IMAGES_IMAGENET // batch_size

    profiler.clear()
    with torch.no_grad():
        test_infra = create_test_infra(
            mesh_device,
            batch_size_per_device,
            act_dtype,
            weight_dtype,
            ttnn.MathFidelity.LoFi,
            use_pretrained_weight=True,
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
            model_location_generator=model_location_generator,
        )

        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(mesh_device)

        def model_wrapper(l1_input_tensor):
            test_infra.input_tensor = l1_input_tensor
            return test_infra.run()

        pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
            model=model_wrapper,
            device=mesh_device,
            dram_input_memory_config=sharded_mem_config_DRAM,
            l1_input_memory_config=input_mem_config,
        )

        profiler.start("compile")
        pipeline.compile(tt_inputs_host)
        profiler.end("compile")
        model_version = "microsoft/resnet-50"
        image_processor = AutoImageProcessor.from_pretrained(model_version)
        logger.info("ImageNet-1k validation Dataset")
        input_loc = str(model_location_generator("ImageNet_data"))
        data_loader = get_data_loader(input_loc, batch_size, iterations, entire_imagenet_dataset)

        input_tensors_all = []
        input_labels_all = []
        tt_inputs_host_all = []

        logger.info("Preparing ImageNet-1k validation Dataset")
        for iter in tqdm(range(iterations), desc="Preparing images"):
            inputs, labels = get_batch(data_loader, image_processor)
            input_tensors_all.append(inputs)
            input_labels_all.append(labels)

        logger.info("Starting inference")
        profiler.start("run")
        logger.info("Preparing ImageNet-1k validation Dataset")
        for inputs in input_tensors_all:
            tt_inputs_host = ttnn.from_torch(
                inputs,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=test_infra.inputs_mesh_mapper,
            )
            tt_inputs_host_all.append(tt_inputs_host)
        outputs = pipeline.enqueue(tt_inputs_host_all).pop_all()
        predictions = []
        for output in outputs:
            output = ttnn.to_torch(output, mesh_composer=test_infra.output_mesh_composer)
            output = torch.reshape(output, (output.shape[0], 1000))
            predictions.append(output.argmax(dim=-1))
        profiler.end("run")

        total_inference_time = profiler.get("run")

        correct = 0
        for iter in range(iterations):
            labels = input_labels_all[iter]
            prediction = predictions[iter]

            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
        pipeline.cleanup()
        accuracy = correct / (batch_size * iterations)

        logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
        if entire_imagenet_dataset:
            assert (
                accuracy == expected_accuracy
            ), f"Accuracy {accuracy} does not match expected accuracy {expected_accuracy}"

        first_iter_time = profiler.get("compile")

        inference_time_avg = total_inference_time / iterations
        compile_time = first_iter_time - 2 * inference_time_avg

    logger.info(
        f"ttnn_resnet50_trace_2cqs_batch_size{batch_size} tests inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"ttnn_resnet50_trace_2cqs_batch_size{batch_size} compile time: {compile_time}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, iterations, act_dtype, weight_dtype",
    ((16, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("entire_imagenet_dataset", [True])
@pytest.mark.parametrize("expected_accuracy", [0.7555288461538462])
def test_run_resnet50_trace_2cqs_accuracy(
    mesh_device,
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
    entire_imagenet_dataset,
    expected_accuracy,
):
    test_run_resnet50_trace_2cqs_inference(
        mesh_device,
        batch_size_per_device,
        iterations,
        imagenet_label_dict,
        act_dtype,
        weight_dtype,
        model_location_generator,
        entire_imagenet_dataset,
        expected_accuracy,
    )
