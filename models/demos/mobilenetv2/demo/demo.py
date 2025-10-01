# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.mobilenetv2.common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE, load_torch_model
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
)
from models.demos.utils.common_demo_utils import get_batch, get_data_loader, get_mesh_mappers, load_imagenet_dataset
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)

NUM_VALIDATION_IMAGES_IMAGENET = 49920


def run_mobilenetv2_imagenet_demo(
    device,
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator=None,
    entire_imagenet_dataset=False,
    expected_accuracy=0.68,
    resolution=(224, 224),
):
    batch_size = batch_size_per_device * device.get_num_devices()
    iterations = iterations // device.get_num_devices()
    if entire_imagenet_dataset:
        iterations = NUM_VALIDATION_IMAGES_IMAGENET // batch_size

    profiler.clear()
    with torch.no_grad():
        inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

        torch_model = Mobilenetv2()
        torch_model = load_torch_model(torch_model, model_location_generator)
        torch_model.eval()

        model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
        ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size_per_device)

        _, host_input_tensor = create_mobilenetv2_input_tensors(
            batch=batch_size,
            input_height=resolution[0],
            input_width=resolution[1],
            pad_channels=16,
            mesh_mapper=inputs_mesh_mapper,
        )

        input_dram_mem_config = get_memory_config_for_persistent_dram_tensor(
            host_input_tensor.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
        )
        logger.info(
            f"Auto-selected persistent DRAM tensor memory config: shape={host_input_tensor.shape}, shard_shape={input_dram_mem_config.shard_spec.shape}, grid={input_dram_mem_config.shard_spec.grid}"
        )

        input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
        assert (
            host_input_tensor.shape[-2] % input_l1_core_grid.num_cores == 0
        ), "Expecting even sharding on L1 input tensor"
        input_l1_mem_config = ttnn.create_sharded_memory_config(
            shape=(host_input_tensor.shape[2] // input_l1_core_grid.num_cores, host_input_tensor.shape[-1]),
            core_grid=input_l1_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
        pipe = create_pipeline_from_config(
            config,
            ttnn_model,
            device,
            dram_input_memory_config=input_dram_mem_config,
            l1_input_memory_config=input_l1_mem_config,
        )

        profiler.start(f"compile")
        pipe.compile(host_input_tensor)
        profiler.end(f"compile")
        logger.info("ImageNet-1k validation Dataset")
        input_loc = load_imagenet_dataset(model_location_generator)
        data_loader = get_data_loader(input_loc, batch_size, iterations, entire_imagenet_dataset)
        input_tensors_all = []
        input_labels_all = []
        for iter in tqdm(range(iterations), desc="Preparing images"):
            inputs, labels = get_batch(data_loader, resolution[0])
            ttnn_input = torch.permute(inputs, (0, 2, 3, 1))
            ttnn_input = torch.nn.functional.pad(ttnn_input, (0, 16 - ttnn_input.shape[-1]), value=0)
            ttnn_input = ttnn.from_torch(
                ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=inputs_mesh_mapper
            )
            ttnn_input = ttnn.reshape(
                ttnn_input,
                (1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]),
            )
            input_tensors_all.append(ttnn_input)
            input_labels_all.append(labels)

        logger.info("Processed ImageNet-1k validation Dataset")

        logger.info("Starting inference")
        profiler.start(f"run")
        outputs = pipe.enqueue(input_tensors_all).pop_all()
        profiler.end(f"run")
        total_inference_time = profiler.get(f"run")

        logger.info("Running accuracy check...")
        correct = 0
        for iter in range(iterations):
            predictions = []
            output = outputs[iter]
            labels = input_labels_all[iter]
            output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
            prediction = output.argmax(dim=-1)
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1

        pipe.cleanup()

        accuracy = correct / (batch_size * iterations)
        logger.info(
            f"Accuracy for total batch size: {batch_size* device.get_num_devices()} over {iterations} iterations is: {accuracy}"
        )
        if entire_imagenet_dataset:
            assert accuracy >= expected_accuracy, f"Accuracy {accuracy} is below expected accuracy {expected_accuracy}"

        first_iter_time = profiler.get(f"compile")
        inference_time_avg = total_inference_time / (iterations * device.get_num_devices())

        compile_time = first_iter_time - 2 * inference_time_avg
        logger.info(f"Compile time: {round(compile_time, 2)} s")
        logger.info(f"Average inference time: {round(1000.0 * inference_time_avg, 2)} ms")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
@pytest.mark.parametrize(
    "iterations, act_dtype, weight_dtype",
    ((100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_mobilenetv2_imagenet_demo(
    device, batch_size, iterations, act_dtype, weight_dtype, imagenet_label_dict, model_location_generator
):
    run_mobilenetv2_imagenet_demo(
        device, batch_size, iterations, imagenet_label_dict, act_dtype, weight_dtype, model_location_generator
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((MOBILENETV2_BATCH_SIZE),),
)
@pytest.mark.parametrize(
    "iterations, act_dtype, weight_dtype",
    ((100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_mobilenetv2_imagenet_demo_dp(
    mesh_device,
    batch_size_per_device,
    iterations,
    act_dtype,
    weight_dtype,
    imagenet_label_dict,
    model_location_generator,
):
    run_mobilenetv2_imagenet_demo(
        mesh_device,
        batch_size_per_device,
        iterations,
        imagenet_label_dict,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )
