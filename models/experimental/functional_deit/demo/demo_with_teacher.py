# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
import pytest
from loguru import logger

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)

from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_deit.tt import ttnn_optimized_sharded_deit

from models.utility_functions import skip_for_wormhole_b0, torch2tt_tensor
from models.experimental.vit.vit_helper_funcs import get_data_loader, get_batch

from transformers import DeiTConfig, DeiTImageProcessor, DeiTForImageClassificationWithTeacher


def run_demo(
    device,
    reset_seeds,
    model_version,
    functional_model,
    input_path,
    imagenet_label_dict,
    batch_size,
    sequence_size,
):
    model = DeiTForImageClassificationWithTeacher.from_pretrained(model_version)
    config = DeiTConfig.from_pretrained(model_version)
    config = functional_model.update_model_config(config, batch_size)
    image_processor = DeiTImageProcessor.from_pretrained(model_version)

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=functional_model.custom_preprocessor,
    )
    profiler.end(f"preprocessing_parameter")

    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["deit.embeddings.cls_token"]
    torch_position_embeddings = model_state_dict["deit.embeddings.position_embeddings"]
    torch_distillation_token = model_state_dict["deit.embeddings.distillation_token"]

    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token)

    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    distillation_tokens = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)

    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    data_loader = get_data_loader(input_path, batch_size, 2)
    correct = 0
    ttnn_predictions = []

    torch_pixel_values, labels = get_batch(data_loader, image_processor)
    torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))

    batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
    patch_size = 16
    torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
    N, H, W, C = torch_pixel_values.shape

    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {
            ttnn.experimental.tensor.CoreRange(
                ttnn.experimental.tensor.CoreCoord(0, 0),
                ttnn.experimental.tensor.CoreCoord(7, 0),
            ),
        }
    )
    n_cores = 8
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, [N * H * W // n_cores, C], ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
    )

    output = None
    pixel_values = torch2tt_tensor(
        torch_pixel_values,
        device,
        ttnn.experimental.tensor.Layout.ROW_MAJOR,
        tt_memory_config=ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            shard_spec,
        ),
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )

    profiler.start(f"inference_time")
    output = functional_model.deit_for_image_classification_teacher(
        config,
        pixel_values,
        head_masks,
        cls_token,
        position_embeddings,
        distillation_tokens,
        parameters=parameters,
    )
    profiler.start(f"inference_time")

    output = ttnn.to_torch(output)
    ttnn_prediction = output[:, 0, :1000].argmax(-1)

    for i in range(batch_size):
        ttnn_predictions.append(imagenet_label_dict[ttnn_prediction[i].item()])
        logger.info(
            f"Iter: {iter} Sample {i} - Expected Label: {imagenet_label_dict[labels[i]]} - Predicted Label: {ttnn_predictions[i]}"
        )

        if imagenet_label_dict[labels[i]] == ttnn_predictions[-1]:
            correct += 1

    ttnn_accuracy = correct / (batch_size)
    logger.info(f"Inference Accuracy : {ttnn_accuracy}")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "inference_time": profiler.get("inference_time"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")

    return measurements


def run_demo_imagenet_1k(
    device,
    reset_seeds,
    model_version,
    functional_model,
    model_location_generator,
    imagenet_label_dict,
    batch_size,
    sequence_size,
    iterations,
):
    model = DeiTForImageClassificationWithTeacher.from_pretrained(model_version)
    config = DeiTConfig.from_pretrained(model_version)
    config = functional_model.update_model_config(config, batch_size)
    image_processor = DeiTImageProcessor.from_pretrained(model_version)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=functional_model.custom_preprocessor,
    )

    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["deit.embeddings.cls_token"]
    torch_position_embeddings = model_state_dict["deit.embeddings.position_embeddings"]
    torch_distillation_token = model_state_dict["deit.embeddings.distillation_token"]

    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token)

    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    distillation_tokens = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)

    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    correct = 0
    for iter in range(iterations):
        ttnn_predictions = []

        torch_pixel_values, labels = get_batch(data_loader, image_processor)
        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))

        batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
        patch_size = 16
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        N, H, W, C = torch_pixel_values.shape

        shard_grid = ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(7, 0),
                ),
            }
        )
        n_cores = 8
        shard_spec = ttnn.experimental.tensor.ShardSpec(
            shard_grid, [N * H * W // n_cores, C], ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
        )

        output = None
        pixel_values = torch2tt_tensor(
            torch_pixel_values,
            device,
            ttnn.experimental.tensor.Layout.ROW_MAJOR,
            tt_memory_config=ttnn.experimental.tensor.MemoryConfig(
                ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.experimental.tensor.BufferType.L1,
                shard_spec,
            ),
            tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
        )

        output = functional_model.deit_for_image_classification_teacher(
            config,
            pixel_values,
            head_masks,
            cls_token,
            position_embeddings,
            distillation_tokens,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        ttnn_prediction = output[:, 0, :1000].argmax(-1)

        for i in range(batch_size):
            ttnn_predictions.append(imagenet_label_dict[ttnn_prediction[i].item()])
            # logger.info(f"Iter: {iter} Sample {i}:")
            # logger.info(f"Expected Label: {imagenet_label_dict[labels[i]]}")
            # logger.info(f"Predicted Label: {ttnn_predictions[i]}")

            if imagenet_label_dict[labels[i]] == ttnn_predictions[-1]:
                correct += 1

    ttnn_accuracy = correct / (batch_size * iterations)
    logger.info(f"ImageNet Inference Accuracy : {ttnn_accuracy}")


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_version", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [198])
def test_demo(
    device,
    reset_seeds,
    model_version,
    input_path,
    imagenet_label_dict,
    batch_size,
    sequence_size,
    functional_model=ttnn_optimized_sharded_deit,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_demo(
        device=device,
        reset_seeds=reset_seeds,
        model_version=model_version,
        functional_model=functional_model,
        input_path=input_path,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=batch_size,
        sequence_size=sequence_size,
    )


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_version", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [198])
@pytest.mark.parametrize("iterations", [3])
def test_demo_imagenet_1k(
    device,
    reset_seeds,
    model_version,
    model_location_generator,
    imagenet_label_dict,
    batch_size,
    sequence_size,
    iterations,
    functional_model=ttnn_optimized_sharded_deit,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_demo_imagenet_1k(
        device=device,
        reset_seeds=reset_seeds,
        model_version=model_version,
        functional_model=functional_model,
        model_location_generator=model_location_generator,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=batch_size,
        sequence_size=sequence_size,
        iterations=iterations,
    )
