# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import math
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor

import ttnn

from models.experimental.functional_vit.tt import ttnn_optimized_interleaved_vit
from models.experimental.vit.vit_helper_funcs import get_data_loader, get_batch

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch2tt_tensor,
)
from models.perf.perf_utils import prep_perf_report
import ast
from pathlib import Path


def get_expected_times(functional_vit):
    return {
        ttnn_optimized_interleaved_vit: (12, 0.08),
    }[functional_vit]


def get_imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("sequence_size", [224])
@pytest.mark.parametrize("functional_vit", [ttnn_optimized_interleaved_vit])
def test_accuracy(
    device,
    use_program_cache,
    model_name,
    batch_size,
    image_size,
    sequence_size,
    functional_vit,
    model_location_generator,
):
    disable_persistent_kernel_cache()

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", config=config)
    model = model.to(torch.bfloat16)

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)

    if functional_vit == ttnn_optimized_interleaved_vit:
        tt_model_name = f"ttnn_{model_name}_optimized"
        batch_size = 8
    else:
        raise ValueError(f"Unknown functional_vit: {functional_vit}")

    # cls_token & position embeddings expand to batch_size
    # TODO: pass batch_size to preprocess_model_parameters
    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
    torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    ##################

    iterations = 50
    imagenet_label_dict = get_imagenet_label_dict()

    data_loader = get_data_loader("ImageNet_data", batch_size, iterations)
    correct = 0
    for iter in range(iterations):
        predictions = []
        inputs, labels = get_batch(data_loader, image_processor)

        torch_pixel_values = torch.permute(inputs, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
        patch_size = 16
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        N, H, W, C = torch_pixel_values.shape
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 0),
                ),
            }
        )
        n_cores = 8
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR, False)

        tt_inputs = torch2tt_tensor(
            torch_pixel_values,
            device,
            ttnn.ROW_MAJOR_LAYOUT,
            tt_memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            ),
            tt_dtype=ttnn.bfloat16,
        )

        if torch_attention_mask is not None:
            head_masks = [
                ttnn.from_torch(
                    torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for index in range(config.num_hidden_layers)
            ]
        else:
            head_masks = [None for _ in range(config.num_hidden_layers)]

        tt_output = functional_vit.vit(
            config,
            tt_inputs,
            head_masks,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        print(tt_output.shape)

        prediction = ttnn.to_torch(tt_output[:, 0, :1000]).argmax(dim=-1)

        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
        del tt_output, tt_inputs, inputs, labels, predictions

        enable_persistent_kernel_cache()

    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
