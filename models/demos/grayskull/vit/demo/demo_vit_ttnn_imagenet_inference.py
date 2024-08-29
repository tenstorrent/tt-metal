# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.tt import ttnn_optimized_sharded_vit
from models.utility_functions import skip_for_wormhole_b0, torch2tt_tensor
from models.experimental.vit.vit_helper_funcs import get_data_loader, get_batch

import ast


def get_imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@skip_for_wormhole_b0()
def test_vit(device):
    torch.manual_seed(0)

    model_name = "google/vit-base-patch16-224"
    batch_size = 8
    sequence_size = 224
    iterations = 100

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config)
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
    )

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

    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
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

    # IMAGENET INFERENCE
    #####################
    imagenet_label_dict = get_imagenet_label_dict()
    data_loader = get_data_loader("ImageNet_data", batch_size, iterations)

    correct = 0
    for iter in range(iterations):
        predictions = []

        torch_pixel_values, labels = get_batch(data_loader, image_processor)

        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
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

        output = None
        pixel_values = torch2tt_tensor(
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

        output = ttnn_optimized_sharded_vit.vit(
            config,
            pixel_values,
            head_masks,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        prediction = output[:, 0, :1000].argmax(dim=-1)

        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1

        # del tt_output, tt_inputs, inputs, labels, predictions

    accuracy = correct / (batch_size * iterations)
    print("ImageNet Inference ACCURACY=", accuracy)
