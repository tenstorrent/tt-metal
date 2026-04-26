# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ast

import pytest
import torch
import transformers
from loguru import logger
from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import is_blackhole, torch2tt_tensor
from models.demos.deit_tiny.tt import ttnn_optimized_sharded_deit_wh
from models.demos.wormhole.deit_tiny.demo.deit_helper_funcs import get_batch, get_synthetic_data_loader


def get_imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
def test_deit(device):
    torch.manual_seed(0)

    model_name = "/data/hf_cache/Deit/deit-tiny/deit-tiny"
    batch_size = 1
    sequence_size = 224
    iterations = 10

    config = transformers.DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name, config=config)

    config = ttnn_optimized_sharded_deit_wh.update_model_config(config, batch_size)
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit_wh.custom_preprocessor,
    )

    # cls_token & position embeddings expand to batch_size
    # TODO: pass batch_size to preprocess_model_parameters
    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["deit.embeddings.cls_token"]
    torch_distillation_token = model_state_dict["deit.embeddings.distillation_token"]
    torch_position_embeddings = model_state_dict["deit.embeddings.position_embeddings"]
    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    distillation_token = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
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

    data_loader = get_synthetic_data_loader(batch_size, iterations, image_size=224, seed=0)

    predictions = []
    for iter in range(iterations):
        torch_pixel_values, _ = get_batch(data_loader, image_processor)

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
                    ttnn.CoreCoord(batch_size - 1, 3),
                ),
            }
        )
        n_cores = batch_size * 3
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

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

        output = ttnn_optimized_sharded_deit_wh.deit(
            config,
            pixel_values,
            head_masks,
            cls_token,
            distillation_token,
            position_embeddings,
            parameters=parameters,
        )
        logits, cls_logits, distillation_logits = output
        logits = ttnn.to_torch(logits)
        cls_logits = ttnn.to_torch(cls_logits)
        distillation_logits = ttnn.to_torch(distillation_logits)
        assert logits.shape[1] == 1
        assert cls_logits.shape[1] == 1
        assert distillation_logits.shape[1] == 1
        prediction = logits[:, 0, :1000].argmax(dim=-1)
        assert prediction.shape[0] == batch_size

        for i in range(batch_size):
            predictions.append(prediction[i].item())
            logger.info(f"Iter: {iter} Sample: {i} - Synthetic prediction index: {predictions[-1]}")

    assert len(predictions) == batch_size * iterations
    logger.info(f"Synthetic DeiT smoke test completed for {len(predictions)} samples")
