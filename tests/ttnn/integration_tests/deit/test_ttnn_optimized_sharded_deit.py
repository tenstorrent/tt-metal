# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

import torch
import pytest
import transformers
from datasets import load_dataset

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import torch_random, skip_for_wormhole_b0
from models.experimental.functional_deit.tt import ttnn_optimized_sharded_deit

from transformers import (
    DeiTImageProcessor,
    DeiTConfig,
    DeiTForImageClassification,
    DeiTForImageClassificationWithTeacher,
)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_deit_patch_embeddings(device, model_name, batch_size, image_size, image_channels):
    config = DeiTConfig.from_pretrained(model_name)
    model = DeiTForImageClassification.from_pretrained(model_name)

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values)
    torch_output, *_ = model.deit.embeddings.patch_embeddings(torch_pixel_values)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit.custom_preprocessor,
    )

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

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            shard_spec,
        ),
    )

    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)

    output = ttnn_optimized_sharded_deit.deit_patch_embeddings(
        config, pixel_values, parameters=parameters, unittest_check=True
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0], 0.9998)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_deit_embeddings(device, reset_seeds, model_name, batch_size, image_size, image_channels):
    config = DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = DeiTForImageClassification.from_pretrained(model_name)

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = DeiTImageProcessor.from_pretrained(model_name)
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
    torch_output, *_ = model.deit.embeddings(torch_pixel_values)

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
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    distillation_tokens = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit.custom_preprocessor,
    )

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

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            shard_spec,
        ),
    )

    output = ttnn_optimized_sharded_deit.deit_embeddings(
        config,
        pixel_values,
        cls_token,
        position_embeddings,
        distillation_tokens,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0], 0.9997)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
def test_deit_attention(device, reset_seeds, model_name, batch_size, sequence_size):
    config = DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)
    model = transformers.models.deit.modeling_deit.DeiTAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(batch_size, 1, 1, sequence_size, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit.custom_preprocessor,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    attention_mask = ttnn.from_torch(
        torch_attention_mask,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    encoder_input = ttnn.to_memory_config(
        hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            hidden_states.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            # orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn_optimized_sharded_deit.deit_attention(
        config,
        encoder_input,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9984)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
def test_deit_intermediate(device, reset_seeds, model_name, batch_size, sequence_size):
    config = DeiTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)
    model = transformers.models.deit.modeling_deit.DeiTIntermediate(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_sharded_deit.deit_intermediate(
        config,
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9996)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
def test_deit_output(device, reset_seeds, model_name, batch_size, sequence_size):
    config = DeiTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)
    model = transformers.models.deit.modeling_deit.DeiTOutput(config).eval()

    torch_intermediate = torch_random((batch_size, sequence_size, config.intermediate_size), -1, 1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_intermediate, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    intermediate = ttnn.from_torch(torch_intermediate, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    residual_sh = ttnn.to_memory_config(
        residual,
        memory_config=ttnn.create_sharded_memory_config(
            residual.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            # orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(residual)

    output = ttnn_optimized_sharded_deit.deit_output(
        config,
        intermediate,
        residual_sh,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9998)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
def test_deit_layer(device, reset_seeds, model_name, batch_size, sequence_size):
    config = DeiTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)
    model = transformers.models.deit.modeling_deit.DeiTLayer(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(batch_size, 1, 1, sequence_size, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_sharded_deit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    attention_mask = ttnn.from_torch(
        torch_attention_mask,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    encoder_input = ttnn.to_memory_config(
        hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            hidden_states.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            # orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn_optimized_sharded_deit.deit_layer(
        config,
        encoder_input,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9997)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
def test_deit_encoder(device, reset_seeds, model_name, batch_size, sequence_size):
    config = DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = DeiTForImageClassification.from_pretrained(model_name, config=config).deit.encoder

    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)
    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    torch_output = model(torch_hidden_states, torch_attention_mask).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_sharded_deit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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

    output = ttnn_optimized_sharded_deit.deit_encoder(
        config,
        hidden_states,
        head_masks,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9459)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
@pytest.mark.parametrize("sequence_size", [224])
def test_deit_for_image_classification(
    device, reset_seeds, model_name, batch_size, image_size, image_channels, sequence_size
):
    config = DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = DeiTForImageClassification.from_pretrained(model_name, config=config)

    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = DeiTImageProcessor.from_pretrained(model_name)
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    torch_output = model(torch_pixel_values).logits

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
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    distillation_tokens = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit.custom_preprocessor,
    )

    torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    batch_size, img_h, img_w, img_c = torch_pixel_values.shape
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

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            shard_spec,
        ),
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

    output = ttnn_optimized_sharded_deit.deit_for_image_classification(
        config,
        pixel_values,
        head_masks,
        cls_token,
        position_embeddings,
        distillation_tokens,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[:, 0, :1000], 0.8062)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
@pytest.mark.parametrize("sequence_size", [224])
def test_deit_for_image_classification_teacher(
    device, reset_seeds, model_name, batch_size, image_size, image_channels, sequence_size
):
    config = DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name, config=config)

    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = DeiTImageProcessor.from_pretrained(model_name)
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    torch_output = model(torch_pixel_values).logits

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
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    distillation_tokens = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit.custom_preprocessor,
    )

    torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    batch_size, img_h, img_w, img_c = torch_pixel_values.shape
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

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            shard_spec,
        ),
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

    output = ttnn_optimized_sharded_deit.deit_for_image_classification_teacher(
        config,
        pixel_values,
        head_masks,
        cls_token,
        position_embeddings,
        distillation_tokens,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[:, 0, :1000], 0.8465)
