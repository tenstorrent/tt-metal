# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vit.common import load_torch_model
from models.demos.vit.tt import ttnn_optimized_sharded_vit_wh as ttnn_optimized_sharded_vit
from models.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(device, model_name, batch_size, image_size, image_channels, model_location_generator):
    torch.manual_seed(0)

    model = load_torch_model(model_location_generator, embedding=True)
    config = model.config

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values)
    torch_output, *_ = model.vit.embeddings.patch_embeddings(torch_pixel_values)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
    )

    torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
    patch_size = config.patch_size  # 16
    torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
    N, H, W, C = torch_pixel_values.shape
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 1),
            ),
        }
    )
    n_cores = 16
    shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        ),
    )
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)

    output = ttnn_optimized_sharded_vit.vit_patch_embeddings(
        config, pixel_values, parameters=parameters, unittest_check=True
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0], 0.999)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_embeddings(device, model_name, batch_size, image_size, image_channels, model_location_generator):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    model = load_torch_model(model_location_generator, embedding=True)

    dataset = load_dataset("huggingface/cats-image", revision="ccdec0af347ae11c5315146402c3e16c8bbf4149")
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
    torch_output, *_ = model.vit.embeddings(torch_pixel_values)

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
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
    )

    torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
    patch_size = config.patch_size  # 16
    torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
    N, H, W, C = torch_pixel_values.shape
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 1),
            ),
        }
    )
    n_cores = 16
    shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        ),
    )

    output = ttnn_optimized_sharded_vit.vit_embeddings(
        config,
        pixel_values,
        cls_token,
        position_embeddings,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output, output[0][:197:], 0.999)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    encoder_input = ttnn.to_memory_config(
        hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            hidden_states.shape,
            core_grid=config.core_grid_8x8,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn_optimized_sharded_vit.vit_attention(
        config,
        encoder_input,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.999)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_intermediate(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    model = transformers.models.vit.modeling_vit.ViTIntermediate(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_sharded_vit.vit_intermediate(
        config,
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9984)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_output(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    model = transformers.models.vit.modeling_vit.ViTOutput(config).eval()

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
            core_grid=config.core_grid_8x8,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(residual)

    output = ttnn_optimized_sharded_vit.vit_output(
        config,
        intermediate,
        residual_sh,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.999)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_layer(device, model_name, batch_size, sequence_size, model_location_generator):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    model = load_torch_model(model_location_generator, embedding=True).vit.encoder.layer[0]

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    encoder_input = ttnn.to_memory_config(
        hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            hidden_states.shape,
            core_grid=config.core_grid_8x8,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn_optimized_sharded_vit.vit_layer(
        config,
        encoder_input,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.985)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  ## padded from 197 to 224
def test_vit_encoder(device, model_name, batch_size, sequence_size, model_location_generator):
    torch.manual_seed(0)

    model = load_torch_model(model_location_generator, embedding=True)
    config = model.config
    model = model.vit.encoder
    model = model.to(torch.float32)
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_hidden_states).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn_optimized_sharded_vit.vit_encoder(
        config,
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.96)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
@pytest.mark.parametrize("sequence_size", [224])
def test_vit(device, model_name, batch_size, image_size, image_channels, sequence_size, model_location_generator):
    torch.manual_seed(0)

    model = load_torch_model(model_location_generator, embedding=True)
    config = model.config
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
    dataset = load_dataset("huggingface/cats-image", revision="ccdec0af347ae11c5315146402c3e16c8bbf4149")
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
    torch_output, *_ = model(torch_pixel_values).logits

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
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
    )

    torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
    patch_size = config.patch_size  # 16
    torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
    N, H, W, C = torch_pixel_values.shape

    if batch_size <= 8:
        fold_core_x = batch_size - 1
        fold_core_y = 1
    else:
        batch_size = 16
        fold_core_x = 7
        fold_core_y = 3

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(fold_core_x, fold_core_y),
            ),
        }
    )
    n_cores = batch_size * 2
    shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        ),
    )

    output = ttnn_optimized_sharded_vit.vit(
        config,
        pixel_values,
        cls_token,
        position_embeddings,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)
    # 1000 classes slicing
    assert_with_pcc(torch_output, output[0, 0, :1000], 0.88)
