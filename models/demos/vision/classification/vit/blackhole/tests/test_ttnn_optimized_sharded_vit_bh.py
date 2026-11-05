# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random
from models.demos.vision.classification.vit.blackhole.tt import (
    ttnn_optimized_sharded_vit_bh as ttnn_optimized_sharded_vit,
)
from models.demos.vision.classification.vit.common.common import load_torch_model
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(device, model_name, batch_size, image_size, image_channels, model_location_generator):
    torch.manual_seed(0)

    model = load_torch_model(model_location_generator, embedding=True)
    config = ttnn_optimized_sharded_vit.update_model_config(model.config, batch_size, device)

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
    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn_optimized_sharded_vit.pixel_values_memory_config(config, device, torch_pixel_values.shape),
    )
    output = ttnn_optimized_sharded_vit.vit_patch_embeddings(
        config, pixel_values, parameters=parameters, unittest_check=True
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0], 0.999)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_embeddings(device, model_name, batch_size, image_size, image_channels, model_location_generator):
    torch.manual_seed(0)

    model = load_torch_model(model_location_generator, embedding=True)
    config = ttnn_optimized_sharded_vit.update_model_config(model.config, batch_size, device)

    dataset = load_dataset(
        "huggingface/cats-image", revision="ccdec0af347ae11c5315146402c3e16c8bbf4149", trust_remote_code=True
    )
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
    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn_optimized_sharded_vit.pixel_values_memory_config(config, device, torch_pixel_values.shape),
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
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    hf_config = transformers.ViTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_vit.update_model_config(hf_config, batch_size, device)
    model = transformers.models.vit.modeling_vit.ViTAttention(hf_config).eval()

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
        memory_config=ttnn_optimized_sharded_vit.activation_memory_config(config, hidden_states.shape),
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
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_mlp(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    hf_config = transformers.ViTConfig.from_pretrained(model_name)
    config = ttnn_optimized_sharded_vit.update_model_config(hf_config, batch_size, device)
    model = transformers.models.vit.modeling_vit.ViTMLP(hf_config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

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
        memory_config=ttnn_optimized_sharded_vit.activation_memory_config(
            config, (batch_size, sequence_size, config.hidden_size)
        ),
    )

    output = ttnn_optimized_sharded_vit.vit_mlp(
        config,
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.995)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_layer(device, model_name, batch_size, sequence_size, model_location_generator):
    torch.manual_seed(0)

    full_model = load_torch_model(model_location_generator, embedding=True)
    config = ttnn_optimized_sharded_vit.update_model_config(full_model.config, batch_size, device)
    model = full_model.vit.layers[0]

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

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
        memory_config=ttnn_optimized_sharded_vit.activation_memory_config(config, hidden_states.shape),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn_optimized_sharded_vit.vit_layer(
        config,
        encoder_input,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_encoder(device, model_name, batch_size, sequence_size, model_location_generator):
    torch.manual_seed(0)

    full_model = load_torch_model(model_location_generator, embedding=True).to(torch.float32)
    config = full_model.config
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size, device)
    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = torch_hidden_states
    for layer in full_model.vit.layers:
        torch_output = layer(torch_output)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: full_model,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        device=device,
    ).vit.layers

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
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
@pytest.mark.parametrize("sequence_size", [224])
def test_vit(device, model_name, batch_size, image_size, image_channels, sequence_size, model_location_generator):
    torch.manual_seed(0)

    model = load_torch_model(model_location_generator, embedding=True)
    config = model.config
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size, device)
    dataset = load_dataset(
        "huggingface/cats-image", revision="ccdec0af347ae11c5315146402c3e16c8bbf4149", trust_remote_code=True
    )
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
    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn_optimized_sharded_vit.pixel_values_memory_config(config, device, torch_pixel_values.shape),
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
    assert_with_pcc(torch_output, output[0, 0, :1000], 0.95)
