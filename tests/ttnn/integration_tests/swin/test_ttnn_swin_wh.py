# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import os.path

import torch

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import SwinModel

from models.demos.wormhole.swin.tt import ttnn_optimized_swin
from transformers import SwinForImageClassification as HF_SwinForImageClassification
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.wormhole.swin.tt.swin_utils import get_relative_position, get_attn_mask
from transformers import AutoFeatureExtractor
from models.utility_functions import is_grayskull, is_wormhole_b0, skip_for_grayskull


@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_patch_embedding(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()
    config = model.config

    # Torch swinpatchembedding
    torch_model = model.embeddings.patch_embeddings
    pixel_values = torch.rand(batch_size, 3, 224, 224)
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    torch_output = torch_model(pixel_values)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )
    tt_pixel = ttnn.from_torch(
        pixel_values, mesh_mapper=inputs_mesh_mapper, device=mesh_device, layout=ttnn.TILE_LAYOUT
    )

    tt_output = ttnn_optimized_swin.patch_embeddings(
        config, tt_pixel, parameters, mesh_device, inputs_mesh_mapper, output_mesh_composer
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0], mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_embedding(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()
    config = model.config

    pixel_values = torch.rand(batch_size, 3, 224, 224)
    torch_model = model.embeddings

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    torch_output = torch_model(pixel_values)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )
    tt_pixel = ttnn.from_torch(
        pixel_values, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    image_size = (config.image_size, config.image_size)
    patch_size = (config.patch_size, config.patch_size)
    num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
    position_ids = torch.zeros(1, num_patches + 1, config.embed_dim)

    tt_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn_optimized_swin.embeddings(
        config,
        pixel_values=tt_pixel,
        position_embeddings=tt_position_ids,
        parameters=parameters,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0], mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_self_attention(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()
    config = model.config

    torch_model = model.encoder.layers[0].blocks[0].attention.self

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    num_heads, window_size, dim = 3, 7, 96

    hidden_states = torch.rand(64, 49, 96)
    attention_mask = torch.ones(64, 49, 49)

    torch_output = torch_model(hidden_states, attention_mask)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )
    tt_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    tt_attention_mask = ttnn.from_torch(
        attention_mask.view(64, 1, 49, 49),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
    )
    bias_table = get_relative_position(
        config=model.config,
        parameters=parameters,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    tt_output = ttnn_optimized_swin.self_attention(
        model.config,
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        hidden_states=tt_hidden_states,
        attention_mask=tt_attention_mask,
        parameters=parameters.encoder.layers[0].blocks[0].attention,
        device=mesh_device,
        relative_position_bias=bias_table[0][0],
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    tt_output_tensor = ttnn.to_torch(tt_output[0], mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_attention(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()
    config = model.config

    torch_model = model.encoder.layers[0].blocks[0].attention
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    num_heads, window_size, dim = 3, 7, 96

    hidden_states = torch.rand(64, 49, 96)

    torch_output = torch_model(hidden_states)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )

    tt_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    bias_table = get_relative_position(
        config=model.config,
        parameters=parameters,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    tt_output = ttnn_optimized_swin.attention(
        model.config,
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        hidden_states=tt_hidden_states,
        parameters=parameters.encoder.layers[0].blocks[0].attention,
        device=mesh_device,
        relative_position_bias=bias_table[0][0],
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0], mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_layer(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()
    config = model.config

    torch_model = model.encoder.layers[0].blocks[0]
    num_heads, window_size, dim = 3, 7, 96
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    hidden_states = torch.rand(batch_size, 3136, 96)
    input_resolution = (56, 56)
    shift_size = 0

    torch_output = torch_model(hidden_states, input_resolution)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )

    tt_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    bias_table = get_relative_position(
        config=model.config,
        parameters=parameters,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    attention_mask_list = get_attn_mask(model.config, inputs_mesh_mapper, mesh_device, output_mesh_composer, batch_size)
    tt_output = ttnn_optimized_swin.swin_layer(
        model.config,
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        shift_size=shift_size,
        hidden_states=tt_hidden_states,
        input_dimensions=(56, 56),
        parameters=parameters.encoder.layers[0].blocks[0],
        device=mesh_device,
        relative_position_bias=bias_table[0][0],
        attn_mask=attention_mask_list[0][0],
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0], mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_stage(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()
    config = model.config

    torch_model = model.encoder.layers[0]
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    dim = 96
    input_resolution = (56, 56)
    num_heads = 3

    hidden_states = torch.rand(batch_size, 3136, 96)
    input_resolution = (56, 56)

    torch_output = torch_model(hidden_states, input_resolution)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )
    tt_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    bias_table = get_relative_position(
        config=model.config,
        parameters=parameters,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    attention_mask_list = get_attn_mask(model.config, inputs_mesh_mapper, mesh_device, output_mesh_composer, batch_size)

    tt_output = ttnn_optimized_swin.swin_stage(
        model.config,
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        hidden_states=tt_hidden_states,
        input_dimensions=(56, 56),
        depth=config.depths[0],
        downsample=True,
        parameter=parameters.encoder.layers[0],
        device=mesh_device,
        relative_position_bias=bias_table[0],
        attn_mask_list=attention_mask_list[0],
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0], mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_encoder(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()
    config = model.config
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    torch_model = model.encoder
    input_resolution = (56, 56)

    hidden_states = torch.rand(batch_size, 3136, 96)

    torch_output = torch_model(hidden_states, input_resolution)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )
    tt_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    bias_table = get_relative_position(
        config=model.config,
        parameters=parameters,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    attention_mask_list = get_attn_mask(model.config, inputs_mesh_mapper, mesh_device, output_mesh_composer, batch_size)

    tt_output = ttnn_optimized_swin.encoder(
        model.config,
        hidden_state=tt_hidden_states,
        input_dimension=(56, 56),
        parameters=parameters.encoder,
        device=mesh_device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    tt_output_tensor = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.98)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_model(mesh_device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained(model_name).eval()

    torch_model = model
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    pixel_values = torch.rand(batch_size, 3, 224, 224)

    torch_output = torch_model(pixel_values)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=model,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )

    tt_pixel_values = ttnn.from_torch(
        pixel_values, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    bias_table = get_relative_position(
        config=model.config,
        parameters=parameters,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    attention_mask_list = get_attn_mask(model.config, inputs_mesh_mapper, mesh_device, output_mesh_composer, batch_size)

    tt_output = ttnn_optimized_swin.swin(
        model.config,
        pixel_values=tt_pixel_values,
        parameters=parameters,
        device=mesh_device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    tt_sequence_output = ttnn.to_torch(tt_output[0], mesh_composer=output_mesh_composer)
    tt_pooled_output = ttnn.to_torch(tt_output[1], mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output[0], tt_sequence_output, 0.97)
    assert_with_pcc(torch_output[1], tt_pooled_output, 0.98)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_for_image_classification(mesh_device, model_name, batch_size, reset_seeds):
    model = HF_SwinForImageClassification.from_pretrained(model_name)
    torch_model = model

    pixel_values = torch.rand(batch_size, 3, 224, 224)

    torch_output = torch_model(pixel_values)
    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
            device=mesh_device,
        )

    tt_pixel_values = ttnn.from_torch(
        pixel_values, mesh_mapper=inputs_mesh_mapper, device=mesh_device, layout=ttnn.TILE_LAYOUT
    )
    bias_table = get_relative_position(
        model.config, parameters.swin, inputs_mesh_mapper, mesh_device, output_mesh_composer
    )
    attention_mask_list = get_attn_mask(model.config, inputs_mesh_mapper, mesh_device, output_mesh_composer, batch_size)

    tt_output = ttnn_optimized_swin.swin_for_image_classification(
        model.config,
        pixel_values=tt_pixel_values,
        parameters=parameters,
        device=mesh_device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
        mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
    pcc = 0.95

    assert_with_pcc(torch_output.logits, tt_output, pcc)
