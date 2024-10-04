# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import SwinModel

from models.demos.swin.tt import ttnn_optimized_swin
from transformers import SwinForImageClassification as HF_SwinForImageClassification
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.swin.tt.swin_utils import get_relative_position, get_attn_mask
from transformers import AutoFeatureExtractor
from models.utility_functions import is_grayskull, is_wormhole_b0


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_patch_embedding(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    # Torch swinpatchembedding
    torch_model = model.embeddings.patch_embeddings
    pixel_values = torch.rand(batch_size, 3, 224, 224)

    torch_output = torch_model(pixel_values)
    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_pixel = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    tt_output = ttnn_optimized_swin.patch_embeddings(config, tt_pixel, parameters, device)
    tt_output_tensor = ttnn.to_torch(tt_output[0])
    assert_with_pcc(torch_output[0], tt_output_tensor.squeeze(0), 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_embedding(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    pixel_values = torch.rand(batch_size, 3, 224, 224)
    torch_model = model.embeddings
    torch_output = torch_model(pixel_values)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_pixel = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    image_size = (config.image_size, config.image_size)
    patch_size = (config.patch_size, config.patch_size)
    num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
    position_ids = torch.zeros(1, num_patches + 1, config.embed_dim)

    tt_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn_optimized_swin.embeddings(
        config,
        pixel_values=tt_pixel,
        position_embeddings=tt_position_ids,
        parameters=parameters,
        device=device,
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0])
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_self_attention(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    torch_model = model.encoder.layers[0].blocks[0].attention.self
    num_heads, window_size, dim = 3, 7, 96

    hidden_states = torch.rand(64, 49, 96)
    attention_mask = torch.ones(64, 49, 49)

    torch_output = torch_model(hidden_states, attention_mask)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias_table = get_relative_position(model.config, parameters, device)
    tt_output = ttnn_optimized_swin.self_attention(
        model.config,
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        hidden_states=tt_hidden_states,
        attention_mask=tt_attention_mask,
        parameters=parameters.encoder.layers[0].blocks[0].attention,
        device=device,
        relative_position_bias=bias_table[0][0],
    )

    tt_output_tensor = ttnn.to_torch(tt_output[0]).squeeze(0)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_attention(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    torch_model = model.encoder.layers[0].blocks[0].attention
    num_heads, window_size, dim = 3, 7, 96

    hidden_states = torch.rand(64, 49, 96)

    torch_output = torch_model(hidden_states)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias_table = get_relative_position(model.config, parameters, device)
    tt_output = ttnn_optimized_swin.attention(
        model.config,
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        hidden_states=tt_hidden_states,
        parameters=parameters.encoder.layers[0].blocks[0].attention,
        device=device,
        relative_position_bias=bias_table[0][0],
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0]).squeeze(0)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_layer(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    torch_model = model.encoder.layers[0].blocks[0]
    num_heads, window_size, dim = 3, 7, 96

    hidden_states = torch.rand(1, 3136, 96)
    input_resolution = (56, 56)
    shift_size = 0

    torch_output = torch_model(hidden_states, input_resolution)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias_table = get_relative_position(model.config, parameters, device)
    attention_mask_list = get_attn_mask(model.config, device)

    tt_output = ttnn_optimized_swin.swin_layer(
        model.config,
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        shift_size=shift_size,
        hidden_states=tt_hidden_states,
        input_dimensions=(56, 56),
        parameters=parameters.encoder.layers[0].blocks[0],
        device=device,
        relative_position_bias=bias_table[0][0],
        attn_mask=attention_mask_list[0][0],
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0])

    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_stage(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    torch_model = model.encoder.layers[0]
    dim = 96
    input_resolution = (56, 56)
    depth = 2
    num_heads = 3

    hidden_states = torch.rand(1, 3136, 96)
    input_resolution = (56, 56)
    shift_size = 0
    input_dimensions = (56, 56)

    torch_output = torch_model(hidden_states, input_resolution)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias_table = get_relative_position(model.config, parameters, device)
    attention_mask_list = get_attn_mask(model.config, device)
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
        device=device,
        relative_position_bias=bias_table[0],
        attn_mask_list=attention_mask_list[0],
    )
    tt_output_tensor = ttnn.to_torch(tt_output[0])
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_encoder(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    torch_model = model.encoder
    dim = 96
    input_resolution = (56, 56)
    num_heads = 3

    hidden_states = torch.rand(1, 3136, 96)

    torch_output = torch_model(hidden_states, input_resolution)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias_table = get_relative_position(model.config, parameters, device)
    attention_mask_list = get_attn_mask(model.config, device)

    tt_output = ttnn_optimized_swin.encoder(
        model.config,
        hidden_state=tt_hidden_states,
        input_dimension=(56, 56),
        parameters=parameters.encoder,
        device=device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
    )

    tt_output_tensor = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.98)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_model(device, model_name, batch_size, reset_seeds):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
    config = model.config

    torch_model = model

    pixel_values = torch.rand(8, 3, 224, 224)

    torch_output = torch_model(pixel_values)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    tt_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias_table = get_relative_position(model.config, parameters, device)
    attention_mask_list = get_attn_mask(model.config, device)
    tt_output = ttnn_optimized_swin.swin(
        model.config,
        pixel_values=tt_pixel_values,
        parameters=parameters,
        device=device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
    )

    tt_sequence_output = ttnn.to_torch(tt_output[0])
    tt_pooled_output = ttnn.to_torch(tt_output[1])

    assert_with_pcc(torch_output[0], tt_sequence_output, 0.72)
    assert_with_pcc(torch_output[1], tt_pooled_output, 0.90)


@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize("batch_size", [8])
def test_swin_for_image_classification(device, model_name, batch_size, reset_seeds):
    model = HF_SwinForImageClassification.from_pretrained(model_name)

    config = model.config
    torch_model = model

    pixel_values = torch.rand(batch_size, 3, 224, 224)

    torch_output = torch_model(pixel_values)

    parameters = preprocess_model_parameters(
        model_name="ttnn_optimized_swin",
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )

    tt_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias_table = get_relative_position(model.config, parameters.swin, device)
    attention_mask_list = get_attn_mask(model.config, device)

    tt_output = ttnn_optimized_swin.swin_for_image_classification(
        model.config,
        pixel_values=tt_pixel_values,
        parameters=parameters,
        device=device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
    )

    tt_output = ttnn.to_torch(tt_output)
    pcc = 0.91
    if is_grayskull:
        pcc = 0.84
    assert_with_pcc(torch_output.logits, tt_output, pcc)
