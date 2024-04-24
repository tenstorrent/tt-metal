# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.reference import torch_functional_vit
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc

# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTPatchEmbeddings(config).eval()

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_vit.custom_preprocessor,
    )

    output = torch_functional_vit.vit_patch_embeddings(
        torch_pixel_values,
        parameters=parameters,
    )
    assert_with_pcc(torch_output, output[0], 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_embeddings(model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTEmbeddings(config).eval()

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_vit.custom_preprocessor,
    )

    # TODO: integrate within paramters
    model_state_dict = model.state_dict()
    torch_cls_token = torch.nn.Parameter(model_state_dict["cls_token"])
    torch_position_embeddings = torch.nn.Parameter(model_state_dict["position_embeddings"])

    output = torch_functional_vit.vit_embeddings(
        config,
        torch_pixel_values,
        torch_position_embeddings,
        torch_cls_token,
        parameters=parameters,
    )
    assert_with_pcc(torch_output, output[0], 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_vit_attention(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )
    print(parameters)

    output = torch_functional_vit.vit_attention(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_vit_intermediate(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTIntermediate(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_vit.vit_intermediate(
        torch_hidden_states,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_vit_output(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTOutput(config).eval()

    torch_intermediate = torch_random(
        (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.float32
    )
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_intermediate, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_vit.vit_output(
        config,
        torch_intermediate,
        torch_residual,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_vit_layer(model_name, batch_size, sequence_size):
    torch.manual_seed(322)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTLayer(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_vit.vit_layer(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [198])
def test_vit_encoder(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTEncoder(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_vit.vit_encoder(
        config,
        torch_hidden_states,
        attention_mask=None,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit(model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model = model.to(torch.bfloat16)

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.bfloat16)
    torch_output, *_ = model(torch_pixel_values).logits

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_vit.custom_preprocessor,
    )

    # TODO: integrate within paramters
    model_state_dict = model.state_dict()
    torch_cls_token = torch.nn.Parameter(model_state_dict["vit.embeddings.cls_token"])
    torch_position_embeddings = torch.nn.Parameter(model_state_dict["vit.embeddings.position_embeddings"])

    output = torch_functional_vit.vit(
        config,
        torch_pixel_values,
        torch_position_embeddings,
        torch_cls_token,
        attention_mask=None,
        parameters=parameters,
    )

    print(torch_output.shape, output.shape)
    assert_with_pcc(torch_output, output[0], 0.9999)
