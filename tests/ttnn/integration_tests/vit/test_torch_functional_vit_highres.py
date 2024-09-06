# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import math
import transformers

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.reference import torch_functional_vit
from models.utility_functions import torch_random, is_blackhole, is_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc

# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def interpolate_pos_encoding(
    position_embeddings: torch.Tensor, patch_size, num_patches, height: int, width: int
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    resolution images.

    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """

    # num_patches = embeddings.shape[1] - 1
    num_positions = position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, 0]
    patch_pos_embed = position_embeddings[:, 1:]
    dim = position_embeddings.shape[-1]
    h0 = height // patch_size
    w0 = width // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [1280])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTPatchEmbeddings(config).eval()

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values, interpolate_pos_encoding=True)

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
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [1280])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_embeddings(model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTEmbeddings(config).eval()

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values, interpolate_pos_encoding=True)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_vit.custom_preprocessor,
    )

    # TODO: integrate within paramters
    model_state_dict = model.state_dict()
    torch_cls_token = torch.nn.Parameter(model_state_dict["cls_token"])
    init_position_embeddings = torch.nn.Parameter(model_state_dict["position_embeddings"])
    patch_size = 16
    tot_patch_count = (image_size // patch_size) * (image_size // patch_size)
    torch_position_embeddings = torch.nn.Parameter(
        interpolate_pos_encoding(init_position_embeddings, patch_size, tot_patch_count, image_size, image_size)
    )

    output = torch_functional_vit.vit_embeddings(
        config,
        torch_pixel_values,
        torch_position_embeddings,
        torch_cls_token,
        parameters=parameters,
    )
    assert_with_pcc(torch_output, output[0], 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [198])
def test_vit_layernorm_before(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTLayer.layernorm_before(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output, *_ = model(torch_hidden_states, interpolate_pos_encoding=True)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_vit.vit_layer.layernorm_before(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
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
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
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
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
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
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
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
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
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
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [1280])
@pytest.mark.parametrize("image_channels", [3])
def test_vit(model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model = model.to(torch.bfloat16)

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.bfloat16)
    # torch_output, *_ = model(torch_pixel_values).last_hidden_state
    torch_output, *_ = model(torch_pixel_values, interpolate_pos_encoding=True).logits

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_vit.custom_preprocessor,
    )

    # TODO: integrate within paramters
    model_state_dict = model.state_dict()
    torch_cls_token = torch.nn.Parameter(model_state_dict["vit.embeddings.cls_token"])
    init_position_embeddings = torch.nn.Parameter(model_state_dict["vit.embeddings.position_embeddings"])
    patch_size = 16
    tot_patch_count = (image_size // patch_size) * (image_size // patch_size)
    torch_position_embeddings = torch.nn.Parameter(
        interpolate_pos_encoding(init_position_embeddings, patch_size, tot_patch_count, image_size, image_size)
    )

    output = torch_functional_vit.vit(
        config,
        torch_pixel_values,
        torch_position_embeddings,
        torch_cls_token,
        attention_mask=None,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output[0], 0.9999)
