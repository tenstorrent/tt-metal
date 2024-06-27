# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import pytest
import transformers
from torch import nn
from loguru import logger
from transformers import DeiTImageProcessor, DeiTModel, DeiTConfig

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import torch_random, skip_for_wormhole_b0

from models.experimental.functional_deit.reference import torch_functional_deit


def interpolate_pos_encoding(
    position_embeddings: torch.Tensor, patch_size, num_patches, height: int, width: int
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    resolution images.
    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """

    # return position_embeddings
    # num_patches = embeddings.shape[1] - 2
    num_positions = position_embeddings.shape[1] - 2

    if num_patches == num_positions and height == width:
        return position_embeddings

    class_pos_embed = position_embeddings[:, 0, :]
    dist_pos_embed = position_embeddings[:, 1, :]
    patch_pos_embed = position_embeddings[:, 2:, :]
    dim = position_embeddings.shape[-1]
    h0 = height // patch_size
    w0 = width // patch_size
    # # we add a small number to avoid floating point error in the interpolation
    # # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    return torch.cat((class_pos_embed.unsqueeze(0), dist_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_deit_attention(model_name, batch_size, sequence_size, reset_seeds):
    config = DeiTConfig.from_pretrained(model_name)
    model = transformers.models.deit.modeling_deit.DeiTAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )
    print(parameters)

    output = torch_functional_deit.deit_attention(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_deit_intermediate(model_name, batch_size, sequence_size, reset_seeds):
    config = DeiTConfig.from_pretrained(model_name)
    model = transformers.models.deit.modeling_deit.DeiTIntermediate(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_deit.deit_intermediate(
        torch_hidden_states,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_deit_output(model_name, batch_size, sequence_size, reset_seeds):
    config = DeiTConfig.from_pretrained(model_name)
    model = transformers.models.deit.modeling_deit.DeiTOutput(config).eval()

    torch_intermediate = torch_random(
        (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.float32
    )
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_intermediate, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_deit.deit_output(
        config,
        torch_intermediate,
        torch_residual,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [196])
def test_deit_layer(model_name, batch_size, sequence_size, reset_seeds):
    config = DeiTConfig.from_pretrained(model_name)
    model = transformers.models.deit.modeling_deit.DeiTLayer(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_deit.deit_layer(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [198])
def test_deit_encoder(model_name, batch_size, sequence_size, reset_seeds):
    config = DeiTConfig.from_pretrained(model_name)
    model = transformers.models.deit.modeling_deit.DeiTEncoder(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_deit.deit_encoder(
        config,
        torch_hidden_states,
        attention_mask=None,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_deit(model_name, batch_size, image_size, image_channels, reset_seeds):
    config = DeiTConfig.from_pretrained(model_name)
    model = DeiTModel.from_pretrained(model_name)
    model = model.eval()
    patch_size = config.patch_size

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    torch_cls_token = parameters.embeddings.cls_token
    init_position_embeddings = parameters.embeddings.position_embeddings

    image_size = (image_size, image_size)
    patch_size = (patch_size, patch_size)

    tot_patch_count = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
    torch_position_embeddings = torch.nn.Parameter(
        interpolate_pos_encoding(init_position_embeddings, patch_size[0], tot_patch_count, image_size[0], image_size[0])
    )

    output = torch_functional_deit.deit(
        config,
        torch_pixel_values,
        attention_mask=None,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output.squeeze(0), 0.9999)
