# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from diffusers import StableDiffusion3Pipeline
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.clip_encoder_layer import CLIPEncoderLayer
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_encoder_layer import ttnn_CLIPEncoderLayer


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CLIPMLP):
            parameters["linear"] = {}
            parameters["linear"]["weight"] = preprocess_linear_weight(model.linear.weight, dtype=ttnn.bfloat8_b)
            parameters["linear"]["bias"] = preprocess_linear_bias(model.linear.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


def test_clip_encoder_layer(device, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    config = pipe.text_encoder.config

    reference_model = CLIPEncoderLayer(config).to(dtype=torch.bfloat16)
    reference_model.eval()

    hidden_states = torch.randn(1, 77, 768, dtype=torch.bfloat16)
    attention_mask = None
    causal_attention_mask = torch.randn(1, 1, 77, 77, dtype=torch.bfloat16)
    output_attentions = False

    torch_output = reference_model(hidden_states, attention_mask, causal_attention_mask, output_attentions)

    parameters = preprocess_model_parameters(initialize_model=lambda: reference_model, device=device)

    ttnn_model = ttnn_CLIPEncoderLayer(config=config)

    ttnn_hidden_states = ttnn.from_torch(
        hidden_states,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_causal_attention_mask = ttnn.from_torch(
        causal_attention_mask,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn_model(ttnn_hidden_states, None, ttnn_causal_attention_mask, False, parameters)

    assert_with_pcc(torch_output[0], ttnn.to_torch(ttnn_output[0]), pcc=0.99)
