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
from models.experimental.functional_stable_diffusion3_5.reference.clip_mlp import CLIPMLP
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_mlp import ttnn_CLIPMLP


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CLIPMLP):
            parameters["linear"] = {}
            parameters["linear"]["weight"] = preprocess_linear_weight(model.linear.weight, dtype=ttnn.bfloat8_b)
            parameters["linear"]["bias"] = preprocess_linear_bias(model.linear.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "x_shape",
    [
        ([2, 1024, 1536]),
    ],
)
def test_clip_mlp(device, x_shape, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    clip_mlp_config = pipe.text_encoder.config

    reference_model = CLIPMLP(clip_mlp_config).to(dtype=torch.bfloat16)

    reference_model.eval()
    hidden_states = torch.randn(1, 77, 768, dtype=torch.bfloat16)

    ttnn_hidden_states = ttnn.from_torch(
        hidden_states,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    parameters = preprocess_model_parameters(initialize_model=lambda: reference_model, device=device)

    ttnn_model = ttnn_CLIPMLP(clip_mlp_config)

    torch_output = reference_model(hidden_states)

    ttnn_output = ttnn_model(ttnn_hidden_states, parameters=parameters)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)
