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
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.sd35_ada_layernorm_zerox import SD35AdaLayerNormZeroX
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_sd35_ada_layernorm_zerox import (
    ttnn_SD35AdaLayerNormZeroX,
)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SD35AdaLayerNormZeroX):
            parameters["linear"] = {}
            parameters["linear"]["weight"] = preprocess_linear_weight(model.linear.weight, dtype=ttnn.bfloat16)
            parameters["linear"]["bias"] = preprocess_linear_bias(model.linear.bias, dtype=ttnn.bfloat16)

            # Its none as elementwise_affine=False
            parameters["norm"] = {}

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_sd35_ada_layernorm_zerox(device, reset_seeds):
    reference_model = SD35AdaLayerNormZeroX(
        embedding_dim=1536,
        norm_type="layer_norm",
        bias=True,
    ).to(dtype=torch.bfloat16)
    reference_model.eval()

    torch_innput_hidden_states = torch.randn(2, 4096, 1536, dtype=torch.bfloat16)
    torch_innput_emb = torch.randn(2, 1536, dtype=torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    ttnn_input_hidden_states = ttnn.from_torch(
        torch_innput_hidden_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_input_emb = ttnn.from_torch(torch_innput_emb, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    torch_output = reference_model(torch_innput_hidden_states, torch_innput_emb)
    ttnn_model = ttnn_SD35AdaLayerNormZeroX(
        embedding_dim=1536,
        norm_type="layer_norm",
        bias=True,
    )

    ttnn_output = ttnn_model(ttnn_input_hidden_states, ttnn_input_emb, parameters=parameters)

    for i in range(len(torch_output)):
        assert_with_pcc(torch_output[i], ttnn.to_torch(ttnn_output[i]), pcc=0.99)
