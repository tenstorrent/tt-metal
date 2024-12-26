# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_pix_art_alpha_text_projection import (
    ttnn_PixArtAlphaTextProjection as tt_module,
)
from models.experimental.functional_stable_diffusion3_5.reference.pix_art_alpha_text_projection import (
    PixArtAlphaTextProjection,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull

torch.manual_seed(42)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, PixArtAlphaTextProjection):
            parameters["linear_1"] = {}
            parameters["linear_1"]["weight"] = preprocess_linear_weight(model.linear_1.weight, dtype=ttnn.bfloat8_b)
            parameters["linear_1"]["bias"] = preprocess_linear_bias(model.linear_1.bias, dtype=ttnn.bfloat8_b)
            parameters["linear_2"] = {}
            parameters["linear_2"]["weight"] = preprocess_linear_weight(model.linear_2.weight, dtype=ttnn.bfloat8_b)
            parameters["linear_2"]["bias"] = preprocess_linear_bias(model.linear_2.bias, dtype=ttnn.bfloat8_b)
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "in_features,hidden_size,out_features,fwd_input",
    [
        (2048, 1536, None, (2, 2048)),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_px_art_alpha_text(in_features, hidden_size, out_features, fwd_input, device):
    torch_sub_module = PixArtAlphaTextProjection(
        in_features=in_features, hidden_size=hidden_size, out_features=out_features
    ).to(dtype=torch.bfloat16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_sub_module, device=device, custom_preprocessor=create_custom_preprocessor(device)
    )
    torch_input = torch.randn([2, 2048], dtype=torch.bfloat16)
    torch_input_unsqueezed = torch_input.unsqueeze(1).unsqueeze(1)
    tt_input = ttnn.from_torch(
        torch_input_unsqueezed,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_sub_module = tt_module(parameters)
    tt_out = tt_sub_module(tt_input, device=device)
    torch_out = torch_sub_module(torch_input).unsqueeze(1).unsqueeze(1)
    tt_out_in_torch = ttnn.to_torch(tt_out)
    assert_with_pcc(torch_out, tt_out_in_torch, 0.99)
