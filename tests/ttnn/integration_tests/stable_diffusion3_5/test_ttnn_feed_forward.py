# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias


from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.feed_forward import FeedForward
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_feed_forward import ttnn_FeedForward
from tests.ttnn.integration_tests.stable_diffusion3_5.test_ttnn_gelu import (
    create_custom_preprocessor as create_custom_preprocessor_gelu,
)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, FeedForward):
            parameters["net"] = {}
            parameters["net"][0] = {}
            gelu_custom_preprocessor = create_custom_preprocessor_gelu(device)
            parameters["net"][0] = gelu_custom_preprocessor(model.net[0], None, None)
            parameters["net"][1] = {}
            parameters["net"][2] = {}
            parameters["net"][2]["weight"] = preprocess_linear_weight(model.net[2].weight, dtype=ttnn.bfloat16)
            parameters["net"][2]["bias"] = preprocess_linear_bias(model.net[2].bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "hidden_states_shape",
    [
        ([2, 4096, 1536]),
        ([2, 333, 1536]),
    ],
)
def test_feed_forward(device, hidden_states_shape, reset_seeds):
    reference_model = FeedForward(
        dim=1536,
        dim_out=1536,
        mult=4,
        dropout=0.0,
        activation_fn="gelu-approximate",
        final_dropout=False,
        inner_dim=None,
        bias=True,
    ).to(dtype=torch.bfloat16)
    reference_model.eval()

    torch_hidden_states = torch.randn(hidden_states_shape, dtype=torch.bfloat16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    torch_output = reference_model(hidden_states=torch_hidden_states)
    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_model = ttnn_FeedForward(
        dim=1536, dim_out=1536, mult=4, activation_fn="gelu-approximate", inner_dim=None, bias=True
    )

    ttnn_output = ttnn_model(hidden_states=ttnn_hidden_states, parameters=parameters)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
