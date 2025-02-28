# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.gelu import GELU
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_gelu import ttnn_GELU


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, GELU):
            parameters["proj"] = {}
            parameters["proj"]["weight"] = preprocess_linear_weight(model.proj.weight, dtype=ttnn.bfloat8_b)
            parameters["proj"]["bias"] = preprocess_linear_bias(model.proj.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "hidden_states_shape",
    [
        # ([2, 4096, 1536]),
        # ([2, 333, 1536]),
        ([2, 1024, 1536]),
        ([2, 160, 1536]),
    ],
)
def test_gelu(device, hidden_states_shape, reset_seeds):
    reference_model = GELU(dim_in=1536, dim_out=6144, approximate="tanh", bias=True).to(dtype=torch.bfloat16)
    reference_model.eval()
    torch_hidden_states = torch.randn(hidden_states_shape, dtype=torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(None), device=device
    )

    torch_output = reference_model(hidden_states=torch_hidden_states)
    ttnn_model = ttnn_GELU(dim_in=1536, dim_out=6144)

    mm_a_y = 8
    mm_a_x = 8
    mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
    mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
    if torch_hidden_states.shape[-2] < 512:
        mm_a_y = 6
        mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
        mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    input_memory_config = ttnn.create_sharded_memory_config(
        torch_hidden_states.shape,
        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        strategy=mm_a_x_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=input_memory_config,  # ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn_model(hidden_states=ttnn_hidden_states, parameters=parameters)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
