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
            parameters["linear"]["weight"] = preprocess_linear_weight(model.linear.weight, dtype=ttnn.bfloat8_b)
            parameters["linear"]["bias"] = preprocess_linear_bias(model.linear.bias, dtype=ttnn.bfloat8_b)

            # Its none as elementwise_affine=False
            parameters["norm"] = {}

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "h",
    ((4096),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_sd35_ada_layernorm_zerox(device, reset_seeds, h):
    reference_model = SD35AdaLayerNormZeroX(
        embedding_dim=1536,
        norm_type="layer_norm",
        bias=True,
    ).to(dtype=torch.bfloat16)
    reference_model.eval()

    torch_input_hidden_states = torch.randn(2, h, 1536, dtype=torch.bfloat16)
    torch_input_emb = torch.randn(2, 1536, dtype=torch.bfloat16)
    torch_output = reference_model(torch_input_hidden_states, torch_input_emb)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    torch_input_hidden_states = torch_input_hidden_states.unsqueeze(1)

    mm_a_y = 8
    mm_a_x = 8
    mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
    mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
    if torch_input_hidden_states.shape[-2] < 512:
        mm_a_y = 6
        mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
        mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    input_memory_config = ttnn.create_sharded_memory_config(
        torch_input_hidden_states.shape,
        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        strategy=mm_a_x_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    ttnn_input_emb = ttnn.from_torch(
        torch_input_emb.unsqueeze(1).unsqueeze(1),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_input_hidden_states = ttnn.from_torch(
        torch_input_hidden_states,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=input_memory_config,  # ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_model = ttnn_SD35AdaLayerNormZeroX(
        embedding_dim=1536,
        norm_type="layer_norm",
        bias=True,
    )

    ttnn_output = ttnn_model(ttnn_input_hidden_states, ttnn_input_emb, parameters=parameters)

    for i in range(len(torch_output)):
        torch_output_shape = torch_output[i].shape
        if len(torch_output_shape) > 2:
            assert_with_pcc(torch_output[i].unsqueeze(1), ttnn.to_torch(ttnn_output[i]), pcc=0.99)
        else:
            assert_with_pcc(
                torch_output[i].unsqueeze(1).unsqueeze(1),
                ttnn.to_torch(ttnn_output[i]),
                pcc=0.99,
            )
