# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.ada_layernorm_zero import AdaLayerNormZero
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_ada_layernorm_zero import ttnn_AdaLayerNormZero


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, AdaLayerNormZero):
            parameters["linear"] = {}
            parameters["linear"]["weight"] = preprocess_linear_weight(model.linear.weight, dtype=ttnn.bfloat8_b)
            parameters["linear"]["bias"] = preprocess_linear_bias(model.linear.bias, dtype=ttnn.bfloat8_b)

            # Its none as elementwise_affine=False
            parameters["norm"] = {}

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "x_shape",
    [
        ([2, 4096, 1536]),
        # ([2, 333, 1536]),
        # ([2, 1024, 1536]),
        ([2, 160, 1536]),
    ],
)
def test_ada_layernorm_zero(device, x_shape, reset_seeds):
    reference_model = AdaLayerNormZero(embedding_dim=1536, num_embeddings=None, norm_type="layer_norm", bias=True).to(
        dtype=torch.bfloat16
    )

    reference_model.eval()
    torch_input_x = torch.randn(x_shape, dtype=torch.bfloat16)
    torch_innput_emb = torch.randn(2, 1536, dtype=torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )
    torch_output = reference_model(
        x=torch_input_x, timestep=None, class_labels=None, hidden_dtype=None, emb=torch_innput_emb
    )

    torch_input_x_unsqueezed = torch_input_x.unsqueeze(1)

    # if torch_input_x_unsqueezed.shape[-2] < 512:
    #     input_memory_config = ttnn.L1_MEMORY_CONFIG
    # else:
    #     mm_a_y = 8
    #     mm_a_x = 8
    #     mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
    #     mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

    #     input_memory_config = ttnn.create_sharded_memory_config(
    #         torch_input_x_unsqueezed.shape,
    #         core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
    #         strategy=mm_a_x_strategy,
    #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #     )

    ttnn_input_emb = ttnn.from_torch(
        torch_innput_emb.unsqueeze(1).unsqueeze(1),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_input_x = ttnn.from_torch(
        torch_input_x_unsqueezed,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # input_memory_config
    )

    ttnn_model = ttnn_AdaLayerNormZero(embedding_dim=1536, num_embeddings=None, norm_type="layer_norm", bias=True)

    ttnn_output = ttnn_model(
        ttnn_input_x, timestep=None, class_labels=None, hidden_dtype=None, emb=ttnn_input_emb, parameters=parameters
    )

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
