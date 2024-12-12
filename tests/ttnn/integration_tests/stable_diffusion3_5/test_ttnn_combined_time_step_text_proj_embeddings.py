# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_combined_time_step_text_proj_embeddings import (
    ttnn_CombinedTimestepTextProjEmbeddings as tt_module,
)
from models.experimental.functional_stable_diffusion3_5.reference.combined_time_step_text_proj_embeddings import (
    CombinedTimestepTextProjEmbeddings,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, CombinedTimestepTextProjEmbeddings):
            parameters["text_embedder"] = {}
            parameters["text_embedder"]["linear_1"] = {}
            parameters["text_embedder"]["linear_1"]["weight"] = preprocess_linear_weight(
                model.text_embedder.linear_1.weight, dtype=ttnn.bfloat16
            )
            parameters["text_embedder"]["linear_1"]["bias"] = preprocess_linear_bias(
                model.text_embedder.linear_1.bias, dtype=ttnn.bfloat16
            )
            parameters["text_embedder"]["linear_2"] = {}
            parameters["text_embedder"]["linear_2"]["weight"] = preprocess_linear_weight(
                model.text_embedder.linear_2.weight, dtype=ttnn.bfloat16
            )
            parameters["text_embedder"]["linear_2"]["bias"] = preprocess_linear_bias(
                model.text_embedder.linear_2.bias, dtype=ttnn.bfloat16
            )
            parameters["timestep_embedder"] = {}
            parameters["timestep_embedder"]["linear_1"] = {}
            parameters["timestep_embedder"]["linear_1"]["weight"] = preprocess_linear_weight(
                model.timestep_embedder.linear_1.weight, dtype=ttnn.bfloat16
            )
            parameters["timestep_embedder"]["linear_1"]["bias"] = preprocess_linear_bias(
                model.timestep_embedder.linear_1.bias, dtype=ttnn.bfloat16
            )
            parameters["timestep_embedder"]["linear_2"] = {}
            parameters["timestep_embedder"]["linear_2"]["weight"] = preprocess_linear_weight(
                model.timestep_embedder.linear_2.weight, dtype=ttnn.bfloat16
            )
            parameters["timestep_embedder"]["linear_2"]["bias"] = preprocess_linear_bias(
                model.timestep_embedder.linear_2.bias, dtype=ttnn.bfloat16
            )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "init_inputs,fwd_inputs",
    [
        ((1536, 2048), (2, 2048)),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_combined_time_step_text_proj_embeddings(init_inputs, fwd_inputs, device, reset_seeds):
    torch_sub_module = CombinedTimestepTextProjEmbeddings(
        embedding_dim=init_inputs[0], pooled_projection_dim=init_inputs[1]
    ).to(dtype=torch.bfloat16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_sub_module, device=device, custom_preprocessor=create_custom_preprocessor(device)
    )
    timesteps = torch.tensor([100, 100], dtype=torch.int32)
    pooled_projection = torch.randn(fwd_inputs, dtype=torch.bfloat16)
    tt_input_timesteps = ttnn.from_torch(
        timesteps, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_input_pool_proj = ttnn.from_torch(
        pooled_projection,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_sub_module = tt_module(embedding_dim=init_inputs[0], pooled_projection_dim=init_inputs[1], parameters=parameters)
    tt_out = tt_sub_module(timestep=tt_input_timesteps, pooled_projection=tt_input_pool_proj, device=device)
    torch_out = torch_sub_module(timesteps, pooled_projection)
    tt_out_in_torch = ttnn.to_torch(tt_out)
    assert_with_pcc(torch_out, tt_out_in_torch, 0.99)
