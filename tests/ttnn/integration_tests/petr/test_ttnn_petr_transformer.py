# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_petr.reference import petr_transformer
from models.experimental.functional_petr.tt.model_preprocessing import (
    create_petr_transformer_input_tensors,
    create_petr_transformer_model_parameters,
)
from models.experimental.functional_petr.tt import ttnn_petr_transformer


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_petr_transformer(device, use_program_cache, reset_seeds):
    (
        x,
        mask,
        query_embed,
        pos_embed,
        ttnn_x,
        ttnn_mask,
        ttnn_query_embed,
        ttnn_pos_embed,
    ) = create_petr_transformer_input_tensors(device)

    torch_model = petr_transformer.PETRTransformer()
    torch_model.eval()
    torch_output, torch_memory = torch_model(x, mask, query_embed, pos_embed)

    parameters = create_petr_transformer_model_parameters(torch_model, x, mask, query_embed, pos_embed, device=device)

    ttnn_model = ttnn_petr_transformer.PETRTransformer(device, parameters)
    ttnn_output, ttnn_memory = ttnn_model(device, ttnn_x, ttnn_mask, ttnn_query_embed, ttnn_pos_embed)

    ttnn_memory = ttnn_memory.reshape(torch_memory.shape)
    assert_with_pcc(torch_memory, ttnn_memory, 0.99)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
