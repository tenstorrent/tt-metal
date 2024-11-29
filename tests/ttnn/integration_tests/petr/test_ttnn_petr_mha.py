# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_petr.reference import mha

from models.experimental.functional_petr.tt import ttnn_mha


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_petr_mha(device, use_program_cache, reset_seeds):
    torch_model = mha.PETRMultiheadAttention(256, 8)
    torch_model.eval()

    query = torch.rand(900, 1, 256)
    key = torch.rand(6000, 1, 256)
    value = torch.rand(6000, 1, 256)
    key_pos = torch.rand(6000, 1, 256)
    query_pos = torch.rand(900, 1, 256)
    key_padding_mask = torch.zeros(1, 6000)

    torch_output, weight = torch_model(
        query, key=key, value=value, key_pos=key_pos, query_pos=query_pos, key_padding_mask=key_padding_mask
    )
    print(torch_model)

    ttnn_model = ttnn_mha.PETRMultiheadAttention(device, torch_model)

    ttnn_query = ttnn.from_torch(query, dtype=ttnn.bfloat16, device=device)
    ttnn_key = ttnn.from_torch(key, dtype=ttnn.bfloat16, device=device)
    ttnn_value = ttnn.from_torch(value, dtype=ttnn.bfloat16, device=device)
    ttnn_key_pos = ttnn.from_torch(key_pos, dtype=ttnn.bfloat16, device=device)
    ttnn_query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, device=device)
    ttnn_key_padding_mask = ttnn.from_torch(key_padding_mask, device=device)

    ttnn_output, tt_weight = ttnn_model(
        ttnn_query,
        key=ttnn_key,
        value=ttnn_value,
        key_pos=ttnn_key_pos,
        query_pos=ttnn_query_pos,
        key_padding_mask=ttnn_key_padding_mask,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)

    tt_weight = ttnn.to_torch(tt_weight)
    tt_weight = tt_weight.reshape(weight.shape)
    assert_with_pcc(weight, tt_weight, 0.99)
