# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.experimental.functional_petr.reference import mha

from models.experimental.functional_petr.tt import ttnn_mha


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_petr_mha(device, reset_seeds):
    torch_model = mha.PETRMultiheadAttention(256, 8)
    torch_model.eval()

    query = torch.rand(900, 1, 256)
    key = torch.rand(6000, 1, 256)
    value = torch.rand(6000, 1, 256)
    key_pos = torch.rand(6000, 1, 256)
    query_pos = torch.rand(900, 1, 256)
    key_padding_mask = torch.zeros(1, 6000)
    print(f"Query stats: mean={query.mean():.6f}, std={query.std():.6f}")
    print(f"Key stats: mean={key.mean():.6f}, std={key.std():.6f}")

    torch_output, weight = torch_model(
        query, key=key, value=value, key_pos=key_pos, query_pos=query_pos, key_padding_mask=key_padding_mask
    )
    print(f"Torch output stats: mean={torch_output.mean():.6f}, std={torch_output.std():.6f}")
    print(f"Torch weight stats: mean={weight.mean():.6f}, std={weight.std():.6f}")

    print(torch_model)

    ttnn_model = ttnn_mha.TTPETRMultiheadAttention(device, torch_model)

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
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)
    print(f"Output PCC: {msg}")

    assert_with_pcc(torch_output, ttnn_output, 0.99)
