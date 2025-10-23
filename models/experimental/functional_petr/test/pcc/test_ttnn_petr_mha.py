# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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

    # Explanation:
    # The following code creates random input tensors for testing the PETR multihead attention:
    #   query = torch.rand(900, 1, 256)
    #   key = torch.rand(6000, 1, 256)
    #   value = torch.rand(6000, 1, 256)
    #   key_pos = torch.rand(6000, 1, 256)
    #   query_pos = torch.rand(900, 1, 256)
    #   key_padding_mask = torch.zeros(1, 6000)
    #
    # The shape (900, 1, 256) for the query means:
    #   - 900 = the number of queries (often called query tokens or query vectors). In detection,
    #       this is the number of object proposals/slots (num_queries).
    #   - 1 = batch size (or extra singleton dimension, depends on how the module expects batches).
    #   - 256 = the embedding dimension or feature size for each query vector.
    # The key/value/positional tensors are sized similarly, with 6000 denoting the number of key tokens
    # (e.g., all spatial locations "flattened" from input features or all candidates the model attends to).
    # The size of key_padding_mask (1, 6000) means there's one batch and each of the 6000 key positions
    # is unmasked (zeros).
    #
    # So, yes: "900 tokens" refers to the number of query vectors, each of size 256.
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

    tt_weight = ttnn.to_torch(tt_weight).to(torch.float32)
    tt_weight = tt_weight.reshape(weight.shape)

    passed, msg = check_with_pcc(weight, tt_weight, pcc=0.99)
    print(f"Weight PCC: {msg}")

    assert_with_pcc(weight, tt_weight, 0.99)
