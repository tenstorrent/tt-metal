# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import tt2torch_tensor

"""
Test for nlp_concat_heads_boltz operation: [num_heads, S, S, head_dim] -> [1, S, S, num_heads * head_dim]

Only the interleaved path is covered: the op's output shard spec needs a grid with >= S rows,
so TensorSpec construction rejects sharded inputs for every realistic S.
"""


def concat_heads_boltz_ref(x):
    num_heads, s0, s1, head_dim = x.shape
    return x.permute(1, 2, 0, 3).reshape(1, s0, s1, num_heads * head_dim)


@pytest.mark.parametrize("num_heads, seq_len, head_dim", [(2, 32, 64), (4, 64, 64)])
def test_nlp_concat_heads_boltz(device, num_heads, seq_len, head_dim):
    torch.manual_seed(1234)

    spacers, input_addresses = [], set()
    num_entries_before = device.num_program_cache_entries()
    for i in range(3):
        # A growing live allocation moves every tensor below to a new address on each cache hit, and fresh
        # data per iteration makes a stale address show up as a mismatch.
        spacers.append(ttnn.Tensor(torch.zeros(1, 1, 32, 32 * (i + 1)), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device))
        x = torch.randn((num_heads, seq_len, seq_len, head_dim)).bfloat16().float()
        xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
        input_addresses.add(xt.buffer_address())
        out = ttnn.experimental.nlp_concat_heads_boltz(xt)

        assert list(out.padded_shape) == [1, seq_len, seq_len, num_heads * head_dim]
        assert torch.equal(tt2torch_tensor(out), concat_heads_boltz_ref(x))

    assert len(input_addresses) > 1
    assert device.num_program_cache_entries() - num_entries_before == 1
