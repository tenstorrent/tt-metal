# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


def _ref_split_heads(x, num_heads):
    batch, _, seq, hidden = x.shape
    head_dim = hidden // (3 * num_heads)
    q, k, v = torch.split(x, num_heads * head_dim, dim=-1)
    q = q.reshape(batch, seq, num_heads, head_dim).transpose(-3, -2)
    k = k.reshape(batch, seq, num_heads, head_dim).transpose(-3, -2).transpose(-2, -1)
    v = v.reshape(batch, seq, num_heads, head_dim).transpose(-3, -2)
    return q, k, v


@pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_split_query_key_value_and_split_heads_interleaved(device, batch, num_heads, dtype):
    """Interleaved path: writer must loop out_c heads, not a hardcoded 16."""
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < batch:
        pytest.skip(f"Grid size {compute_grid_size} is not supported")

    torch.manual_seed(1234)
    seq_len = 32 * min(12, compute_grid_size.x)
    head_dim = 64
    hidden = 3 * num_heads * head_dim
    torch_input = torch.randn(batch, 1, seq_len, hidden)

    xt = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q, k, v = ttnn.experimental.split_query_key_value_and_split_heads(
        xt,
        ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_heads=num_heads,
    )

    assert list(q.padded_shape) == [batch, num_heads, seq_len, head_dim]
    assert list(k.padded_shape) == [batch, num_heads, head_dim, seq_len]
    assert list(v.padded_shape) == [batch, num_heads, seq_len, head_dim]

    ref_q, ref_k, ref_v = _ref_split_heads(torch_input, num_heads)
    for name, got, ref in (
        ("Q", ttnn.to_torch(q), ref_q),
        ("K", ttnn.to_torch(k), ref_k),
        ("V", ttnn.to_torch(v), ref_v),
    ):
        p, o = comp_pcc(got, ref)
        logger.info(f"{name} {o}")
        assert p, f"{name} mismatch with num_heads={num_heads}"
