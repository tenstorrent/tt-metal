# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Legacy SDPA validation of op-selected (zero) chunk sizes, on a single device.

Kept apart from test_sdpa_recipe_blocking.py, whose two-device fabric meshes should not be interleaved
with plain single-device opens."""

import pytest
import torch
import ttnn


def test_legacy_zero_chunks_raise_not_crash(device):
    """q/k chunk sizes default to 0 (op-selected) but only recipes resolve them. Legacy entry points
    that skip the dispatcher's recipe check (e.g. chunked prefill) must reject 0 in validation instead
    of dividing by it (found on Wormhole: SIGFPE in chunk_start_idx % q_chunk_size)."""
    b, nh, s, d, block = 1, 1, 128, 64, 32
    blocks = s // block
    q = ttnn.from_torch(torch.randn(b, nh, 64, d), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    paged = ttnn.from_torch(torch.randn(blocks, nh, block, d), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table = ttnn.from_torch(torch.arange(blocks, dtype=torch.int32).reshape(b, blocks), device=device, dtype=ttnn.int32)
    config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=device.compute_with_storage_grid_size())
    assert config.q_chunk_size == 0 and config.k_chunk_size == 0
    with pytest.raises(RuntimeError):
        ttnn.transformer.chunked_scaled_dot_product_attention(q, paged, paged, page_table, 0, program_config=config)

