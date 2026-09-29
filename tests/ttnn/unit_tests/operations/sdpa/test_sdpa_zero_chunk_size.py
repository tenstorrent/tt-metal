# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# A chunk size of 0 passed the tile-multiple check and then reached an integer division on the
# host, which killed the process with SIGFPE instead of raising.
@pytest.mark.parametrize("q_chunk_size, k_chunk_size, name", [(0, 32, "q_chunk_size"), (32, 0, "k_chunk_size")])
def test_sdpa_zero_chunk_size(device, expect_error, q_chunk_size, k_chunk_size, name):
    q, k, v = (
        ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device) for _ in range(3)
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1), q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size
    )
    with expect_error(RuntimeError, f"{name} must be a positive multiple of TILE_SIZE"):
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=program_config)
