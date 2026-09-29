# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# A chunk size of 0 passed the tile-multiple check, and a K tensor with 0 heads passed the head
# relationship check; both then reached an integer division on the host, which killed the process
# with SIGFPE instead of raising.


def _tensors(device, count, heads=1):
    return [
        ttnn.from_torch(torch.randn(1, heads, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
        for _ in range(count)
    ]


_ZERO_CHUNK = pytest.mark.parametrize(
    "q_chunk_size, k_chunk_size, name", [(0, 32, "q_chunk_size"), (32, 0, "k_chunk_size")]
)


@_ZERO_CHUNK
def test_sdpa_zero_chunk_size(device, expect_error, q_chunk_size, k_chunk_size, name):
    q, k, v = _tensors(device, 3)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1), q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size
    )
    with expect_error(RuntimeError, f"{name} must be a positive multiple of TILE_SIZE"):
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=program_config)


@_ZERO_CHUNK
def test_joint_sdpa_zero_chunk_size(device, expect_error, q_chunk_size, k_chunk_size, name):
    q, k, v, joint_q, joint_k, joint_v = _tensors(device, 6)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1), q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size
    )
    with expect_error(RuntimeError, f"{name} must be a positive multiple of TILE_SIZE"):
        ttnn.transformer.joint_scaled_dot_product_attention(
            q, k, v, joint_q, joint_k, joint_v, joint_strategy="rear", program_config=program_config
        )


def test_sdpa_zero_k_heads(device, expect_error):
    (q,) = _tensors(device, 1)
    k, v = _tensors(device, 2, heads=0)
    with expect_error(RuntimeError, "Q num_heads must be >= K num_heads"):
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)


def test_joint_sdpa_zero_heads(device, expect_error):
    q, k, v, joint_q, joint_k, joint_v = _tensors(device, 6, heads=0)
    program_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(1, 1), q_chunk_size=32, k_chunk_size=32)
    with expect_error(RuntimeError, "Q num_heads must be equal to K num_heads, and greater than 0"):
        ttnn.transformer.joint_scaled_dot_product_attention(
            q, k, v, joint_q, joint_k, joint_v, joint_strategy="rear", program_config=program_config
        )
