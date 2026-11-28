# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.experimental.quasar.transformer import joint_scaled_dot_product_attention, scaled_dot_product_attention


def _tensors(device, count, heads=1):
    return [
        ttnn.from_torch(torch.randn(1, heads, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
        for _ in range(count)
    ]


def _program_config(q_chunk_size=32, k_chunk_size=32, grid=(1, 1)):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size
    )


# Chunk sizes, head counts and the grid are used as host-side divisors, so zero must raise a RuntimeError.
@pytest.mark.parametrize("q_chunk_size, k_chunk_size, name", [(0, 32, "q_chunk_size"), (32, 0, "k_chunk_size")])
def test_quasar_sdpa_zero_chunk_size(device, expect_error, q_chunk_size, k_chunk_size, name):
    q, k, v = _tensors(device, 3)
    with expect_error(RuntimeError, f"{name} must be a positive multiple of TILE_SIZE"):
        scaled_dot_product_attention(
            q, k, v, is_causal=True, program_config=_program_config(q_chunk_size, k_chunk_size)
        )


def test_quasar_sdpa_zero_k_heads(device, expect_error):
    (q,) = _tensors(device, 1)
    k, v = _tensors(device, 2, heads=0)
    with expect_error(RuntimeError, "K num_heads must be greater than 0"):
        scaled_dot_product_attention(q, k, v, is_causal=False)


@pytest.mark.parametrize("q_chunk_size, k_chunk_size, name", [(0, 32, "q_chunk_size"), (32, 0, "k_chunk_size")])
def test_quasar_joint_sdpa_zero_chunk_size(device, expect_error, q_chunk_size, k_chunk_size, name):
    tensors = _tensors(device, 6)
    with expect_error(RuntimeError, f"{name} must be a positive multiple of TILE_SIZE"):
        joint_scaled_dot_product_attention(
            *tensors, joint_strategy="rear", program_config=_program_config(q_chunk_size, k_chunk_size)
        )


def test_quasar_joint_sdpa_zero_heads(device, expect_error):
    tensors = _tensors(device, 6, heads=0)
    with expect_error(RuntimeError, "Q num_heads must be equal to K num_heads, and greater than 0"):
        joint_scaled_dot_product_attention(*tensors, joint_strategy="rear", program_config=_program_config())


@pytest.mark.parametrize("grid", [(0, 1), (1, 0)])
def test_quasar_joint_sdpa_zero_grid(device, expect_error, grid):
    tensors = _tensors(device, 6)
    with expect_error(RuntimeError, "compute_with_storage_grid_size must be non-zero"):
        joint_scaled_dot_product_attention(*tensors, joint_strategy="rear", program_config=_program_config(grid=grid))
