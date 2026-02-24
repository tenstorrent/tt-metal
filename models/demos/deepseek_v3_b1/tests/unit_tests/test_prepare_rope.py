# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_rope: RoPE transformation matrix and cos/sin creation.

Validates that prepared tensors match the layout and structure expected by
test_rope.py and test_pre_sdpa.py (pre-SDPA op).
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.prepare_rope import (
    ROPE_HEAD_DIM,
    create_rope_trans_mat_tensor,
    get_cos_sin_torch,
    get_rope_trans_mat_core_range_set,
    get_rot_transformation_mat,
)


def test_rotation_matrix_structure():
    """
    Test that the rotation transformation matrix has the correct structure.

    The transformation matrix swaps adjacent pairs with sign changes:
    For input [a, b, c, d, ...], rotation gives [-b, a, -d, c, ...]
    No device required.
    """
    trans_mat = get_rot_transformation_mat()

    assert trans_mat.shape == (1, 1, 32, 32), f"Expected shape (1, 1, 32, 32), got {trans_mat.shape}"

    mat = trans_mat[0, 0]
    for i in range(0, 32, 2):
        assert mat[i, i + 1] == 1, f"Expected mat[{i}, {i+1}] = 1"
        assert mat[i + 1, i] == -1, f"Expected mat[{i+1}, {i}] = -1"


def test_rope_trans_mat_core_range_set():
    """Test that the core range set for trans_mat has the expected size (qrope + kv_rope cores)."""
    # Grid large enough for pre-SDPA layout (12 cols, 10 rows for qrope 8-11,0-7 and kv_rope 8,8-9)
    device_grid_size = ttnn.CoreCoord(12, 10)
    crs = get_rope_trans_mat_core_range_set(device_grid_size)
    num_cores = crs.num_cores()
    # qrope_grid: 4*8 = 32, kv_cache_branch_rope_crs: 2 cores
    assert num_cores == 34, f"Expected 34 cores for trans_mat, got {num_cores}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_create_rope_trans_mat_tensor_pre_sdpa_layout(bh_2d_mesh_device):
    """Test that create_rope_trans_mat_tensor returns a tensor with pre-SDPA layout (HEIGHT_SHARDED L1, 32x32 per core)."""
    device_grid_size = bh_2d_mesh_device.compute_with_storage_grid_size()
    if device_grid_size.x < 12:
        pytest.skip(
            f"Device grid {device_grid_size.x}x{device_grid_size.y} too small; "
            "pre-SDPA trans_mat requires at least 12 columns"
        )

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    trans_mat = create_rope_trans_mat_tensor(submesh)

    assert trans_mat is not None
    # Memory config: HEIGHT_SHARDED, L1
    mem_config = trans_mat.memory_config()
    assert mem_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert mem_config.buffer_type == ttnn.BufferType.L1

    # Shard shape should be (32, 32) per core (API may return list or tuple)
    shard_spec = mem_config.shard_spec
    assert tuple(shard_spec.shape) == (32, 32), f"Expected shard shape (32, 32), got {shard_spec.shape}"

    # Core set size should match get_rope_trans_mat_core_range_set
    crs = get_rope_trans_mat_core_range_set(device_grid_size)
    assert shard_spec.grid.num_cores() == crs.num_cores()


def test_get_cos_sin_torch_shape():
    """Test that get_cos_sin_torch returns tensors with shape [1, 1, max_seq_len, 64]."""
    max_seq_len = 8192
    cos, sin = get_cos_sin_torch(max_seq_len)

    assert cos.shape == (
        1,
        1,
        max_seq_len,
        ROPE_HEAD_DIM,
    ), f"Expected cos shape (1, 1, {max_seq_len}, 64), got {cos.shape}"
    assert sin.shape == (
        1,
        1,
        max_seq_len,
        ROPE_HEAD_DIM,
    ), f"Expected sin shape (1, 1, {max_seq_len}, 64), got {sin.shape}"


def test_get_cos_sin_torch_formula():
    """Test that cos/sin values match the Meta-style formula (base=10000, inv_freq, outer)."""
    max_seq_len = 128
    base = 10000.0
    head_dim = ROPE_HEAD_DIM

    cos, sin = get_cos_sin_torch(max_seq_len, head_dim=head_dim, base=base)

    # Recompute with same formula as in get_cos_sin_torch
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    expected_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    expected_sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)

    cos_squeezed = cos.squeeze(0).squeeze(0)
    sin_squeezed = sin.squeeze(0).squeeze(0)
    torch.testing.assert_close(cos_squeezed.float(), expected_cos)
    torch.testing.assert_close(sin_squeezed.float(), expected_sin)
