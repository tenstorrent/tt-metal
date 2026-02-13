# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.tensor_utils import (
    get_rot_transformation_mat,
    pad_dim_to_size,
    pad_to_shape,
    parse_shard_dims_from_mesh_mapper_config,
    zeros_like_kv_cache,
    zeros_like_paged_cache,
)


def test_pad_dim_to_size():
    """Test the pad_dim_to_size utility function."""

    # Test padding on last dimension
    x = torch.randn(1, 1, 32, 100)
    padded = pad_dim_to_size(x, dim=-1, size=128)
    assert padded.shape == (1, 1, 32, 128)

    # Original data should be preserved
    assert torch.allclose(padded[:, :, :, :100], x)

    # Padding should be zeros
    assert torch.allclose(padded[:, :, :, 100:], torch.zeros(1, 1, 32, 28))

    # Test no padding needed
    x2 = torch.randn(1, 1, 32, 128)
    padded2 = pad_dim_to_size(x2, dim=-1, size=128)
    assert torch.equal(padded2, x2)

    # Test padding on different dimension
    x3 = torch.randn(1, 1, 24, 128)
    padded3 = pad_dim_to_size(x3, dim=-2, size=32)
    assert padded3.shape == (1, 1, 32, 128)

    # Test error when target size is smaller
    with pytest.raises(ValueError, match="smaller than current size"):
        pad_dim_to_size(x, dim=-1, size=50)


def test_pad_dim_to_size_positive_dim():
    """Test pad_dim_to_size with positive dimension index."""

    x = torch.randn(2, 3, 4, 5)

    # Pad dim=0
    padded = pad_dim_to_size(x, dim=0, size=4)
    assert padded.shape == (4, 3, 4, 5)
    assert torch.equal(padded[:2], x)

    # Pad dim=1
    padded = pad_dim_to_size(x, dim=1, size=8)
    assert padded.shape == (2, 8, 4, 5)
    assert torch.equal(padded[:, :3], x)


def test_pad_to_shape():
    """Test the pad_to_shape utility function."""

    # Pad multiple dimensions at once
    x = torch.randn(1, 2, 24, 100)
    padded = pad_to_shape(x, (1, 4, 32, 128))
    assert padded.shape == (1, 4, 32, 128)

    # Original data preserved
    assert torch.allclose(padded[:, :2, :24, :100], x)

    # Padding is zeros
    assert torch.allclose(padded[:, 2:, :, :], torch.zeros(1, 2, 32, 128))
    assert torch.allclose(padded[:, :, 24:, :], torch.zeros(1, 4, 8, 128))
    assert torch.allclose(padded[:, :, :, 100:], torch.zeros(1, 4, 32, 28))


def test_pad_to_shape_no_op():
    """Test pad_to_shape returns same tensor when no padding needed."""

    x = torch.randn(1, 2, 32, 128)
    padded = pad_to_shape(x, (1, 2, 32, 128))
    assert padded is x  # Should be the exact same object


def test_pad_to_shape_single_dim():
    """Test pad_to_shape with only one dimension needing padding."""

    x = torch.randn(2, 3, 4, 5)
    padded = pad_to_shape(x, (2, 3, 4, 8))
    assert padded.shape == (2, 3, 4, 8)
    assert torch.equal(padded[:, :, :, :5], x)


def test_pad_to_shape_error_on_smaller_target():
    """Test pad_to_shape raises error when target is smaller than source."""

    x = torch.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match="smaller than current size"):
        pad_to_shape(x, (2, 3, 4, 3))


def test_parse_shard_dims_from_mesh_mapper_config():
    """Test parsing shard dims from MeshMapperConfig repr.

    This test will fail if TTNN changes the repr format, alerting us to update the parser.
    """

    # Single shard dimension
    config1 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([8]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config1) == [-1]

    # Different shard dimension
    config2 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-2)],
        mesh_shape_override=ttnn.MeshShape([4]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config2) == [-2]

    # Positive dimension
    config3 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(0)],
        mesh_shape_override=ttnn.MeshShape([2]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config3) == [0]

    # Two dimensions sharded (2D mesh)
    config4 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-2), ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([2, 4]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config4) == [-2, -1]

    # Mixed: one sharded, one replicated (only shard dims should be returned)
    config5 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([2, 4]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config5) == [-1]

    # All replicated (no shard dims)
    config6 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate()],
        mesh_shape_override=ttnn.MeshShape([8]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config6) == []


def test_get_rot_transformation_mat():
    """
    Test that get_rot_transformation_mat produces the correct rotation matrix for RoPE.

    The rotation transformation matrix is used by ttnn.experimental.rotary_embedding_llama.
    It has the pattern:
    - rot_emb_matrix[i, i+1] = 1 for even i
    - rot_emb_matrix[i+1, i] = -1 for even i
    """
    result = get_rot_transformation_mat()

    # Validate shape
    assert result.shape == (1, 1, 32, 32), f"Expected shape (1, 1, 32, 32), got {result.shape}"

    # Validate specific known values
    # Position (0, 1) should be 1
    assert result[0, 0, 0, 1].item() == pytest.approx(1.0)
    # Position (1, 0) should be -1
    assert result[0, 0, 1, 0].item() == pytest.approx(-1.0)
    # Position (0, 0) should be 0
    assert result[0, 0, 0, 0].item() == pytest.approx(0.0)
    # Position (2, 3) should be 1
    assert result[0, 0, 2, 3].item() == pytest.approx(1.0)
    # Position (3, 2) should be -1
    assert result[0, 0, 3, 2].item() == pytest.approx(-1.0)
    # Position (30, 31) should be 1
    assert result[0, 0, 30, 31].item() == pytest.approx(1.0)
    # Position (31, 30) should be -1
    assert result[0, 0, 31, 30].item() == pytest.approx(-1.0)

    # Validate that non-pattern positions are 0
    assert result[0, 0, 0, 2].item() == pytest.approx(0.0)
    assert result[0, 0, 1, 1].item() == pytest.approx(0.0)


def test_zeros_like_kv_cache():
    """Test zeros_like_kv_cache creates correct shape tensor."""
    batch_size, n_kv_heads, max_seq_len, head_dim = 32, 8, 2048, 128

    result = zeros_like_kv_cache(batch_size, n_kv_heads, max_seq_len, head_dim)

    assert result.shape == (batch_size, n_kv_heads, max_seq_len, head_dim)
    assert result.dtype == torch.float32
    assert torch.all(result == 0)


def test_zeros_like_paged_cache():
    """Test zeros_like_paged_cache creates correct shape tensor."""
    from dataclasses import dataclass

    @dataclass
    class MockPagedConfig:
        max_num_blocks: int = 64
        block_size: int = 64

    paged_config = MockPagedConfig()
    n_kv_heads = 8
    head_dim = 128

    result = zeros_like_paged_cache(paged_config, n_kv_heads, head_dim)

    assert result.shape == (paged_config.max_num_blocks, n_kv_heads, paged_config.block_size, head_dim)
    assert result.dtype == torch.float32
    assert torch.all(result == 0)


if __name__ == "__main__":
    test_pad_dim_to_size()
    print("  ✓ test_pad_dim_to_size")

    test_pad_dim_to_size_positive_dim()
    print("  ✓ test_pad_dim_to_size_positive_dim")

    test_pad_to_shape()
    print("  ✓ test_pad_to_shape")

    test_pad_to_shape_no_op()
    print("  ✓ test_pad_to_shape_no_op")

    test_pad_to_shape_single_dim()
    print("  ✓ test_pad_to_shape_single_dim")

    test_pad_to_shape_error_on_smaller_target()
    print("  ✓ test_pad_to_shape_error_on_smaller_target")

    test_parse_shard_dims_from_mesh_mapper_config()
    print("  ✓ test_parse_shard_dims_from_mesh_mapper_config")

    test_get_rot_transformation_mat()
    print("  ✓ test_get_rot_transformation_mat")

    test_zeros_like_kv_cache()
    print("  ✓ test_zeros_like_kv_cache")

    test_zeros_like_paged_cache()
    print("  ✓ test_zeros_like_paged_cache")

    print("\nAll tensor_utils tests passed! ✓")
