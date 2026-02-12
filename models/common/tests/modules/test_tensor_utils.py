# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.tensor_utils import pad_dim_to_size, pad_to_shape, parse_shard_dims_from_mesh_mapper_config


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

    print("\nAll tensor_utils tests passed! ✓")
