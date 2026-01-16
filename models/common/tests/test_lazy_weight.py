# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for LazyWeight module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import ttnn
from models.common.modules.lazy_weight import (
    LazyWeight,
    _auto_pad_for_sharding,
    _from_torch_and_dump,
    resolve_lazy_weight,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_tensor():
    """Create a mock torch tensor for testing."""
    return torch.randn(1, 32, 128)


@pytest.fixture
def mock_tensor_2d():
    """Create a 2D mock torch tensor for testing."""
    return torch.randn(64, 128)


@pytest.fixture
def mock_mesh_device():
    """Create a mock MeshDevice for unit tests."""
    device = MagicMock(spec=ttnn.MeshDevice)
    device.get_num_devices.return_value = 8
    device.id.return_value = "test_device_0"
    return device


@pytest.fixture
def mock_memory_config():
    """Create a mock MemoryConfig."""
    return ttnn.DRAM_MEMORY_CONFIG


@pytest.fixture
def mock_mesh_mapper_config():
    """Create a mock MeshMapperConfig for sharding."""
    return ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([8]),
    )


# ============================================================================
# Test LazyWeight Construction
# ============================================================================


class TestLazyWeightConstruction:
    """Tests for LazyWeight dataclass construction and __post_init__."""

    def test_construction_with_tensor(self, mock_tensor):
        """Test basic construction with a tensor."""
        lw = LazyWeight(source=mock_tensor)
        assert lw.source is mock_tensor
        assert lw.dtype == ttnn.bfloat16
        assert lw.pad_value == 0.0
        assert lw.device is None
        assert lw.mesh_mapper_config is None
        assert lw.memory_config is None
        assert lw.layout is None
        assert lw._value is None

    def test_construction_with_all_parameters(
        self, mock_tensor, mock_mesh_device, mock_memory_config, mock_mesh_mapper_config
    ):
        """Test construction with all parameters specified."""
        cache_dir = Path("/tmp/cache")
        lw = LazyWeight(
            source=mock_tensor,
            cache_dir_weight_name=(cache_dir, "test_weight"),
            pad_value=1.0,
            dtype=ttnn.bfloat4_b,
            device=mock_mesh_device,
            mesh_mapper_config=mock_mesh_mapper_config,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        assert lw.source is mock_tensor
        assert lw.cache_dir_weight_name == (cache_dir, "test_weight")
        assert lw.pad_value == 1.0
        assert lw.dtype == ttnn.bfloat4_b
        assert lw.device is mock_mesh_device
        assert lw.mesh_mapper_config is mock_mesh_mapper_config
        assert lw.memory_config is mock_memory_config
        assert lw.layout == ttnn.TILE_LAYOUT

    def test_post_init_validates_shape(self):
        """Test that __post_init__ validates tensor shape."""
        mock_source = MagicMock()
        mock_source.shape = None

        with pytest.raises(AssertionError, match="source must have a shape"):
            LazyWeight(source=mock_source)

    def test_post_init_validates_empty_shape(self):
        """Test that __post_init__ rejects empty shape."""
        mock_source = MagicMock()
        mock_source.shape = ()

        with pytest.raises(AssertionError, match="source must have a shape"):
            LazyWeight(source=mock_source)


# ============================================================================
# Test is_resolved Method
# ============================================================================


class TestIsResolved:
    """Tests for LazyWeight.is_resolved() method."""

    def test_not_resolved_missing_device(self, mock_tensor, mock_memory_config, mock_mesh_mapper_config):
        """Test is_resolved returns False when device is missing."""
        lw = LazyWeight(
            source=mock_tensor,
            memory_config=mock_memory_config,
            mesh_mapper_config=mock_mesh_mapper_config,
            layout=ttnn.TILE_LAYOUT,
        )
        assert not lw.is_resolved()

    def test_not_resolved_missing_layout(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test is_resolved returns False when layout is missing."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
        )
        assert not lw.is_resolved()

    def test_not_resolved_missing_memory_config(self, mock_tensor, mock_mesh_device):
        """Test is_resolved returns False when memory_config is missing."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        assert not lw.is_resolved()

    def test_is_resolved_with_all_required_fields(
        self, mock_tensor, mock_mesh_device, mock_memory_config, mock_mesh_mapper_config
    ):
        """Test is_resolved returns True when all required fields are set."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            mesh_mapper_config=mock_mesh_mapper_config,
            layout=ttnn.TILE_LAYOUT,
        )
        assert lw.is_resolved()

    def test_is_resolved_with_none_mesh_mapper_config(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test is_resolved returns False when mesh_mapper_config is None (replicated case requires it to be explicitly set)."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper_config=None,  # None means not resolved per is_resolved check
        )
        # mesh_mapper_config is in required_fields so None means not resolved
        assert not lw.is_resolved()


# ============================================================================
# Test _get_fingerprint Method
# ============================================================================


class TestGetFingerprint:
    """Tests for LazyWeight._get_fingerprint() method."""

    def test_fingerprint_includes_source_shape(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint includes source shape."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        fingerprint = lw._get_fingerprint()
        assert "srcshape_1_32_128" in fingerprint

    def test_fingerprint_includes_non_default_memory_config(self, mock_tensor, mock_mesh_device):
        """Test that fingerprint includes memory_config hash when non-default."""
        # Use L1_MEMORY_CONFIG which is different from DRAM_MEMORY_CONFIG
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        fingerprint = lw._get_fingerprint()
        assert "memcfg_" in fingerprint

    def test_fingerprint_includes_dtype(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint includes dtype."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        fingerprint = lw._get_fingerprint()
        assert "dtype_BFLOAT16" in fingerprint

    def test_fingerprint_includes_layout(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint includes layout."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        fingerprint = lw._get_fingerprint()
        assert "layout_TILE" in fingerprint

    def test_fingerprint_includes_pad_value_when_non_default(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint includes pad_value when non-default."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            pad_value=1.5,
        )
        fingerprint = lw._get_fingerprint()
        assert "pad_1.5" in fingerprint

    def test_fingerprint_excludes_pad_value_when_default(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint excludes pad_value when default (0.0)."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            pad_value=0.0,
        )
        fingerprint = lw._get_fingerprint()
        assert "pad_" not in fingerprint

    def test_fingerprint_includes_device_id(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint includes device id."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        fingerprint = lw._get_fingerprint()
        assert "device_" in fingerprint

    def test_fingerprint_device_without_id_method(self, mock_tensor, mock_memory_config):
        """Test fingerprint uses 'single' when device has no id() method."""
        device = MagicMock()
        del device.id  # Remove id attribute
        lw = LazyWeight(
            source=mock_tensor,
            device=device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        fingerprint = lw._get_fingerprint()
        assert "device_single" in fingerprint

    def test_fingerprint_changes_with_different_dtype(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint changes with different dtype."""
        lw1 = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        lw2 = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat4_b,
        )
        assert lw1._get_fingerprint() != lw2._get_fingerprint()


# ============================================================================
# Test _get_mesh_mapper_fingerprint Method
# ============================================================================


class TestGetMeshMapperFingerprint:
    """Tests for LazyWeight._get_mesh_mapper_fingerprint() method."""

    def test_fingerprint_replicated_when_none(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that fingerprint is 'replicated' when mesh_mapper_config is None."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper_config=None,
        )
        fingerprint = lw._get_mesh_mapper_fingerprint()
        assert fingerprint == "replicated"

    def test_fingerprint_with_mesh_mapper_config(
        self, mock_tensor, mock_mesh_device, mock_memory_config, mock_mesh_mapper_config
    ):
        """Test that fingerprint includes mapper hash when config is set."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper_config=mock_mesh_mapper_config,
        )
        fingerprint = lw._get_mesh_mapper_fingerprint()
        assert fingerprint.startswith("mapper_")
        # Hash should be 12 characters
        assert len(fingerprint) == len("mapper_") + 12

    def test_different_configs_produce_different_fingerprints(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that different mesh_mapper_configs produce different fingerprints."""
        config1 = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape([8]),
        )
        config2 = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-2)],
            mesh_shape_override=ttnn.MeshShape([8]),
        )
        lw1 = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper_config=config1,
        )
        lw2 = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper_config=config2,
        )
        assert lw1._get_mesh_mapper_fingerprint() != lw2._get_mesh_mapper_fingerprint()


# ============================================================================
# Test _get_cache_fill_path Method
# ============================================================================


class TestGetCacheFillPath:
    """Tests for LazyWeight._get_cache_fill_path() method."""

    def test_returns_none_when_cache_dir_none(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that _get_cache_fill_path returns None when cache_dir is None."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        result = lw._get_cache_fill_path(cache_dir=None, weight_name="test")
        assert result is None

    def test_returns_none_when_weight_name_none(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that _get_cache_fill_path returns None when weight_name is None."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        result = lw._get_cache_fill_path(cache_dir=Path("/tmp/cache"), weight_name=None)
        assert result is None

    def test_returns_path_with_fingerprint(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that _get_cache_fill_path returns correct path with fingerprint."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        cache_dir = Path("/tmp/cache")
        result = lw._get_cache_fill_path(cache_dir=cache_dir, weight_name="my_weight")
        assert result is not None
        assert result.parent == cache_dir
        assert result.name.startswith("my_weight_")
        assert result.suffix == ".tensorbin"


# ============================================================================
# Test padded_shape Property
# ============================================================================


class TestPaddedShape:
    """Tests for LazyWeight.padded_shape property."""

    def test_padded_shape_raises_when_not_resolved(self, mock_tensor, mock_mesh_device):
        """Test that padded_shape raises assertion when not resolved."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            # Missing memory_config and layout
        )
        with pytest.raises(AssertionError, match="LazyWeight must be resolved"):
            _ = lw.padded_shape

    def test_padded_shape_raises_when_device_none(self, mock_tensor, mock_memory_config, mock_mesh_mapper_config):
        """Test that padded_shape raises ValueError when device is None."""
        lw = LazyWeight(
            source=mock_tensor,
            device=None,
            memory_config=mock_memory_config,
            mesh_mapper_config=mock_mesh_mapper_config,
            layout=ttnn.TILE_LAYOUT,
        )
        # First it will fail is_resolved check
        with pytest.raises(AssertionError, match="LazyWeight must be resolved"):
            _ = lw.padded_shape

    def test_padded_shape_no_padding_when_replicated(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that padded_shape returns original shape when replicated (mesh_mapper_config is None)."""
        # Need to set mesh_mapper_config to something to pass is_resolved - use a mock
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper_config=None,
        )
        # Override is_resolved to return True for this test
        with patch.object(lw, "is_resolved", return_value=True):
            result = lw.padded_shape
            assert result == tuple(mock_tensor.shape)

    def test_padded_shape_no_padding_single_device(self, mock_tensor, mock_memory_config, mock_mesh_mapper_config):
        """Test that padded_shape returns original shape for single device."""
        single_device = MagicMock(spec=ttnn.MeshDevice)
        single_device.get_num_devices.return_value = 1

        lw = LazyWeight(
            source=mock_tensor,
            device=single_device,
            memory_config=mock_memory_config,
            mesh_mapper_config=mock_mesh_mapper_config,
            layout=ttnn.TILE_LAYOUT,
        )
        result = lw.padded_shape
        assert result == tuple(mock_tensor.shape)

    def test_padded_shape_pads_for_sharding(self, mock_memory_config):
        """Test that padded_shape correctly pads for sharding across multiple devices."""
        # Create tensor with shape that needs padding: dim -1 is 100
        # With 8 devices, shard_dim = 100/8 = 12.5, needs to be tile aligned (32)
        # padded_shard = ceil(12.5/32)*32 = 32
        # padded_hidden = 32 * 8 = 256
        tensor = torch.randn(1, 32, 100)
        device = MagicMock(spec=ttnn.MeshDevice)
        device.get_num_devices.return_value = 8

        config = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape([8]),
        )

        lw = LazyWeight(
            source=tensor,
            device=device,
            memory_config=mock_memory_config,
            mesh_mapper_config=config,
            layout=ttnn.TILE_LAYOUT,
        )
        result = lw.padded_shape
        # Original: (1, 32, 100), padded dim -1 should be 256 (32*8 where 32 is tile-aligned shard)
        assert result == (1, 32, 256)

    def test_padded_shape_raises_valueerror_when_device_none_but_resolved(
        self, mock_tensor, mock_memory_config, mock_mesh_mapper_config
    ):
        """Test that padded_shape raises ValueError when device is None but is_resolved mocked True."""
        lw = LazyWeight(
            source=mock_tensor,
            device=None,
            memory_config=mock_memory_config,
            mesh_mapper_config=mock_mesh_mapper_config,
            layout=ttnn.TILE_LAYOUT,
        )
        # Override is_resolved to return True to reach the ValueError branch
        with patch.object(lw, "is_resolved", return_value=True):
            with pytest.raises(ValueError, match="device must be set to compute padded_shape"):
                _ = lw.padded_shape


# ============================================================================
# Test get_device_weight Method
# ============================================================================


class TestGetDeviceWeight:
    """Tests for LazyWeight.get_device_weight() method."""

    def test_raises_when_device_none(self, mock_tensor, mock_memory_config):
        """Test that get_device_weight raises when device is None."""
        lw = LazyWeight(
            source=mock_tensor,
            device=None,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        with pytest.raises(ValueError, match="device must be provided"):
            lw.get_device_weight()

    def test_raises_when_layout_none(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that get_device_weight raises when layout is None."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=None,
        )
        with pytest.raises(ValueError, match="layout must be provided"):
            lw.get_device_weight()

    def test_raises_when_memory_config_none(self, mock_tensor, mock_mesh_device):
        """Test that get_device_weight raises when memory_config is None."""
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=None,
            layout=ttnn.TILE_LAYOUT,
        )
        with pytest.raises(ValueError, match="memory_config must be provided"):
            lw.get_device_weight()

    def test_returns_cached_value_on_second_call(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that get_device_weight returns cached _value on second call."""
        mock_ttnn_tensor = MagicMock()
        lw = LazyWeight(
            source=mock_tensor,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )
        # Manually set _value to simulate previous call
        lw._value = mock_ttnn_tensor

        result = lw.get_device_weight()
        assert result is mock_ttnn_tensor

    def test_get_device_weight_replicated_flow(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test get_device_weight with replicated tensor (no sharding) - mocked."""
        mock_ttnn_tensor = MagicMock()
        mock_mapper = MagicMock()

        with patch("models.common.modules.lazy_weight.ttnn") as mock_ttnn:
            mock_ttnn.replicate_tensor_to_mesh_mapper.return_value = mock_mapper
            mock_ttnn.from_torch.return_value = mock_ttnn_tensor

            lw = LazyWeight(
                source=mock_tensor,
                device=mock_mesh_device,
                memory_config=mock_memory_config,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper_config=None,  # Replicated
            )

            result = lw.get_device_weight()

            # Verify replicate_tensor_to_mesh_mapper was called
            mock_ttnn.replicate_tensor_to_mesh_mapper.assert_called_once_with(mock_mesh_device)
            assert result is mock_ttnn_tensor

    def test_get_device_weight_sharded_flow(
        self, mock_tensor, mock_mesh_device, mock_memory_config, mock_mesh_mapper_config
    ):
        """Test get_device_weight with sharded tensor - mocked."""
        mock_ttnn_tensor = MagicMock()
        mock_mapper = MagicMock()

        with patch("models.common.modules.lazy_weight.ttnn") as mock_ttnn:
            mock_ttnn.create_mesh_mapper.return_value = mock_mapper
            mock_ttnn.from_torch.return_value = mock_ttnn_tensor

            lw = LazyWeight(
                source=mock_tensor,
                device=mock_mesh_device,
                memory_config=mock_memory_config,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper_config=mock_mesh_mapper_config,
            )

            result = lw.get_device_weight()

            # Verify create_mesh_mapper was called with the config
            mock_ttnn.create_mesh_mapper.assert_called_once_with(mock_mesh_device, mock_mesh_mapper_config)
            assert result is mock_ttnn_tensor

    def test_get_device_weight_with_cache_dir_weight_name(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test get_device_weight extracts cache_dir and weight_name from tuple."""
        mock_ttnn_tensor = MagicMock()
        mock_mapper = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            with patch("models.common.modules.lazy_weight.ttnn") as mock_ttnn:
                mock_ttnn.replicate_tensor_to_mesh_mapper.return_value = mock_mapper
                mock_ttnn.from_torch.return_value = mock_ttnn_tensor
                mock_ttnn.StorageType.HOST = ttnn.StorageType.HOST
                mock_ttnn_tensor.storage_type.return_value = ttnn.StorageType.HOST

                lw = LazyWeight(
                    source=mock_tensor,
                    cache_dir_weight_name=(cache_dir, "my_weight"),
                    device=mock_mesh_device,
                    memory_config=mock_memory_config,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper_config=None,
                )

                result = lw.get_device_weight()
                assert result is not None

    def test_get_device_weight_cache_hit(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test get_device_weight loads from cache when file exists."""
        mock_ttnn_tensor = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create LazyWeight and compute cache path WITHOUT mocking
            lw = LazyWeight(
                source=mock_tensor,
                cache_dir_weight_name=(cache_dir, "cached_weight"),
                device=mock_mesh_device,
                memory_config=mock_memory_config,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper_config=None,
            )

            # Get expected cache path and create fake cache file
            cache_path = lw._get_cache_fill_path(cache_dir, "cached_weight")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.touch()

            # Now patch ttnn.load_tensor for the get_device_weight call
            with patch("models.common.modules.lazy_weight.ttnn.load_tensor") as mock_load:
                mock_load.return_value = mock_ttnn_tensor

                result = lw.get_device_weight()

                # Verify load_tensor was called (cache hit)
                mock_load.assert_called_once()
                assert result is mock_ttnn_tensor


# ============================================================================
# Test _auto_pad_for_sharding Function
# ============================================================================


class TestAutoPadForSharding:
    """Tests for _auto_pad_for_sharding() function."""

    def test_no_padding_when_shapes_match(self, mock_tensor):
        """Test that no padding is applied when shapes match."""
        result = _auto_pad_for_sharding(mock_tensor, tuple(mock_tensor.shape))
        assert result is mock_tensor

    def test_padding_applied_when_shapes_differ(self):
        """Test that padding is applied when shapes differ."""
        tensor = torch.randn(1, 32, 100)
        padded_shape = (1, 32, 128)

        result = _auto_pad_for_sharding(tensor, padded_shape)
        assert result.shape == padded_shape
        # Original data should be preserved
        assert torch.equal(result[:, :, :100], tensor)

    def test_padding_with_custom_pad_value(self):
        """Test padding with custom pad value."""
        tensor = torch.zeros(1, 32, 100)
        padded_shape = (1, 32, 128)

        result = _auto_pad_for_sharding(tensor, padded_shape, pad_value=1.0)
        assert result.shape == padded_shape
        # Padded region should have pad_value
        assert torch.allclose(result[:, :, 100:], torch.ones(1, 32, 28))


# ============================================================================
# Test _from_torch_and_dump Function
# ============================================================================


class TestFromTorchAndDump:
    """Tests for _from_torch_and_dump() function."""

    def test_no_cache_when_cache_file_none(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that no caching occurs when cache_file_name is None."""
        mock_ttnn_tensor = MagicMock()
        mock_mapper = MagicMock()

        with patch("models.common.modules.lazy_weight.ttnn") as mock_ttnn:
            mock_ttnn.from_torch.return_value = mock_ttnn_tensor

            result = _from_torch_and_dump(
                tensor=mock_tensor,
                device=mock_mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mock_memory_config,
                mesh_mapper=mock_mapper,
                is_replicated=False,
                cache_file_name=None,
            )

            mock_ttnn.from_torch.assert_called_once()
            assert result is mock_ttnn_tensor

    def test_cache_dump_when_cache_file_provided(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that tensor is cached when cache_file_name is provided."""
        mock_ttnn_tensor = MagicMock()
        mock_ttnn_tensor.storage_type.return_value = ttnn.StorageType.HOST

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.tensorbin"

            with patch("models.common.modules.lazy_weight.ttnn") as mock_ttnn:
                mock_ttnn.from_torch.return_value = mock_ttnn_tensor
                mock_ttnn.StorageType.HOST = ttnn.StorageType.HOST

                _from_torch_and_dump(
                    tensor=mock_tensor,
                    device=mock_mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mock_memory_config,
                    mesh_mapper=None,
                    is_replicated=True,
                    cache_file_name=cache_file,
                )

                # Verify from_torch was called with device=None for caching
                call_kwargs = mock_ttnn.from_torch.call_args.kwargs
                assert call_kwargs.get("device") is None

                # Verify dump_tensor_flatbuffer was called
                mock_ttnn._ttnn.tensor.dump_tensor_flatbuffer.assert_called_once()

    def test_replicated_uses_none_mesh_mapper(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that replicated tensors use None mesh_mapper in from_torch."""
        mock_ttnn_tensor = MagicMock()
        mock_mapper = MagicMock()

        with patch("models.common.modules.lazy_weight.ttnn") as mock_ttnn:
            mock_ttnn.from_torch.return_value = mock_ttnn_tensor

            _from_torch_and_dump(
                tensor=mock_tensor,
                device=mock_mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mock_memory_config,
                mesh_mapper=mock_mapper,
                is_replicated=True,
                cache_file_name=None,
            )

            call_kwargs = mock_ttnn.from_torch.call_args.kwargs
            assert call_kwargs.get("mesh_mapper") is None

    def test_sharded_uses_mesh_mapper(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that sharded tensors use the provided mesh_mapper."""
        mock_ttnn_tensor = MagicMock()
        mock_mapper = MagicMock()

        with patch("models.common.modules.lazy_weight.ttnn") as mock_ttnn:
            mock_ttnn.from_torch.return_value = mock_ttnn_tensor

            _from_torch_and_dump(
                tensor=mock_tensor,
                device=mock_mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mock_memory_config,
                mesh_mapper=mock_mapper,
                is_replicated=False,
                cache_file_name=None,
            )

            call_kwargs = mock_ttnn.from_torch.call_args.kwargs
            assert call_kwargs.get("mesh_mapper") is mock_mapper


# ============================================================================
# Test resolve_lazy_weight Function
# ============================================================================


class TestResolveLazyWeight:
    """Tests for resolve_lazy_weight() function."""

    def test_resolves_none_fields(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that resolve_lazy_weight fills in None fields."""
        lw = LazyWeight(source=mock_tensor)

        resolved = resolve_lazy_weight(
            lw,
            device=mock_mesh_device,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        assert resolved.device is mock_mesh_device
        assert resolved.memory_config is mock_memory_config
        assert resolved.layout == ttnn.TILE_LAYOUT

    def test_does_not_override_existing_fields(self, mock_tensor, mock_mesh_device, mock_memory_config):
        """Test that resolve_lazy_weight does not override non-None fields."""
        original_device = MagicMock(spec=ttnn.MeshDevice)
        lw = LazyWeight(
            source=mock_tensor,
            device=original_device,
            dtype=ttnn.bfloat4_b,
        )

        resolved = resolve_lazy_weight(
            lw,
            device=mock_mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=mock_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        # Original non-None fields should be preserved
        assert resolved.device is original_device
        assert resolved.dtype == ttnn.bfloat4_b
        # None fields should be filled
        assert resolved.memory_config is mock_memory_config
        assert resolved.layout == ttnn.TILE_LAYOUT

    def test_returns_new_instance(self, mock_tensor, mock_mesh_device):
        """Test that resolve_lazy_weight returns a new LazyWeight instance."""
        lw = LazyWeight(source=mock_tensor)
        resolved = resolve_lazy_weight(lw, device=mock_mesh_device)

        assert resolved is not lw
        assert resolved.device is mock_mesh_device
        assert lw.device is None  # Original unchanged

    def test_with_no_overrides(self, mock_tensor):
        """Test resolve_lazy_weight with no kwargs returns equivalent instance."""
        lw = LazyWeight(source=mock_tensor, dtype=ttnn.bfloat16)
        resolved = resolve_lazy_weight(lw)

        assert resolved is not lw
        assert resolved.source is lw.source
        assert resolved.dtype == lw.dtype


# ============================================================================
# Integration Tests (require device - use parametrized fixture)
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),
    ],
    ids=[
        "1x1",
    ],
    indirect=True,
)
class TestLazyWeightIntegration:
    """Integration tests for LazyWeight that require actual TTNN device."""

    def test_get_device_weight_replicated(self, ttnn_mesh_device: ttnn.MeshDevice):
        """Test get_device_weight with replicated tensor (no mesh_mapper_config)."""
        tensor = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

        lw = LazyWeight(
            source=tensor,
            device=ttnn_mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper_config=None,  # Replicated
        )

        result = lw.get_device_weight()
        assert result is not None
        # Subsequent calls should return cached value
        result2 = lw.get_device_weight()
        assert result2 is result

    def test_get_device_weight_with_cache(self, ttnn_mesh_device: ttnn.MeshDevice):
        """Test get_device_weight with caching enabled."""
        tensor = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            lw = LazyWeight(
                source=tensor,
                cache_dir_weight_name=(cache_dir, "test_weight"),
                device=ttnn_mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper_config=None,
            )

            # First call should create cache
            result1 = lw.get_device_weight()
            assert result1 is not None

            # Verify cache file was created
            cache_files = list(cache_dir.glob("*.tensorbin"))
            assert len(cache_files) == 1
            assert cache_files[0].name.startswith("test_weight_")

    def test_get_device_weight_cache_hit(self, ttnn_mesh_device: ttnn.MeshDevice):
        """Test get_device_weight loads from cache on second instance."""
        tensor = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # First LazyWeight creates cache
            lw1 = LazyWeight(
                source=tensor,
                cache_dir_weight_name=(cache_dir, "test_weight"),
                device=ttnn_mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper_config=None,
            )
            lw1.get_device_weight()

            # Second LazyWeight should load from cache
            lw2 = LazyWeight(
                source=tensor,
                cache_dir_weight_name=(cache_dir, "test_weight"),
                device=ttnn_mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper_config=None,
            )
            result2 = lw2.get_device_weight()

            assert result2 is not None


if __name__ == "__main__":
    # Run unit tests that don't require device
    pytest.main([__file__, "-v", "-k", "not Integration"])
