# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LazyBuffer — lazy device buffer allocation for mutable state tensors."""

import pytest
import torch

import ttnn
from models.common.modules.lazy_buffer import LazyBuffer, resolve_lazy_buffer

# ==============================================================================
# Unit tests (no device)
# ==============================================================================


class TestLazyBufferUnit:
    def test_defaults(self):
        """Default dtype=int32, layout=TILE. device/mesh_mapper/memory_config=None."""
        source = torch.zeros(4, 8, dtype=torch.int32)
        buf = LazyBuffer(source=source)
        assert buf.dtype == ttnn.int32
        assert buf.layout == ttnn.TILE_LAYOUT
        assert buf.device is None
        assert buf.mesh_mapper is None
        assert buf.memory_config is None
        assert torch.equal(buf.source, source)

    def test_custom_fields(self):
        source = torch.ones(32, 1, dtype=torch.float32)
        buf = LazyBuffer(
            source=source,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert buf.dtype == ttnn.bfloat16
        assert buf.layout == ttnn.ROW_MAJOR_LAYOUT
        assert buf.memory_config == ttnn.DRAM_MEMORY_CONFIG

    def test_is_resolved_false_without_device(self):
        buf = LazyBuffer(source=torch.zeros(4, 8))
        assert not buf.is_resolved()

    def test_is_resolved_false_without_dtype(self):
        buf = LazyBuffer(source=torch.zeros(4, 8), dtype=None, device="fake")
        assert not buf.is_resolved()

    def test_is_resolved_false_without_layout(self):
        buf = LazyBuffer(source=torch.zeros(4, 8), layout=None, device="fake")
        assert not buf.is_resolved()

    def test_is_resolved_true_when_all_set(self):
        buf = LazyBuffer(source=torch.zeros(4, 8), device="fake", dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
        assert buf.is_resolved()

    def test_raises_without_device(self):
        buf = LazyBuffer(source=torch.zeros(4, 8))
        with pytest.raises(ValueError, match="device must be set"):
            buf.get_device_buffer()

    def test_raises_without_layout(self):
        buf = LazyBuffer(source=torch.zeros(4, 8), layout=None, device="fake")
        with pytest.raises(ValueError, match="layout must be set"):
            buf.get_device_buffer()

    def test_update_before_materialize_replaces_source(self):
        """update() before get_device_buffer() just swaps the source tensor."""
        source = torch.zeros(4, 8)
        buf = LazyBuffer(source=source)
        new_source = torch.ones(4, 8)
        buf.update(new_source)
        assert torch.equal(buf.source, new_source)
        assert buf._value is None  # not yet materialized


class TestResolveLazyBuffer:
    def test_fills_none_fields(self):
        buf = LazyBuffer(source=torch.zeros(4, 8), dtype=ttnn.bfloat16, layout=None, memory_config=None)
        resolved = resolve_lazy_buffer(buf, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert resolved.layout == ttnn.ROW_MAJOR_LAYOUT
        assert resolved.memory_config == ttnn.DRAM_MEMORY_CONFIG

    def test_does_not_overwrite_non_none(self):
        buf = LazyBuffer(source=torch.zeros(4, 8), dtype=ttnn.bfloat16)
        resolved = resolve_lazy_buffer(buf, dtype=ttnn.int32)
        assert resolved.dtype == ttnn.bfloat16  # preserved, not overwritten

    def test_preserves_source(self):
        source = torch.randn(4, 8)
        buf = LazyBuffer(source=source)
        resolved = resolve_lazy_buffer(buf, device="fake")
        assert torch.equal(resolved.source, source)

    def test_returns_new_instance(self):
        buf = LazyBuffer(source=torch.zeros(4, 8))
        resolved = resolve_lazy_buffer(buf, device="fake")
        assert buf is not resolved
        assert buf.device is None  # original unchanged
        assert resolved.device == "fake"


# ==============================================================================
# Device tests
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
class TestLazyBufferDevice:
    def test_get_device_buffer_returns_ttnn_tensor(self, ttnn_mesh_device):
        buf = LazyBuffer(
            source=torch.zeros(32, 64, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        result = buf.get_device_buffer()
        assert isinstance(result, ttnn.Tensor)

    def test_get_device_buffer_idempotent(self, ttnn_mesh_device):
        """Second call returns the exact same Python object (cached handle)."""
        buf = LazyBuffer(
            source=torch.zeros(32, 64, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        first = buf.get_device_buffer()
        second = buf.get_device_buffer()
        assert first is second

    def test_update_after_materialize_preserves_handle(self, ttnn_mesh_device):
        """update() after materialization keeps the same tensor handle (no reallocation)."""
        buf = LazyBuffer(
            source=torch.zeros(32, 1, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_mesh_device,
        )
        handle_before = buf.get_device_buffer()
        buf.update(torch.ones(32, 1, dtype=torch.float32))
        handle_after = buf.get_device_buffer()
        assert handle_before is handle_after

    def test_update_after_materialize_changes_device_data(self, ttnn_mesh_device):
        """update() actually writes new data to device — readback should match new source."""
        buf = LazyBuffer(
            source=torch.zeros(32, 1, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_mesh_device,
        )
        buf.get_device_buffer()  # materialize with zeros

        # Update with ones
        new_source = torch.ones(32, 1, dtype=torch.float32)
        buf.update(new_source)

        # Readback and verify
        readback = ttnn.to_torch(buf.get_device_buffer()).float()
        assert torch.allclose(
            readback[:32, :1], new_source, atol=0.01
        ), f"Readback mismatch: expected ones, got {readback[:4, :1].flatten()}"

    def test_update_before_materialize_uses_new_source(self, ttnn_mesh_device):
        """update() before get_device_buffer() means materialization uses the new source."""
        buf = LazyBuffer(
            source=torch.zeros(32, 1, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_mesh_device,
        )

        # Update BEFORE materializing
        new_source = torch.full((32, 1), 42.0, dtype=torch.float32)
        buf.update(new_source)

        # Now materialize — should use new_source (42.0), not original (0.0)
        readback = ttnn.to_torch(buf.get_device_buffer()).float()
        assert readback[0, 0].item() == pytest.approx(42.0, abs=0.5), f"Expected ~42.0, got {readback[0, 0].item()}"

    def test_with_dram_memory_config(self, ttnn_mesh_device):
        buf = LazyBuffer(
            source=torch.randn(32, 64, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        t = buf.get_device_buffer()
        assert isinstance(t, ttnn.Tensor)

    def test_row_major_layout(self, ttnn_mesh_device):
        buf = LazyBuffer(
            source=torch.zeros(32, 1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=ttnn_mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        t = buf.get_device_buffer()
        assert isinstance(t, ttnn.Tensor)
