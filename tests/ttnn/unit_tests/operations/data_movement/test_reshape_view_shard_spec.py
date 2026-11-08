# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A view-compatible reshape must honour an explicit sharded memory config.

A reshape whose result is metadata-only keeps the input's buffer, so it cannot change the shard spec:
the buffer carries the input's own. The view gate used to compare only sharded-ness and L1/DRAM
placement, so a reshape to a different shape that happened to be a view silently dropped the caller's
requested shard spec and returned the input's instead.
"""

import pytest

import torch

import ttnn

pytestmark = pytest.mark.use_module_device

DTYPE = ttnn.bfloat16

IN_SHAPE = (1, 1, 64, 64)
OUT_SHAPE = (1, 2, 32, 64)

# The input is height-sharded over two cores stacked vertically, and the request is a different
# width-sharded spec over two cores side by side, so no COALESCE can make them the same config.
IN_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))])
OUT_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])

IN_MEM_CONFIG = ttnn.MemoryConfig(
    ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.types.BufferType.L1,
    ttnn.ShardSpec(IN_GRID, (32, 64), ttnn.ShardOrientation.ROW_MAJOR),
)
OUT_MEM_CONFIG = ttnn.MemoryConfig(
    ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.types.BufferType.L1,
    ttnn.ShardSpec(OUT_GRID, (64, 32), ttnn.ShardOrientation.ROW_MAJOR),
)


def _shard_spec_summary(tensor):
    # memory_config().shard_spec is a read-only property here: it is the ShardSpec, or None.
    spec = tensor.memory_config().shard_spec
    assert spec is not None, "expected a sharded output"
    grid = spec.grid
    bbox = grid.bounding_box()
    return (
        tensor.memory_config().memory_layout,
        grid.num_cores(),
        tuple(spec.shape),
        spec.orientation,
        (bbox.start.x, bbox.start.y),
        (bbox.end.x, bbox.end.y),
    )


def test_reshape_to_view_shape_honours_requested_shard_spec(device):
    """The returned tensor must carry the requested shard spec, not the input's."""
    torch.manual_seed(0)
    source = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    x = ttnn.from_torch(source, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=IN_MEM_CONFIG)
    assert _shard_spec_summary(x) == (
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
        2,
        (32, 64),
        ttnn.ShardOrientation.ROW_MAJOR,
        (0, 0),
        (0, 1),
    )

    out = ttnn.reshape(x, OUT_SHAPE, memory_config=OUT_MEM_CONFIG)

    assert _shard_spec_summary(out) == (
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
        2,
        (64, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
        (0, 0),
        (1, 0),
    )
    assert list(out.shape) == list(OUT_SHAPE)
    assert torch.equal(ttnn.to_torch(out).float(), source.reshape(OUT_SHAPE).float())


def test_reshape_to_view_shape_keeps_input_spec_when_none_requested(device):
    """Control: no memory_config keeps the view, so the input's spec still comes back."""
    torch.manual_seed(0)
    source = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    x = ttnn.from_torch(source, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=IN_MEM_CONFIG)

    out = ttnn.reshape(x, OUT_SHAPE)

    assert _shard_spec_summary(out) == (
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
        2,
        (32, 64),
        ttnn.ShardOrientation.ROW_MAJOR,
        (0, 0),
        (0, 1),
    )
    assert torch.equal(ttnn.to_torch(out).float(), source.reshape(OUT_SHAPE).float())


def test_reshape_to_view_shape_keeps_view_when_config_echoes_input(device):
    """Control: re-requesting the tensor's own config explicitly stays on the cheap view path."""
    torch.manual_seed(0)
    source = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    x = ttnn.from_torch(source, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=IN_MEM_CONFIG)

    out = ttnn.reshape(x, OUT_SHAPE, memory_config=IN_MEM_CONFIG)

    assert _shard_spec_summary(out) == (
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
        2,
        (32, 64),
        ttnn.ShardOrientation.ROW_MAJOR,
        (0, 0),
        (0, 1),
    )
    assert torch.equal(ttnn.to_torch(out).float(), source.reshape(OUT_SHAPE).float())


def test_reshape_to_view_shape_interleaved_request(device):
    """An explicitly interleaved request also has to move the buffer, not keep the sharded one."""
    torch.manual_seed(0)
    source = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    x = ttnn.from_torch(source, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=IN_MEM_CONFIG)

    out = ttnn.reshape(x, OUT_SHAPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert out.memory_config().memory_layout == ttnn.types.TensorMemoryLayout.INTERLEAVED
    assert out.memory_config().shard_spec is None
    assert list(out.shape) == list(OUT_SHAPE)
    assert torch.equal(ttnn.to_torch(out).float(), source.reshape(OUT_SHAPE).float())
