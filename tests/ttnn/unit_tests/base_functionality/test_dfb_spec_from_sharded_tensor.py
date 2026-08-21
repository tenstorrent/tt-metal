# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.dfb_spec_from_sharded_tensor: derive a Metal 2.0 DataflowBufferSpec from a sharded tensor.

The Metal 2.0 counterpart of ttnn.cb_descriptor_from_sharded_tensor. Every entry-format field of
a DataflowBufferSpec (entry_size, num_entries, data_format_metadata, tile_format_metadata) used to
be a hand-typed integer next to a TensorParameter that already knew it; the ProgramSpec validator
explicitly declines to check them, so a one-word mistake silently corrupts tensor data.

These tests assert the derivation, not any on-device behaviour: the derived numbers are checked
against independently computed expectations and cross-checked against the values
ttnn.cb_descriptor_from_sharded_tensor derives for the same tensor.
"""

import pytest
import torch

import ttnn

# Nothing here varies the device config, and every test only allocates and inspects.
pytestmark = pytest.mark.use_module_device


def _one_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _two_cores():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])


def _height_sharded(device, shape, shard_shape, cores, *, layout, dtype, buffer_type=None):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type or ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


def _data_format_of(dtype):
    """The tt::DataFormat integer the DFB constructor derives from a ttnn.DataType.

    data_format_metadata is surfaced as a raw enum integer, so build the reference through the
    constructor rather than hardcoding the value.
    """
    return ttnn.DataflowBufferSpec(unique_id="ref", entry_size=1, num_entries=1, data_format=dtype).data_format_metadata


# ---------------------------------------------------------------- tiled tensors


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tiled_shard_derives_one_entry_per_tile(device, dtype):
    """A TILE tensor is already tile-paged: one entry per tile of the resident shard."""
    # 4 tile rows x 2 tile cols, height-split over 2 cores -> 2x2 = 4 tiles per core.
    tensor = _height_sharded(device, (1, 1, 128, 64), (64, 64), _two_cores(), layout=ttnn.TILE_LAYOUT, dtype=dtype)

    dfb = ttnn.dfb_spec_from_sharded_tensor("in", tensor)

    assert dfb.unique_id == "in"
    assert dfb.entry_size == ttnn.tile_size(dtype)
    assert dfb.num_entries == 4
    assert dfb.data_format_metadata == _data_format_of(dtype)
    assert dfb.tile_format_metadata is not None
    assert dfb.tile_format_metadata.tile_shape == [32, 32]
    assert dfb.unpack_face_geometry_metadata is None
    assert dfb.borrowed_from is None


def test_tiled_shard_agrees_with_cb_descriptor_helper(device):
    """The DFB's entry stride and total bytes must match what the ProgramDescriptor path derives."""
    tensor = _height_sharded(
        device, (1, 1, 128, 64), (64, 64), _two_cores(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    dfb = ttnn.dfb_spec_from_sharded_tensor("in", tensor)
    cb = ttnn.cb_descriptor_from_sharded_tensor(0, tensor)

    assert dfb.entry_size == cb.format_descriptors[0].page_size
    assert dfb.entry_size * dfb.num_entries == cb.total_size
    assert dfb.data_format_metadata == cb.format_descriptors[0].data_format_as_uint8


def test_page_as_tile_is_a_noop_for_a_tiled_tensor(device):
    """A TILE tensor is tile-paged either way; page_as_tile may be passed uniformly."""
    tensor = _height_sharded(
        device, (1, 1, 128, 64), (64, 64), _two_cores(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    plain = ttnn.dfb_spec_from_sharded_tensor("in", tensor)
    as_tile = ttnn.dfb_spec_from_sharded_tensor("in", tensor, page_as_tile=True)

    assert (as_tile.entry_size, as_tile.num_entries) == (plain.entry_size, plain.num_entries)
    assert as_tile.tile_format_metadata == plain.tile_format_metadata


# ---------------------------------------------------------------- row-major tensors


def test_row_major_shard_derives_one_entry_per_stick(device):
    """For a ROW_MAJOR tensor the natural entry is a stick, and no tile format is claimed."""
    tensor = _height_sharded(
        device, (1, 1, 32, 64), (32, 64), _one_core(), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )

    dfb = ttnn.dfb_spec_from_sharded_tensor("in", tensor)

    assert dfb.entry_size == 64 * 2  # one 64-wide bfloat16 stick
    assert dfb.num_entries == 32  # 32 sticks in the shard
    assert dfb.tile_format_metadata is None, "a stick is not a tile"
    assert dfb.data_format_metadata == _data_format_of(ttnn.bfloat16)


def test_page_as_tile_repages_a_row_major_shard(device):
    """Same shard, two views: 32 sticks of 64 B, or the single tile those sticks add up to."""
    tensor = _height_sharded(
        device, (1, 1, 32, 32), (32, 32), _one_core(), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )

    sticks = ttnn.dfb_spec_from_sharded_tensor("in", tensor)
    tiles = ttnn.dfb_spec_from_sharded_tensor("in", tensor, page_as_tile=True)

    assert (sticks.entry_size, sticks.num_entries) == (32 * 2, 32)
    assert (tiles.entry_size, tiles.num_entries) == (ttnn.tile_size(ttnn.bfloat16), 1)
    assert tiles.tile_format_metadata is not None
    # Both views describe exactly the same bytes.
    assert tiles.entry_size * tiles.num_entries == sticks.entry_size * sticks.num_entries


def test_page_as_tile_on_a_sub_tile_shard_falls_back_to_one_partial_entry(device):
    """A 16-row shard is half a tile: one partial-tile entry, mirroring set_cb_page_size_for_tile.

    Such an entry needs a matching unpack_face_geometry to be unpacked correctly, which the helper
    passes through rather than guessing.
    """
    tensor = _height_sharded(
        device, (1, 1, 16, 32), (16, 32), _one_core(), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )

    dfb = ttnn.dfb_spec_from_sharded_tensor(
        "in", tensor, page_as_tile=True, unpack_face_geometry=ttnn.FaceGeometry(16, 2)
    )

    assert dfb.entry_size == 16 * 32 * 2  # the whole shard, less than one tile
    assert dfb.num_entries == 1
    assert dfb.tile_format_metadata is not None
    assert dfb.unpack_face_geometry_metadata.face_r_dim == 16
    assert dfb.unpack_face_geometry_metadata.num_faces == 2


# ---------------------------------------------------------------- borrowing and depth


def test_borrowed_from_is_recorded(device):
    tensor = _height_sharded(
        device, (1, 1, 64, 64), (64, 64), _one_core(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    dfb = ttnn.dfb_spec_from_sharded_tensor("in", tensor, borrowed_from="a")

    assert dfb.borrowed_from == "a"
    assert dfb.entry_size * dfb.num_entries == 4 * ttnn.tile_size(ttnn.bfloat16)


def test_num_entries_override_keeps_the_derived_entry_size(device):
    """A separately allocated staging DFB: the tensor's format, a caller-chosen depth."""
    tensor = _height_sharded(
        device, (1, 1, 64, 64), (64, 64), _one_core(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    derived = ttnn.dfb_spec_from_sharded_tensor("stage", tensor)
    staged = ttnn.dfb_spec_from_sharded_tensor("stage", tensor, num_entries=2)

    assert derived.num_entries == 4
    assert staged.num_entries == 2
    assert staged.entry_size == derived.entry_size
    assert staged.borrowed_from is None


# ---------------------------------------------------------------- rejections


def test_interleaved_tensor_is_rejected(device, expect_error):
    tensor = ttnn.from_torch(
        torch.randn(1, 1, 64, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    with expect_error(RuntimeError, "tensor must be sharded"):
        ttnn.dfb_spec_from_sharded_tensor("in", tensor)


def test_borrowed_dfb_larger_than_the_shard_is_rejected(device, expect_error):
    """The ProgramSpec validator and the attach path both catch this later; catch it at build time."""
    tensor = _height_sharded(
        device, (1, 1, 64, 64), (64, 64), _one_core(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    with expect_error(RuntimeError, "does not fit in the shard it borrows"):
        ttnn.dfb_spec_from_sharded_tensor("in", tensor, num_entries=5, borrowed_from="a")

    # Not borrowing: the depth is the caller's business, no sizing claim is made against the tensor.
    deep = ttnn.dfb_spec_from_sharded_tensor("stage", tensor, num_entries=5)
    assert deep.num_entries == 5


def test_host_tensor_is_rejected(device, expect_error):
    """A host tensor has no shard, so it trips the same check an interleaved one does.

    (Tensor.is_sharded() is false for host storage, so the "must be allocated" guard behind it
    only ever fires for a deallocated device tensor.)
    """
    host = ttnn.from_torch(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat16)

    with expect_error(RuntimeError, "must be sharded"):
        ttnn.dfb_spec_from_sharded_tensor("in", host)
