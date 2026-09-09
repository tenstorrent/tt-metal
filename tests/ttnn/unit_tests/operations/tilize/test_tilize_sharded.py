# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — sharded and L1 placement.

Pins the two things the golden suite grades and the one thing it cannot see:

  1. VALUES — every placement (HEIGHT / WIDTH / BLOCK / ND, ROW_MAJOR and
     COL_MAJOR, L1 and DRAM, plus the interleaved crossovers) is bit-identical,
     because tilize does no arithmetic.
  2. NATIVENESS — a core's own L1 shard is consumed through a CB PLACED ON the
     shard buffer, never re-read through a TensorAccessor. That distinction is
     invisible in the values (an accessor read of a local shard returns exactly
     the right bytes), so it is asserted against the ProgramDescriptor: a native
     side's CB reports `has_buffer` and the native-output path carries no writer
     kernel at all.

Device comes from the directory conftest's module-scoped `device` fixture.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.tilize.tilize_program_descriptor import (
    CB_INPUT_ROWS,
    CB_OUTPUT_TILES,
    create_program_descriptor,
    derive_plan,
)

_L1 = ttnn.BufferType.L1
_DRAM = ttnn.BufferType.DRAM
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def _crs(start, end):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end))})


def _shard(buffer, grid, shard_shape, orientation, scheme):
    """scheme=None -> the nd (NdShardSpec) API."""
    if scheme is None:
        return ttnn.MemoryConfig(buffer, ttnn.NdShardSpec(ttnn.Shape(list(shard_shape)), grid, orientation))
    return ttnn.MemoryConfig(scheme, buffer, ttnn.ShardSpec(grid, tuple(shard_shape), orientation))


def _interleaved(buffer):
    return ttnn.DRAM_MEMORY_CONFIG if buffer == _DRAM else ttnn.L1_MEMORY_CONFIG


def _skip_if_grid_too_big(device, mem_config):
    if not mem_config.is_sharded():
        return
    spec = mem_config.shard_spec or mem_config.nd_shard_spec
    limit = device.dram_grid_size() if mem_config.buffer_type == _DRAM else device.compute_with_storage_grid_size()
    for core_range in spec.grid.ranges():
        if core_range.end.x > limit.x - 1 or core_range.end.y > limit.y - 1:
            pytest.skip("shard grid exceeds this device's grid")


# name -> (shape, in_memory_config_factory, out_memory_config_factory)
# Mirrors eval/golden_tests/tilize/feature_spec.py's sharded_legacy_2d /
# sharded_nd groups plus short_wide_width_sharded, so a regression here is the
# same regression the golden suite would report.
CASES = {
    "height_sharded_same_spec": (
        [1, 1, 512, 64],
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (128, 64), _ROW, _HEIGHT),
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (128, 64), _ROW, _HEIGHT),
    ),
    "width_sharded_same_spec": (
        [1, 1, 64, 512],
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (64, 128), _ROW, _WIDTH),
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (64, 128), _ROW, _WIDTH),
    ),
    "block_sharded_col_major": (
        [1, 1, 128, 128],
        lambda: _shard(_L1, _crs((0, 0), (1, 1)), (64, 64), _COL, _BLOCK),
        lambda: _shard(_L1, _crs((0, 0), (1, 1)), (64, 64), _COL, _BLOCK),
    ),
    "block_sharded_row_major": (
        [1, 1, 128, 128],
        lambda: _shard(_L1, _crs((0, 0), (1, 1)), (64, 64), _ROW, _BLOCK),
        lambda: _shard(_L1, _crs((0, 0), (1, 1)), (64, 64), _ROW, _BLOCK),
    ),
    "dram_sharded_out": (
        [1, 1, 128, 64],
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), (64, 64), _ROW, _HEIGHT),
        lambda: _shard(_DRAM, _crs((0, 0), (1, 0)), (64, 64), _ROW, _HEIGHT),
    ),
    "interleaved_to_height_sharded": (
        [1, 1, 128, 64],
        lambda: _interleaved(_DRAM),
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (32, 64), _ROW, _HEIGHT),
    ),
    "height_sharded_to_interleaved": (
        [1, 1, 128, 64],
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (32, 64), _ROW, _HEIGHT),
        lambda: _interleaved(_DRAM),
    ),
    "cross_spec_height_in_width_out": (
        [1, 1, 128, 128],
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), (64, 128), _ROW, _HEIGHT),
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), (128, 64), _ROW, _WIDTH),
    ),
    "nd_same_spec": (
        [1, 1, 128, 64],
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), [1, 1, 64, 64], _ROW, None),
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), [1, 1, 64, 64], _ROW, None),
    ),
    "nd_in_legacy_out": (
        [1, 1, 128, 64],
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), [1, 1, 64, 64], _ROW, None),
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), (64, 64), _ROW, _HEIGHT),
    ),
    "interleaved_to_nd": (
        [1, 1, 128, 64],
        lambda: _interleaved(_DRAM),
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), [1, 1, 64, 64], _ROW, None),
    ),
    "short_wide_width_sharded": (
        [1, 1, 32, 4096],
        lambda: _interleaved(_DRAM),
        lambda: _shard(_L1, _crs((0, 0), (7, 0)), (32, 512), _ROW, _WIDTH),
    ),
    # Rank 3 with the leading dim folding into the shard's height: the shard's
    # 2-D view has to fold exactly the way R does.
    "rank3_height_sharded": (
        [4, 64, 64],
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (64, 64), _ROW, _HEIGHT),
        lambda: _shard(_L1, _crs((0, 0), (3, 0)), (64, 64), _ROW, _HEIGHT),
    ),
    # More shards than cores: the ND round-robin gives each core two blocks at a
    # stride, which is the only place `block_stride` is not 1 on the shard plan.
    "nd_two_shards_per_core": (
        [1, 1, 256, 64],
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), [1, 1, 64, 64], _ROW, None),
        lambda: _shard(_L1, _crs((0, 0), (1, 0)), [1, 1, 64, 64], _ROW, None),
    ),
}


def _run(device, name):
    shape, in_cfg_fn, out_cfg_fn = CASES[name]
    in_cfg, out_cfg = in_cfg_fn(), out_cfg_fn()
    _skip_if_grid_too_big(device, in_cfg)
    _skip_if_grid_too_big(device, out_cfg)

    torch_in = torch.randn(shape).bfloat16()
    tt_in = ttnn.from_torch(
        torch_in, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    tt_out = ttnn.tilize(tt_in, out_cfg, dtype=ttnn.bfloat16)
    assert tt_out.layout == ttnn.TILE_LAYOUT
    got = ttnn.to_torch(tt_out)
    assert list(got.shape) == shape, f"{name}: shape {list(got.shape)} != {shape}"
    assert torch.equal(
        got, torch_in
    ), f"{name}: not bit-identical, {int((got != torch_in).sum())} of {torch_in.numel()} elements differ"
    return tt_in, tt_out


@pytest.mark.parametrize("name", list(CASES))
def test_sharded_identity(device, name):
    """Values: every placement re-lays the bytes exactly."""
    _run(device, name)


# (case, expects a zero-copy input CB, expects a zero-copy output CB)
NATIVE_EXPECTATIONS = [
    ("height_sharded_same_spec", True, True),
    ("width_sharded_same_spec", True, True),
    ("block_sharded_col_major", True, True),
    ("nd_same_spec", True, True),
    ("nd_in_legacy_out", True, True),
    ("nd_two_shards_per_core", True, True),
    # DRAM destination: the shard is not in a worker's L1, so the write is real.
    ("dram_sharded_out", True, False),
    ("height_sharded_to_interleaved", True, False),
    # A different cut on each side — the output write is a genuine remote gather.
    ("cross_spec_height_in_width_out", True, False),
    ("interleaved_to_height_sharded", False, True),
    ("interleaved_to_nd", False, True),
    ("short_wide_width_sharded", False, True),
]


@pytest.mark.parametrize("name,input_native,output_native", NATIVE_EXPECTATIONS)
def test_sharded_side_is_zero_copy(device, name, input_native, output_native):
    """Nativeness, read off the ProgramDescriptor rather than off the values.

    A local shard read back through a TensorAccessor produces exactly the right
    answer, so `test_sharded_identity` passing says nothing about whether the
    axis was implemented. What does say it: the CB carrying the sharded side is
    PLACED ON that tensor's buffer, and the native-output path emits no writer.
    """
    tt_in, tt_out = _run(device, name)
    descriptor = create_program_descriptor(tt_in, tt_out)
    by_index = {cb.format_descriptors[0].buffer_index: cb for cb in descriptor.cbs}

    assert (
        by_index[CB_INPUT_ROWS].has_buffer() == input_native
    ), f"{name}: input CB zero-copy={by_index[CB_INPUT_ROWS].has_buffer()}, expected {input_native}"
    assert (
        by_index[CB_OUTPUT_TILES].has_buffer() == output_native
    ), f"{name}: output CB zero-copy={by_index[CB_OUTPUT_TILES].has_buffer()}, expected {output_native}"
    if input_native:
        assert by_index[CB_INPUT_ROWS].buffer_address() == tt_in.buffer_address()
    if output_native:
        assert by_index[CB_OUTPUT_TILES].buffer_address() == tt_out.buffer_address()
        # No writer kernel: reader + compute only.
        assert len(descriptor.kernels) == 2, f"{name}: {len(descriptor.kernels)} kernels, expected 2 (no writer)"
    else:
        assert len(descriptor.kernels) == 3


def test_shard_fixes_the_block_grid(device):
    """The knob-turn itself: on a sharded call the block extents are READ off
    the shard spec, not solved — one block per shard, on the shard's own core."""
    tt_in, tt_out = _run(device, "block_sharded_col_major")
    grid = tt_in.device().compute_with_storage_grid_size()
    plan = derive_plan(tt_in, tt_out, low_l1=False, grid=grid)
    assert plan.block_width_tiles == 2  # 64-element shard width
    assert plan.num_w_chunks == 2 and plan.num_row_groups == 2
    assert plan.num_blocks_total == 4
    assert plan.all_cores.num_cores() == 4
    assert sorted(num for _c, _s, num, _st in plan.assignment) == [1, 1, 1, 1]
