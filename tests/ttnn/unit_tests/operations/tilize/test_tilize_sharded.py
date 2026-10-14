# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2: sharded placements — regime selection, resident (zero-copy) shards, crossovers,
partial final shards and accessor fallbacks."""

import importlib

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd

op_module = importlib.import_module("ttnn.operations.tilize.tilize")

L1 = ttnn.BufferType.L1
DRAM = ttnn.BufferType.DRAM
ROW = ttnn.ShardOrientation.ROW_MAJOR
COL = ttnn.ShardOrientation.COL_MAJOR
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED

READER_RESIDENT_ARG = 29  # tilize_reader.cpp: in_resident
WRITER_RESIDENT_ARG = 8  # tilize_writer.cpp: out_resident


def _grid(x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})


def _legacy(scheme, buf, grid, shape, orient=ROW):
    return ttnn.MemoryConfig(scheme, buf, ttnn.ShardSpec(grid, shape, orient))


def _nd(buf, grid, shape, orient=ROW):
    return ttnn.MemoryConfig(buf, ttnn.NdShardSpec(ttnn.Shape(shape), grid, orient))


def _residency(input_tensor, mem_config):
    """(reader in_resident flags, writer out_resident flags) of the program the op would build."""
    out_spec = op_module._output_spec(list(input_tensor.shape), input_tensor.dtype, mem_config, ttnn.Tile([32, 32]))
    out = ttnn.allocate_tensor_on_device(out_spec, input_tensor.device())
    desc = pd.create_program_descriptor(input_tensor, out)
    readers = [k for k in desc.kernels if str(k.kernel_source).endswith("tilize_reader.cpp")]
    writers = [k for k in desc.kernels if str(k.kernel_source).endswith("tilize_writer.cpp")]
    return (
        {int(k.compile_time_args[READER_RESIDENT_ARG]) for k in readers},
        {int(k.compile_time_args[WRITER_RESIDENT_ARG]) for k in writers},
    )


def _run(device, shape, in_cfg, out_cfg):
    x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg)
    y = tilize(t, memory_config=out_cfg)
    assert y.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(y), x)
    return t


CASES = {
    # same spec -> both sides resident, no NoC traffic at all
    "same_height": ([1, 1, 512, 64], lambda: _legacy(HEIGHT, L1, _grid(3, 0), (128, 64)), "same", (1, 1)),
    "same_width": ([1, 1, 64, 512], lambda: _legacy(WIDTH, L1, _grid(3, 0), (64, 128)), "same", (1, 1)),
    "same_block_col": ([1, 1, 128, 128], lambda: _legacy(BLOCK, L1, _grid(1, 1), (64, 64), COL), "same", (1, 1)),
    "same_block_row_3x2": ([1, 1, 192, 64 * 3], lambda: _legacy(BLOCK, L1, _grid(2, 1), (96, 64)), "same", (1, 1)),
    "same_nd_rank3": ([2, 64, 64], lambda: _nd(L1, _grid(1, 0), [1, 64, 64]), "same", (1, 1)),
    # partial final shard along the rows (last core holds 64 of 128 rows)
    "same_height_partial": ([1, 1, 320, 64], lambda: _legacy(HEIGHT, L1, _grid(2, 0), (128, 64)), "same", (1, 1)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_same_spec_resident(device, name):
    shape, cfg, _, expected = CASES[name]
    t = _run(device, shape, cfg(), cfg())
    assert _residency(t, cfg()) == ({expected[0]}, {expected[1]})


@pytest.mark.parametrize(
    "shape, in_cfg, out_cfg, expected",
    [
        # input pinned (resident input, accessor writer)
        ([1, 1, 128, 64], lambda: _legacy(HEIGHT, L1, _grid(1, 0), (64, 64)), lambda: ttnn.DRAM_MEMORY_CONFIG, (1, 0)),
        (
            [1, 1, 128, 64],
            lambda: _legacy(HEIGHT, L1, _grid(1, 0), (64, 64)),
            lambda: _legacy(HEIGHT, DRAM, _grid(1, 0), (64, 64)),
            (1, 0),
        ),
        # cross spec: height in, width out (input pinned, writes to the other core's shard)
        (
            [1, 1, 128, 128],
            lambda: _legacy(HEIGHT, L1, _grid(1, 0), (64, 128)),
            lambda: _legacy(WIDTH, L1, _grid(1, 0), (128, 64)),
            (1, 0),
        ),
        # output pinned (accessor reader, resident output)
        ([1, 1, 128, 64], lambda: ttnn.DRAM_MEMORY_CONFIG, lambda: _legacy(HEIGHT, L1, _grid(3, 0), (32, 64)), (0, 1)),
        ([1, 1, 32, 4096], lambda: ttnn.DRAM_MEMORY_CONFIG, lambda: _legacy(WIDTH, L1, _grid(7, 0), (32, 512)), (0, 1)),
        # DRAM-sharded input: never resident -> output pinned
        (
            [1, 1, 128, 64],
            lambda: _legacy(HEIGHT, DRAM, _grid(1, 0), (64, 64)),
            lambda: _legacy(HEIGHT, L1, _grid(1, 0), (64, 64)),
            (0, 1),
        ),
        # sharded_full_grid: 2-core shards cannot fill the grid on 64 tile-rows -> accessors both
        # sides; width-sharded RM input = 2 pages per stick (piece reads)
        (
            [1, 1, 2048, 128],
            lambda: _legacy(WIDTH, L1, _grid(1, 0), (2048, 64)),
            lambda: ttnn.DRAM_MEMORY_CONFIG,
            (0, 0),
        ),
        ([1, 1, 2048, 64], lambda: _legacy(HEIGHT, L1, _grid(1, 0), (1024, 64)), lambda: ttnn.L1_MEMORY_CONFIG, (0, 0)),
        # ragged last width shard (32 of 64 cols): not resident, accessor reads of 3 pages per stick
        ([1, 1, 64, 160], lambda: _legacy(WIDTH, L1, _grid(2, 0), (64, 64)), lambda: ttnn.DRAM_MEMORY_CONFIG, (0, 0)),
    ],
    ids=[
        "height_to_interleaved",
        "height_to_dram_sharded",
        "cross_height_width",
        "interleaved_to_height",
        "interleaved_to_width_wide",
        "dram_sharded_in",
        "full_grid_width_in",
        "full_grid_height_in",
        "ragged_width_in",
    ],
)
def test_crossovers(device, shape, in_cfg, out_cfg, expected):
    t = _run(device, shape, in_cfg(), out_cfg())
    assert _residency(t, out_cfg()) == ({expected[0]}, {expected[1]})


def test_padded_into_block_sharded(device):
    x = torch.randn([1, 1, 50, 100]).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out_cfg = _legacy(BLOCK, L1, _grid(1, 1), (32, 64))
    y = tilize(t, memory_config=out_cfg, pad_value=-1.5)
    assert torch.equal(ttnn.to_torch(y), x)
    padded = y.cpu().to_torch_with_padded_shape()
    expected = torch.nn.functional.pad(x, (0, 28, 0, 14), value=-1.5)
    assert torch.equal(padded, expected)
