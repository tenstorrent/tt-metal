# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Edge-path coverage for `ttnn.tilize` on Quasar (interleaved input, default multi-core program).

WHAT IT COVERS
--------------
On Quasar this program spreads the row reads of a block (one block = one row of tiles across the tensor width)
over 1, 2 or 4 data-movement cores, and writes each finished block with one barrier. The reader count is the
smaller of two limits: what the blocks held in flight leave of L1, and how many blocks a core has to read.
The cases walk both limits:

  - 1, 2, 3, 4 and 8 blocks per core, which ask for 1, 2, 2, 4 and 4 readers;
  - a last core that holds 3 blocks while the program runs 4 readers;
  - rows wide enough that L1 allows only 2 readers, or only 1, however many blocks there are;
  - float32 and int32, whose tiles are twice the size of bfloat16 tiles;
  - input and output in L1 instead of DRAM.

Tensor heights follow the device grid, so every case gives each core the stated number of blocks on the
2-cluster CI grid (TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2") and on a full grid alike.

WHAT IT VALIDATES
-----------------
tilize moves values without changing them, so the output must equal the input bit for bit.

RUN
---
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" \
  pytest tests/ttnn/unit_tests/operations/quasar_data_movement/test_quasar_tilize_reader_threads.py
"""

import pytest
import torch

import ttnn

# Every case uses the same device configuration, so the module opens the device once.
pytestmark = pytest.mark.use_module_device

TILE = 32
L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)


def _random_input(shape, dtype):
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).to(torch.bfloat16)


def _run_tilize(device, blocks_per_core, last_core_blocks, width_tiles, dtype, memory_config):
    """Tilize a tensor `width_tiles` wide whose blocks split as `blocks_per_core` on every core but the last."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    num_blocks = (num_cores - 1) * blocks_per_core + last_core_blocks
    shape = (1, 1, num_blocks * TILE, width_tiles * TILE)

    torch.manual_seed(0)
    x = _random_input(shape, dtype)
    tt_in = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config)

    out = ttnn.tilize(tt_in, memory_config=memory_config)

    assert out.layout == ttnn.TILE_LAYOUT
    got = ttnn.to_torch(out)
    assert tuple(got.shape) == shape
    assert torch.equal(got.to(x.dtype), x)


# (blocks on every core but the last, blocks on the last core, block width in tiles)
_FOUR_READERS = (4, 4, 4)
_LAST_CORE_SHORT = (4, 3, 8)


@pytest.mark.parametrize(
    "blocks_per_core, last_core_blocks, width_tiles",
    [
        pytest.param(1, 1, 2, id="1_block_per_core"),
        pytest.param(2, 2, 4, id="2_blocks_per_core"),
        pytest.param(3, 3, 4, id="3_blocks_per_core"),
        pytest.param(*_FOUR_READERS, id="4_blocks_per_core"),
        pytest.param(8, 8, 2, id="8_blocks_per_core"),
        pytest.param(*_LAST_CORE_SHORT, id="last_core_3_blocks_of_4"),
    ],
)
def test_quasar_tilize_blocks_per_core(device, blocks_per_core, last_core_blocks, width_tiles):
    _run_tilize(device, blocks_per_core, last_core_blocks, width_tiles, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "width_tiles",
    [
        pytest.param(256, id="256_tile_rows_two_readers_fit"),
        pytest.param(512, id="512_tile_rows_one_reader_fits"),
    ],
)
def test_quasar_tilize_wide_rows(device, width_tiles):
    _run_tilize(device, 4, 4, width_tiles, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.int32], ids=["float32", "int32"])
@pytest.mark.parametrize(
    "blocks_per_core, last_core_blocks, width_tiles",
    [
        pytest.param(*_FOUR_READERS, id="4_blocks_per_core"),
        pytest.param(*_LAST_CORE_SHORT, id="last_core_3_blocks_of_4"),
    ],
)
def test_quasar_tilize_32bit(device, blocks_per_core, last_core_blocks, width_tiles, dtype):
    _run_tilize(device, blocks_per_core, last_core_blocks, width_tiles, dtype, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "blocks_per_core, last_core_blocks, width_tiles",
    [
        pytest.param(1, 1, 2, id="1_block_per_core"),
        pytest.param(*_LAST_CORE_SHORT, id="last_core_3_blocks_of_4"),
    ],
)
def test_quasar_tilize_l1(device, blocks_per_core, last_core_blocks, width_tiles):
    _run_tilize(device, blocks_per_core, last_core_blocks, width_tiles, ttnn.bfloat16, L1_INTERLEAVED)
