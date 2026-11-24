# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Edge-path coverage for `ttnn.untilize` on Quasar (interleaved input, default multi-core program).

WHAT IT COVERS
--------------
On Quasar this program reads up to 32 tiles per barrier and lets four data-movement cores write the rows of
each block (one block = one row of tiles across the tensor width). The cases walk the paths where the
batching and the split of blocks over cores and writers can go wrong:

  - blocks wider than one batch. A batch has to end at the block boundary, so a 33-tile block is read as
    32 + 1, a 37-tile block as 32 + 5 and a 64-tile block as 32 + 32;
  - a last core that receives fewer blocks than the others;
  - a single block per core;
  - float32 and int32, whose tiles are twice the size of bfloat16 tiles;
  - input and output in L1 instead of DRAM.

Tensor heights follow the device grid, so every case gives each core the stated number of blocks on the
2-cluster CI grid (TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2") and on a full grid alike. On the CI grid
every case, the wide ones included, takes this program and not the block program.

WHAT IT VALIDATES
-----------------
untilize moves values without changing them, so the output must equal the input bit for bit.

RUN
---
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" \
  pytest tests/ttnn/unit_tests/operations/quasar_data_movement/test_quasar_untilize_multicore.py
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


def _run_untilize(device, blocks_per_core, last_core_blocks, width_tiles, dtype, memory_config):
    """Untilize a tensor `width_tiles` wide whose blocks split as `blocks_per_core` on every core but the last."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    num_blocks = (num_cores - 1) * blocks_per_core + last_core_blocks
    shape = (1, 1, num_blocks * TILE, width_tiles * TILE)

    torch.manual_seed(0)
    x = _random_input(shape, dtype)
    tt_in = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    out = ttnn.untilize(tt_in, memory_config=memory_config)

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out)
    assert tuple(got.shape) == shape
    assert torch.equal(got.to(x.dtype), x)


# (blocks on every core but the last, blocks on the last core, block width in tiles)
_ONE_BLOCK_PER_CORE = (1, 1, 2)
_UNEVEN_SPLIT = (4, 3, 4)
_SHORT_LAST_BATCH = (2, 2, 37)


@pytest.mark.parametrize(
    "blocks_per_core, last_core_blocks, width_tiles",
    [
        pytest.param(*_ONE_BLOCK_PER_CORE, id="one_block_per_core"),
        pytest.param(*_UNEVEN_SPLIT, id="last_core_one_block_short"),
        pytest.param(4, 4, 16, id="16_tile_blocks"),
        pytest.param(2, 2, 32, id="32_tile_blocks_one_full_batch"),
        pytest.param(2, 2, 33, id="33_tile_blocks_batch_plus_one"),
        pytest.param(*_SHORT_LAST_BATCH, id="37_tile_blocks_short_last_batch"),
        pytest.param(2, 2, 64, id="64_tile_blocks_two_full_batches"),
    ],
)
def test_quasar_untilize_multicore_block_widths(device, blocks_per_core, last_core_blocks, width_tiles):
    _run_untilize(device, blocks_per_core, last_core_blocks, width_tiles, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.int32], ids=["float32", "int32"])
@pytest.mark.parametrize(
    "blocks_per_core, last_core_blocks, width_tiles",
    [
        pytest.param(*_UNEVEN_SPLIT, id="last_core_one_block_short"),
        pytest.param(*_SHORT_LAST_BATCH, id="37_tile_blocks_short_last_batch"),
    ],
)
def test_quasar_untilize_multicore_32bit(device, blocks_per_core, last_core_blocks, width_tiles, dtype):
    _run_untilize(device, blocks_per_core, last_core_blocks, width_tiles, dtype, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "blocks_per_core, last_core_blocks, width_tiles",
    [
        pytest.param(*_UNEVEN_SPLIT, id="last_core_one_block_short"),
        pytest.param(4, 4, 16, id="16_tile_blocks"),
    ],
)
def test_quasar_untilize_multicore_l1(device, blocks_per_core, last_core_blocks, width_tiles):
    _run_untilize(device, blocks_per_core, last_core_blocks, width_tiles, ttnn.bfloat16, L1_INTERLEAVED)
