# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Edge-path coverage for `ttnn.slice` on a TILE-layout tensor on Quasar (interleaved input and output).

WHAT IT COVERS
--------------
On Quasar this program moves up to 32 tiles per barrier once the busiest core has at least 8 output tiles;
below that it keeps the tile-at-a-time kernels. The cases walk both sides of each limit, counted in output
tiles per core:

  - 7 on every core: stays on the tile-at-a-time kernels;
  - 8 on one core and 7 on the rest: the smallest batched case, where most cores run one batch of 7;
  - 8 on every core: one batch of 8;
  - 33: a full batch of 32 followed by a batch of 1;
  - 43: a full batch followed by a short one;
  - 64: two full batches and nothing left over;
  - 100: three full batches and a short one.

Further cases cut the tensor away from the origin in the channel dim, end the slice in the middle of a tile,
use the 32-bit and 8-bit data types, and keep the tensors in L1.

Output shapes follow the device grid, so every case gives the cores the stated number of tiles on the
2-cluster CI grid (TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2") and on a full grid alike.

WHAT IT VALIDATES
-----------------
slice moves values without changing them, so the output must equal the torch slice bit for bit.

RUN
---
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" \
  pytest tests/ttnn/unit_tests/operations/quasar_data_movement/test_quasar_slice_tile_batches.py
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
    if dtype == ttnn.uint8:
        return torch.randint(0, 255, shape, dtype=torch.uint8)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).to(torch.bfloat16)


def _check_slice(device, shape, begins, ends, dtype, memory_config):
    torch.manual_seed(0)
    x = _random_input(shape, dtype)
    tt_in = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    out = ttnn.slice(tt_in, list(begins), list(ends), memory_config=memory_config)

    assert out.layout == ttnn.TILE_LAYOUT
    golden = x[tuple(slice(b, e) for b, e in zip(begins, ends))]
    got = ttnn.to_torch(out)
    assert tuple(got.shape) == tuple(golden.shape)
    assert torch.equal(got.to(golden.dtype), golden)


def _num_cores(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y


def _run_even_split(device, tiles_per_core, dtype, memory_config):
    """Output is one tile row per core, `tiles_per_core` wide, cut one tile in from the top and the left."""
    out_h, out_w = _num_cores(device) * TILE, tiles_per_core * TILE
    shape = (1, 1, out_h + TILE, out_w + 2 * TILE)
    _check_slice(device, shape, (0, 0, TILE, TILE), (1, 1, TILE + out_h, TILE + out_w), dtype, memory_config)


@pytest.mark.parametrize(
    "tiles_per_core",
    [
        pytest.param(7, id="7_tiles_below_batch_gate"),
        pytest.param(8, id="8_tiles_at_batch_gate"),
        pytest.param(33, id="33_tiles_batch_plus_one"),
        pytest.param(43, id="43_tiles_short_last_batch"),
        pytest.param(64, id="64_tiles_two_full_batches"),
        pytest.param(100, id="100_tiles_three_batches_and_a_short_one"),
    ],
)
def test_quasar_slice_tile_batch_sizes(device, tiles_per_core):
    _run_even_split(device, tiles_per_core, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)


def test_quasar_slice_tile_one_core_at_batch_gate(device):
    # One tile row, 7 tiles per core plus one: the first core gets 8 and switches the program to batches.
    out_w = (7 * _num_cores(device) + 1) * TILE
    shape = (1, 1, 2 * TILE, out_w + TILE)
    _check_slice(device, shape, (0, 0, TILE, 0), (1, 1, 2 * TILE, out_w), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)


def test_quasar_slice_tile_channel_offset(device):
    # Channels 1 and 2 of 3 are kept, so the first tile read is a whole channel into the input.
    channel_h, channel_w = _num_cores(device) * TILE, 11 * TILE
    shape = (1, 3, channel_h, channel_w)
    _check_slice(device, shape, (0, 1, 0, 0), (1, 3, channel_h, channel_w), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)


def test_quasar_slice_tile_partial_edge_tiles(device):
    # The slice ends 12 rows and 20 columns into its last tiles, so the edge tiles of the output are part filled.
    out_h, out_w = (2 * _num_cores(device) - 1) * TILE + 12, 16 * TILE + 20
    shape = (1, 1, out_h + 2 * TILE, out_w + 2 * TILE)
    _check_slice(
        device,
        shape,
        (0, 0, TILE, TILE),
        (1, 1, TILE + out_h, TILE + out_w),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.int32, ttnn.uint8], ids=["float32", "int32", "uint8"])
@pytest.mark.parametrize(
    "tiles_per_core",
    [
        pytest.param(7, id="7_tiles_below_batch_gate"),
        pytest.param(43, id="43_tiles_short_last_batch"),
    ],
)
def test_quasar_slice_tile_other_dtypes(device, tiles_per_core, dtype):
    _run_even_split(device, tiles_per_core, dtype, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "tiles_per_core",
    [
        pytest.param(8, id="8_tiles_at_batch_gate"),
        pytest.param(43, id="43_tiles_short_last_batch"),
    ],
)
def test_quasar_slice_tile_l1(device, tiles_per_core):
    _run_even_split(device, tiles_per_core, ttnn.bfloat16, L1_INTERLEAVED)
