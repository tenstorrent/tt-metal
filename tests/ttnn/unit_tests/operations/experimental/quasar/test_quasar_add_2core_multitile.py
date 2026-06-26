# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Two-core MULTI-TILE ttnn.experimental.quasar.add_ for the RTL Quasar emulator.

Extends test_quasar_add_2core.py (1 tile/core) to num_tiles tiles per core, so that with the
experimental multi-NEO compute (TT_QUASAR_ADD_NEOS=2) each core's shard is large enough to split
across 2 Tensix engines (NEO_0 + NEO_1). With only 1 tile/core a 2-engine split is degenerate (one
engine idle), so this test gives >=2 tiles/core.

Layout: 2 cores in a row (CoreRange((0,0),(1,0))), height-sharded, num_tiles tiles per core.
Total height = 2 * num_tiles tiles. Targets emu-quasar-2x3.

Run (2-NEO, 2-core) with the profiler/timer:
    TT_QUASAR_ADD_NEOS=2 TT_QUASAR_ADD_KERNEL_TIMER=1 TT_METAL_DPRINT_CORES=all \
    TT_METAL_SIMULATOR=<sim>/emu-quasar-2x3/ NNG_SOCKET_ADDR=... NNG_SOCKET_LOCAL_PORT=5555 \
    ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
    python -m pytest -svv tests/.../test_quasar_add_2core_multitile.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

_PCC = 0.9997
_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 0))})  # 2 cores in a row


def _height_sharded_config(num_tiles):
    # Each of the 2 cores owns a shard of num_tiles tiles (num_tiles*32 tall, 32 wide).
    shard = [num_tiles * 32, 32]
    return ttnn.create_sharded_memory_config(
        shard,
        core_grid=_GRID,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("num_tiles", [4, 16])
def test_quasar_add_2core_multitile(device, num_tiles):
    torch.manual_seed(0)
    # 2 cores, each num_tiles tall -> total 2*num_tiles tiles.
    shape = torch.Size([2 * num_tiles * 32, 32])
    a_pt = torch.randn(shape, dtype=torch.bfloat16)
    b_pt = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = _height_sharded_config(num_tiles)
    a_tt = ttnn.from_torch(a_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    out_tt = ttnn.experimental.quasar.add_(a_tt, b_tt, activations=[])

    golden = torch.add(a_pt, b_pt)
    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC)
