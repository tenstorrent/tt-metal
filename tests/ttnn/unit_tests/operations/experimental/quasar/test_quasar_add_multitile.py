# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-tile single-core ttnn.experimental.quasar.add_ for profiling at scale.

The single-tile test (test_quasar_add_1core.py) showed the compute critical path is ~72% DFB-drain
stall (region D) — but single-tile is the worst case for stall (no cross-tile pipelining to hide
latency). This test puts N tiles on ONE core (height-sharded, N tiles tall) so the compute kernel's
per-chunk loop runs multiple times and we can see whether the DFB-drain stall shrinks as a fraction
of total work at scale.

Parametrized by num_tiles. Stays single-core (emu-quasar-1x3) to avoid multi-core Zebu contention.

Run with the timer:
    TT_QUASAR_ADD_KERNEL_TIMER=1 TT_METAL_DPRINT_CORES=all \
    TT_METAL_SIMULATOR=<sim>/emu-quasar-1x3/ NNG_SOCKET_ADDR=... NNG_SOCKET_LOCAL_PORT=5555 \
    ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
    python -m pytest -svv tests/.../test_quasar_add_multitile.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

_PCC = 0.9997


def _height_sharded_config(num_tiles):
    # All num_tiles tiles on a single core (1x1 grid), stacked in height.
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))})
    shard = [num_tiles * 32, 32]  # num_tiles tiles tall, on one core
    return ttnn.create_sharded_memory_config(
        shard,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("num_tiles", [1, 4, 16, 64])
def test_quasar_add_multitile(device, num_tiles):
    torch.manual_seed(0)
    shape = torch.Size([num_tiles * 32, 32])
    a_pt = torch.randn(shape, dtype=torch.bfloat16)
    b_pt = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = _height_sharded_config(num_tiles)
    a_tt = ttnn.from_torch(a_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    out_tt = ttnn.experimental.quasar.add_(a_tt, b_tt, activations=[])

    golden = torch.add(a_pt, b_pt)
    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC)
