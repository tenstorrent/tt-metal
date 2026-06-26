# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Two-core ttnn.experimental.quasar.add_ test for the RTL Quasar emulator.

Minimal multi-core case: 2 shards (one 32x32 tile each) across 2 cores. This is the smallest
step up from the passing single-core test (test_quasar_add_1core.py) and is the real probe of
whether multi-core sharded program dispatch works on the emulator (the 32-core 9x4 run
deadlocked post-bring-up/pre-dispatch; this isolates whether ANY multi-core dispatch works).

Targets emu-quasar-2x3, whose functional_workers are [0-1, 1-1] (a 2-wide row). The logical
2-core CoreRange((0,0),(1,0)) maps onto those two workers.

Run on emu-quasar-2x3:
    TT_METAL_SIMULATOR=<simulators>/build/emu-quasar-2x3/ \
        NNG_SOCKET_ADDR=tcp://<host>:<P_USER_DBD_PORT> NNG_SOCKET_LOCAL_PORT=5555 \
        ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
        python -m pytest -svv tests/ttnn/unit_tests/operations/experimental/quasar/test_quasar_add_2core.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

_PCC = 0.9997

# Two cores in a row, one 32x32 tile per shard -> 2 shards total.
_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 0))})  # 2 cores
_SHARD = [32, 32]  # one tile per shard
_SHAPE = torch.Size([2 * 32, 32])  # 2 tiles tall -> 1 shard per core


def _height_sharded_config():
    return ttnn.create_sharded_memory_config(
        _SHARD,
        core_grid=_GRID,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("fuse_relu", [False, True])
def test_quasar_add_2core(device, fuse_relu):
    torch.manual_seed(0)
    a_pt = torch.randn(_SHAPE, dtype=torch.bfloat16)
    b_pt = torch.randn(_SHAPE, dtype=torch.bfloat16)

    mem_config = _height_sharded_config()
    a_tt = ttnn.from_torch(a_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)] if fuse_relu else []
    out_tt = ttnn.experimental.quasar.add_(a_tt, b_tt, activations=activations)

    golden = torch.add(a_pt, b_pt)
    if fuse_relu:
        golden = torch.relu(golden)
    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC)
