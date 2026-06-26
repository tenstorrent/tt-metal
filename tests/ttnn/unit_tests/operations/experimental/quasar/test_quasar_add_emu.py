# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Emulator-friendly ttnn.experimental.quasar.add_ test, tuned for fast iteration on the
RTL Quasar emulator during Quasar bring-up work in tt-metal.

Two deliberate choices make this practical on the cycle-accurate RTL emulator (~480x
slower than silicon, where device bring-up dominates wall time):

  1. @pytest.mark.use_module_device -> the (slow) device bring-up is paid ONCE for the
     whole module, not once per test function.
  2. A 2-core shard that fits the smallest launchable RTL config, emu-quasar-2x3, whose
     functional_workers are [0-1, 1-1] (a 2-wide row at y=1). Fewer cores = far cheaper
     bring-up than the 8x4 (32-core) config.

Run on the 2x3 RTL emulator:
    TT_METAL_SIMULATOR=<simulators>/build/emu-quasar-2x3/ \
        NNG_SOCKET_ADDR=tcp://<host>:<P_USER_DBD_PORT> NNG_SOCKET_LOCAL_PORT=5555 \
        ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
        python -m pytest -svv tests/ttnn/unit_tests/operations/experimental/quasar/test_quasar_add_emu.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Module-scoped device: bring the emulated device up once for every test in this file.
pytestmark = pytest.mark.use_module_device

_PCC = 0.9997

# emu-quasar-2x3 functional_workers = [0-1, 1-1]: a 2-wide row at y=1.
# Use a 2-core column at x=0 (y=0..1 in logical core-grid terms) for a height-sharded add.
# 2 shards of one 32x32 tile each -> 2 cores, which fits the 2-worker config.
_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))})  # 2 cores
_SHARD = [32, 32]  # one tile per shard
_SHAPE = torch.Size([2 * 32, 32])  # 2 tiles tall -> 2 shards


def _height_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _run_add(device, fuse_relu):
    torch.manual_seed(0)
    a_pt = torch.randn(_SHAPE, dtype=torch.bfloat16)
    b_pt = torch.randn(_SHAPE, dtype=torch.bfloat16)

    mem_config = _height_sharded_config(_SHARD, _GRID)
    a_tt = ttnn.from_torch(a_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)] if fuse_relu else []
    out_tt = ttnn.experimental.quasar.add_(a_tt, b_tt, activations=activations)

    golden = torch.add(a_pt, b_pt)
    if fuse_relu:
        golden = torch.relu(golden)
    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC)


def test_quasar_add_2core(device):
    """Plain 2-core height-sharded add."""
    _run_add(device, fuse_relu=False)


def test_quasar_add_2core_relu(device):
    """Same, with fused RELU (the ResNet residual pattern)."""
    _run_add(device, fuse_relu=True)
