# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single-core ttnn.experimental.quasar.add_ test for the RTL Quasar emulator.

The full ResNet-add test shards across 4 cores; the multi-core program dispatch deadlocks
on the emulator (post-bring-up, pre-dispatch). This variant shards a single 32x32 tile onto
a SINGLE core, mirroring the single-core path that the C++ Bmm test runs successfully on
emu-quasar-1x3. It still exercises the real DFB no-bcast ADD path (matches_metal_v2_slice
admits any fully-sharded config with matching a/b/out shard specs — one core is valid).

Run on emu-quasar-1x3 (single worker at logical [0-1]):
    TT_METAL_SIMULATOR=<simulators>/build/emu-quasar-1x3/ \
        NNG_SOCKET_ADDR=tcp://<host>:<P_USER_DBD_PORT> NNG_SOCKET_LOCAL_PORT=5555 \
        ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
        python -m pytest -svv tests/ttnn/unit_tests/operations/experimental/quasar/test_quasar_add_1core.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

_PCC = 0.9997

# Single core, single 32x32 tile, height-sharded.
_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))})  # 1 core
_SHARD = [32, 32]  # one tile
_SHAPE = torch.Size([32, 32])  # exactly one shard on one core


def _height_sharded_config():
    return ttnn.create_sharded_memory_config(
        _SHARD,
        core_grid=_GRID,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("fuse_relu", [False, True])
def test_quasar_add_1core(device, fuse_relu):
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
