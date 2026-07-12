# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar max_pool2d 3x3 CHANNEL sweep (channel-tile / block_ct_dim stress).

Sweeps the per-core channel count 32, 64, 96, ... 256 (= 1..8 channel tiles) on a fixed 3x3
stride-2 pad-1 HEIGHT_SHARDED pool. Because the pool is height-sharded, every core holds ALL
`channels`, so channels/32 is exactly the number of channel tiles (block_ct_dim) the reduce sees.
Geometry is 32x32 -> out 16x16 (out_w=16 >= 2), so later output columns exercise the row-to-row
L1 stride: the exact path the unpack_tilizeA_B l1-index fix (PR #49485, replacing the retired
438b29b6e20 block_ct_dim workaround) corrects. Without a correct block_ct_dim scaling, C>=64 with
out_col>=2 aliases an earlier column's data (wrong-window max).

Checks per case (identical to test_max_pool2d_correctness.py):
  1. HARD leak invariant: got.max() <= input.max() + eps.
  2. PCC vs torch.nn.functional.max_pool2d >= 0.99.

Run (craq-sim, kernel asserts OFF):
  ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_channel_sweep.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar
from tests.ttnn.utils_for_testing import assert_with_pcc

# per-core channels = total channels (height-sharded); 32*n for n = 1..8 channel tiles.
CHANNELS = [32, 64, 96, 128, 160, 192, 224, 256]

# fixed 3x3 pool geometry; 32x32 in -> 16x16 out (out_w >= 2 stresses the multi-column stride).
IN_H = IN_W = 32
KERNEL, STRIDE, PADDING = (3, 3), (2, 2), (1, 1)


def _run_max_pool_channels(mesh_device, channels):
    device = mesh_device
    torch.manual_seed(0)
    batch = 1
    out_h = (IN_H - KERNEL[0] + 2 * PADDING[0]) // STRIDE[0] + 1
    out_w = (IN_W - KERNEL[1] + 2 * PADDING[1]) // STRIDE[1] + 1

    x_nchw = torch.rand((batch, channels, IN_H, IN_W), dtype=torch.bfloat16)
    input_max = x_nchw.float().max().item()
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE), padding=list(PADDING)
    )
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * IN_H * IN_W, channels).contiguous()
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * out_h * out_w, channels).contiguous()

    tensor_height = batch * IN_H * IN_W
    assert tensor_height % 32 == 0
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    height_tiles = tensor_height // 32
    num_cores = max(c for c in range(1, max_cores + 1) if height_tiles % c == 0)
    shard_height = (height_tiles // num_cores) * 32
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, mem_config)

    max_pool2d = ttnn.experimental.quasar.max_pool2d if is_quasar() else ttnn.max_pool2d
    out = max_pool2d(
        input_tensor=x,
        batch_size=batch,
        input_h=IN_H,
        input_w=IN_W,
        channels=channels,
        kernel_size=list(KERNEL),
        stride=list(STRIDE),
        padding=list(PADDING),
        dilation=[1, 1],
    )
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).float().reshape(1, 1, batch * out_h * out_w, channels)
    got_max = got.max().item()
    assert got_max <= input_max + 1e-2, (
        f"pool leaked stale L1: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(cores={num_cores}, ch={channels}={channels // 32} tiles, {IN_H}x{IN_W}, out {out_h}x{out_w})"
    )
    assert_with_pcc(golden_flat, got, pcc=0.99)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("channels", CHANNELS, ids=[f"{c}c_{c // 32}tiles" for c in CHANNELS])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_max_pool2d_channel_sweep(mesh_device, channels):
    _run_max_pool_channels(mesh_device, channels)
