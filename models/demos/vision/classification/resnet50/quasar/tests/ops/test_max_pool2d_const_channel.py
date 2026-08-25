# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar max_pool2d CONFIRMATION test: per-channel-CONSTANT input (row-invariant).

WHY THIS EXISTS
---------------
The Quasar max_pool2d runs end-to-end but returns numerically wrong output on random input
(PCC ~0.03; per-stick reduce means < 0.5, which is arithmetically impossible for a max of
non-negative values). The suspected root cause is the strided reduce-col tilize
(`_llk_unpack_reduce_col_tilizeA_strided_` / the `UNPACR0_STRIDE` primitive) reading L1 at the
WRONG row stride -> it feeds misaligned/garbage data into the reduce. That bug can ONLY corrupt
data that varies from row to row within the window: if every window row holds the SAME value, a
wrong row stride still reads that same value, so the max comes out correct.

This test exploits exactly that: the input is CONSTANT across all spatial positions (H, W, and
the 3x3 window rows) and varies only by CHANNEL. For such input the true max-pool output equals
the per-channel constant everywhere (max of identical values = that value; -inf pad can't lower it).

INTERPRETATION
--------------
  * PASS (output == per-channel constants) -> the reduce/pack pipeline is correct for row-INVARIANT
    data. Since the random-input test FAILS, the defect is exclusively in the row-to-row stride of
    the strided unpack-tilize (UNPACR_STRIDE) -> confirmed LLK/sim primitive bug, NOT a host/op bug.
  * FAIL (garbage even on constant input) -> the defect is BEFORE/AROUND the stride: the
    pack_untilize->scratch layout or the reduce config, not just the row stride.

So this is a discriminator, not a correctness gate on the primary bug.

RUN (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_const_channel.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar
from tests.ttnn.utils_for_testing import assert_with_pcc

# 64ch (=2 channel tiles) is the smallest case that fails on random input; 128 adds a second tile pair.
CHANNELS = [64, 128]

IN_H = IN_W = 32
KERNEL, STRIDE, PADDING = (3, 3), (2, 2), (1, 1)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("channels", CHANNELS, ids=[f"{c}c" for c in CHANNELS])
def test_quasar_max_pool2d_const_channel(mesh_device, channels):
    device = mesh_device
    batch = 1
    out_h = (IN_H - KERNEL[0] + 2 * PADDING[0]) // STRIDE[0] + 1
    out_w = (IN_W - KERNEL[1] + 2 * PADDING[1]) // STRIDE[1] + 1

    # Per-channel CONSTANT: channel c holds value (c+1)/channels everywhere (distinct per channel,
    # in (0, 1]). Constant across N, H, W (and thus across the 3x3 window rows) -> row-invariant.
    per_channel = ((torch.arange(channels, dtype=torch.float32) + 1.0) / channels).to(torch.bfloat16)
    x_nchw = per_channel.view(1, channels, 1, 1).expand(batch, channels, IN_H, IN_W).contiguous()

    input_max = x_nchw.float().max().item()
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE), padding=list(PADDING)
    )
    # Sanity: for a per-channel-constant input the golden max is exactly the per-channel constant.
    expected = per_channel.float().view(1, channels, 1, 1).expand(1, channels, out_h, out_w)
    assert torch.allclose(golden_nchw, expected, atol=1e-2), "golden self-check failed"

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

    # HARD leak invariant (same as the random test): a max can't exceed the input.
    assert got_max <= input_max + 1e-2, (
        f"pool leaked stale L1: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(ch={channels}={channels // 32} tiles, const-per-channel input)"
    )
    # The discriminator: on row-invariant input the reduce/pack must reproduce the per-channel constant.
    # If this PASSES while the random-input test fails, the bug is the UNPACR_STRIDE row stride only.
    assert_with_pcc(golden_flat, got, pcc=0.99)
