# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op repro for the Quasar resnet50 STEM max_pool2d.

This isolates the exact ttnn.experimental.quasar.max_pool2d call the resnet50 model makes
right after the stem conv (see quasar/tt/ttnn_functional_resnet50.py, the max_pool2d call
~line 744) so the LLK team can test/fix it on Quasar with a PCC check, without running the
whole model.

Resnet50 stem parameters (conv1 output feeds the pool):
    batch_size = 1
    input_h = input_w = 112        (conv1 output; conv1 is 4x4 stride-1 on a 115x115 fold)
    channels = 64                  (conv1_output_channels)
    kernel_size = [3, 3]
    stride      = [2, 2]
    padding     = [1, 1]
    dilation    = [1, 1]
    -> output_h = output_w = (112 - 3 + 2*1)//2 + 1 = 56

Input sharding: the model feeds the pool a HEIGHT_SHARDED, TILE_LAYOUT, bf16 tensor of
logical shape [1, 1, batch*input_h*input_w, channels] = [1, 1, 12544, 64]. We reproduce that
here (mirroring tests/ttnn/unit_tests/operations/test_craqsim_unpacr_stride_maxpool.py's
sharded-input setup, but with the real stem dims). 12544 rows = 392 tiles; we split across an
8x1 = 8-core grid (392/8 = 49 tiles = 1568 rows/core) so the height shards evenly and
tile-aligned, matching the "stem maxpool shards evenly" behavior of the model.

LAYOUT / GOLDEN CONVERSION (READ CAREFULLY)
-------------------------------------------
ttnn pool tensors are channels-last, flattened: logical layout is [N, H, W, C] collapsed to
[1, 1, N*H*W, C]. torch.nn.functional.max_pool2d is channels-first: [N, C, H, W]. So:
  1. Build a random NCHW tensor (N, C, H, W).
  2. Golden: F.max_pool2d(nchw) -> (N, C, Hout, Wout).
  3. ttnn input: permute NCHW -> NHWC, reshape to [1, 1, N*H*W, C].
  4. Golden for compare: permute golden NCHW -> NHWC, reshape to [1, 1, N*Hout*Wout, C].
ttnn.to_torch(out) is already [1, 1, N*Hout*Wout, C], so it compares directly against (4).
(Max pooling pads with -inf, so the [1,1] padding does not pollute any window max; torch's
implicit -inf padding matches ttnn's max-pool padding.)

KNOWN QUASAR HANG (this is the point of the test -- LLK target)
---------------------------------------------------------------
On Quasar today, max_pool2d HANGS in the pool-reduce dest handshake inside
compute_pool_2d.cpp: the pack side's `tile_regs_wait`, the math WFD, and the unpack `UPTW`
never rendezvous (a dest-sync / dest-register synchronization issue in the pool reduce LLK).
So this test is EXPECTED to hang / fail on Quasar right now; getting it to PASS is the LLK
team's target. It is marked xfail for documentation, but note that xfail does NOT rescue a
true hang -- run it under a timeout (e.g. `timeout 300 pytest ...` or pytest-timeout) so CI
does not block.

HOW TO RUN
----------
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d.py
On the functional simulator (kernel asserts OFF so execution reaches the reduce):
  unset TT_METAL_LLK_ASSERTS
  unset TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(300)  # the known pool-reduce dest-sync HANG would otherwise block the suite; cap it
# @pytest.mark.xfail(
#    reason="Quasar max_pool2d hangs in the pool-reduce dest handshake in compute_pool_2d.cpp "
#    "(pack tile_regs_wait / math WFD / unpack UPTW never rendezvous). LLK dest-sync target. "
#    "NOTE: xfail does not rescue a true hang; run under a timeout.",
#    strict=False,
# )
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_max_pool2d_resnet50_stem(mesh_device):
    torch.manual_seed(0)
    device = mesh_device

    # Resnet50 stem max_pool2d parameters.
    batch_size = 1
    channels = 64
    input_h = 112
    input_w = 112
    kernel_size = [3, 3]
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]

    output_h = (input_h - kernel_size[0] + 2 * padding[0]) // stride[0] + 1  # 56
    output_w = (input_w - kernel_size[1] + 2 * padding[1]) // stride[1] + 1  # 56

    # --- Golden in torch NCHW ---
    x_nchw = torch.rand((batch_size, channels, input_h, input_w), dtype=torch.bfloat16)
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )  # (N, C, Hout, Wout)

    # --- ttnn input: NCHW -> NHWC -> [1, 1, N*H*W, C] ---
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_h * input_w, channels).contiguous()

    tensor_height = batch_size * input_h * input_w  # 12544
    tensor_width = channels  # 64

    # HEIGHT_SHARDED, GRID-ADAPTIVE so it runs on BOTH the full 32-core Quasar part AND the 2-core
    # emulator: pick the largest core count that fits the device grid and evenly divides the 392 height
    # tiles (392 = 2^3*7^2 -> 2/4/8/14/28 all divide) so the shard stays tile-aligned. With a CoreRangeSet,
    # create_sharded_memory_config requires use_height_and_width_as_shard_shape=True and `shape` = the
    # PER-CORE shard shape (else "height and width must be shard shape with CoreRangeSet" at setup).
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    height_tiles = tensor_height // 32  # 392
    num_cores = max(c for c in range(1, max_cores + 1) if height_tiles % c == 0)
    shard_height = (height_tiles // num_cores) * 32
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, tensor_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    # ==== EXPECTED TO HANG HERE ON QUASAR (pool-reduce dest handshake in compute_pool_2d.cpp) ====
    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    ttnn.synchronize_device(device)

    # --- Golden for compare: NCHW -> NHWC -> [1, 1, N*Hout*Wout, C] ---
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * output_h * output_w, channels).contiguous()

    got = ttnn.to_torch(out).float()
    got = got.reshape(1, 1, batch_size * output_h * output_w, channels)

    assert_with_pcc(golden_flat, got, pcc=0.99)
