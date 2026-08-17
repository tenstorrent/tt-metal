# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op repro for the Quasar resnet50 final global avg_pool2d.

This isolates the exact ttnn.experimental.quasar.avg_pool2d call the resnet50 model makes at
the very end, just before the FC layer (see quasar/tt/ttnn_functional_resnet50.py, the
avg_pool2d call ~line 998) so the LLK team can test/fix it on Quasar with a PCC check, without
running the whole model.

Resnet50 final global-average-pool parameters (input is the layer4 output):
    batch_size = 1
    input_h = input_w = 7          (layer4 spatial output for 224x224 input)
    channels = 2048                (layer4 output channels; x.shape[3])
    kernel_size = [input_h, input_w] = [7, 7]   (global pool)
    stride      = [1, 1]
    padding     = [0, 0, 0, 0]     (4-elem top/bottom/left/right, all zero)
    output_layout = ttnn.TILE_LAYOUT
    dtype = ttnn.bfloat16
    compute_kernel_config = LoFi   (matches the model)
    -> output_h = output_w = 1

Input sharding: the model feeds this pool a WIDTH_SHARDED, TILE_LAYOUT, bf16 tensor of logical
shape [1, 1, batch*input_h*input_w, channels] = [1, 1, 49, 2048]. We reproduce that: width
2048 = 64 tiles, split across an 8x4 = 32-core grid (2048/32 = 64 = 2 tiles/core), matching
the model's fit_width_sharded_cores(2048, 64, device) width-sharded layout.

LAYOUT / GOLDEN CONVERSION (READ CAREFULLY)
-------------------------------------------
ttnn pool tensors are channels-last, flattened: logical layout is [N, H, W, C] collapsed to
[1, 1, N*H*W, C]. torch.nn.functional.avg_pool2d is channels-first: [N, C, H, W]. So:
  1. Build a random NCHW tensor (N, C, H, W).
  2. Golden: F.avg_pool2d(nchw) -> (N, C, 1, 1) for a global 7x7 pool.
  3. ttnn input: permute NCHW -> NHWC, reshape to [1, 1, N*H*W, C].
  4. Golden for compare: permute golden NCHW -> NHWC, reshape to [1, 1, N*1*1, C] = [1,1,N,C].
ttnn.to_torch(out) is [1, 1, N*Hout*Wout, C] = [1, 1, N, C], compared directly against (4).
(padding is 0 here, so count_include_pad is irrelevant.)

PRECISION
---------
The model runs this pool at LoFi math fidelity in bf16, and averaging 49 values accumulates
rounding error, so we assert PCC >= 0.98 (looser than the 0.99 used elsewhere), as instructed.

QUASAR STATUS
-------------
avg_pool2d shares the same pool-reduce compute path (compute_pool_2d.cpp) as max_pool2d, whose
Quasar dest-sync handshake currently hangs on the stem max pool (see test_max_pool2d.py). If
this global avg pool also stalls in the pool reduce, that is the same LLK dest-sync target.

HOW TO RUN
----------
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_avg_pool2d.py
On the functional simulator (kernel asserts OFF so execution reaches the reduce):
  unset TT_METAL_LLK_ASSERTS
  unset TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_avg_pool2d.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# avg_pool2d shares compute_pool_2d's reduce dest-sync path, so it may hang like max_pool2d; cap the run
# so a stall surfaces as a timeout failure (a valid "not-yet-working" signal for the LLK team) not a block.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_avg_pool2d_resnet50_global(mesh_device):
    torch.manual_seed(0)
    device = mesh_device

    # Resnet50 final global avg_pool2d parameters.
    batch_size = 1
    channels = 2048
    input_h = 7
    input_w = 7
    kernel_size = [input_h, input_w]
    stride = [1, 1]
    padding = [0, 0, 0, 0]

    output_h = (input_h - kernel_size[0] + padding[0] + padding[1]) // stride[0] + 1  # 1
    output_w = (input_w - kernel_size[1] + padding[2] + padding[3]) // stride[1] + 1  # 1

    # --- Golden in torch NCHW ---
    x_nchw = torch.rand((batch_size, channels, input_h, input_w), dtype=torch.bfloat16)
    golden_nchw = torch.nn.functional.avg_pool2d(
        x_nchw.float(),
        kernel_size=kernel_size,
        stride=stride,
        padding=0,  # 4-elem [0,0,0,0] -> torch scalar 0
    )  # (N, C, 1, 1)

    # --- ttnn input: NCHW -> NHWC -> [1, 1, N*H*W, C] ---
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_h * input_w, channels).contiguous()

    tensor_height = batch_size * input_h * input_w  # 49
    tensor_width = channels  # 2048

    # WIDTH_SHARDED over an 8x4 = 32-core grid: width 2048 -> 2048/32 = 64 (2 tiles) per core.
    # With a CoreRangeSet, create_sharded_memory_config requires use_height_and_width_as_shard_shape=True
    # and `shape` = the PER-CORE shard shape (else it raises "height and width must be shard shape with
    # CoreRangeSet" at setup). WIDTH_SHARDED keeps the FULL height on every core, so the shard height must
    # be the TILE-PADDED height (nearest_32(49)=64) for the TILE_LAYOUT input to place — this mirrors the
    # model's nearest_32(x.shape[2]) at ttnn_functional_resnet50.py:989.
    # WIDTH_SHARDED, GRID-ADAPTIVE (runs on the full 32-core part AND the 2-core emulator): largest core
    # count that fits the device grid and evenly divides the 64 width tiles (2/4/8/16/32/64) -> tile-aligned
    # width shard. WIDTH_SHARDED keeps the FULL height per core, so shard height = tile-padded nearest_32(49)
    # = 64 for the TILE_LAYOUT input. CoreRangeSet -> needs use_height_and_width_as_shard_shape=True + per-core shape.
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    width_tiles = tensor_width // 32  # 64
    num_cores = max(c for c in range(1, max_cores + 1) if width_tiles % c == 0)
    shard_height = ((tensor_height + 31) // 32) * 32  # nearest_32(49) = 64
    shard_width = (width_tiles // num_cores) * 32
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, shard_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    out = ttnn.experimental.quasar.avg_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
        ),
    )
    ttnn.synchronize_device(device)

    # --- Golden for compare: NCHW -> NHWC -> [1, 1, N*Hout*Wout, C] ---
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * output_h * output_w, channels).contiguous()

    got = ttnn.to_torch(out).float()
    got = got.reshape(1, 1, batch_size * output_h * output_w, channels)

    # bf16 + LoFi averaging of 49 elements -> looser PCC.
    assert_with_pcc(golden_flat, got, pcc=0.98)
