# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro of the resnet50/quasar MAXPOOL untilize_with_halo hang.

In the full model this is the op right after the stem conv (conv -> max_pool2d). With the conv kernels
no-op'd, the model reaches max_pool2d and hangs inside its input halo (untilize_with_halo): 9 of 32
cores never signal done, the watcher shows halo_gather / pack_untilize loaded, and wait_until_cores_done
spins forever. This isolates that call.

The maxpool input is the stem-conv output: [1,1, N*H*W, C] = [1,1,12544,64], TILE, HEIGHT_SHARDED over the
full compute grid (matches the model's shard: [416,64] on {[0-0 - 7-3]} = 32 cores). We build that tensor
directly and make the identical max_pool2d call (3x3, stride 2, pad 1). Grid-adaptive: on the full 32-core
grid it reproduces the hang; on the 2x3 it uses 2 cores (slower).

Run (craq-sim, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 \
  pytest test_maxpool_hang.py::test_maxpool_hang

A healthy pool returns; the bug hangs in untilize_with_halo (never reaches synchronize_device).
"""

import math

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_maxpool_hang(mesh_device):
    device = mesh_device

    # --- stem maxpool params, verbatim from resnet50.run() (post-conv max_pool2d) ---
    batch_size = 1
    channels = 64  # conv1_output_channels
    input_h = 112  # conv output height (x_height)
    input_w = 112  # conv output width  (x_width)

    tensor_height = batch_size * input_h * input_w  # 12544  (N*H*W)
    tensor_width = channels  # 64

    # --- build the maxpool input directly in the conv-output layout: [1,1,N*H*W,C], TILE, HEIGHT_SHARDED ---
    # The maxpool halo expects the input padded to a (num_cores * TILE) boundary, sharded evenly: e.g. on 32
    # cores round_up(12544, 32*32)=13312 -> 416 rows/core (13 tiles). We must match that exact shard shape
    # (else the halo's borrowed-DFB size mismatches the buffer). Use an explicit shard shape over the full
    # compute-grid core range (mirrors test_conv_hang's pattern).
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = compute_grid.x * compute_grid.y
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)

    tile = 32
    padded_height = math.ceil(tensor_height / (num_cores * tile)) * (num_cores * tile)
    shard_height = padded_height // num_cores  # 416 on 32 cores

    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, tensor_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x_torch = torch.rand((1, 1, tensor_height, tensor_width), dtype=torch.bfloat16)
    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    # --- the exact stem max_pool2d call (its internal untilize_with_halo is where the hang lives) ---
    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
    )

    # If the pool/halo is healthy this returns; the bug hangs above and never reaches here.
    ttnn.synchronize_device(device)
    assert out is not None
