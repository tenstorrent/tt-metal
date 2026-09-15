# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
SINGLE-CORE (1x3 grid) Quasar max_pool2d repro for the LLK team.

WHY THIS EXISTS
---------------
The stem max_pool2d repro (test_max_pool2d.py) is grid-adaptive and shards across multiple cores on
the 2x3 emulator grid. The LLK team asked for a variant that runs on ONE core on the 1x3 simulator
grid, so the pool-reduce (`unpack_tilizeA_B` + `reduce_tile` in compute_pool_2d.cpp) can be debugged
in isolation — no cross-core sharding.

This is the SAME op (`ttnn.experimental.quasar.max_pool2d`, 3x3 / stride 2 / pad 1, bf16,
HEIGHT_SHARDED, TILE_LAYOUT) as the stem pool, but with a SMALL 32x32 input FORCED onto a SINGLE core
(num_cores=1, core (0,0)) so the whole reduce runs on one core and fits its L1.

It targets the known Quasar pool issues (reduce dest-sync hang and/or the strided reduce-col tilize
producing wrong values — see ~/reduce_col_strided.md and test_max_pool2d_strided_reduce.py), reduced
to the smallest single-core case.

RUN (1x3 simulator, slow dispatch + forced JIT; kernel asserts OFF to reach the reduce):
  unset TT_METAL_LLK_ASSERTS; unset TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_1x3.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(300)  # a pool-reduce dest-sync hang would otherwise block the suite; cap it
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_max_pool2d_1x3_single_core(mesh_device):
    torch.manual_seed(0)
    device = mesh_device

    # Small single-core pool: stem reduce config (3x3 s2 p1, 64ch) on a tiny 32x32 input.
    batch_size = 1
    channels = 64
    input_h = 32
    input_w = 32
    kernel_size = [3, 3]
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]

    output_h = (input_h - kernel_size[0] + 2 * padding[0]) // stride[0] + 1  # 16
    output_w = (input_w - kernel_size[1] + 2 * padding[1]) // stride[1] + 1  # 16

    # --- Golden in torch NCHW ---
    x_nchw = torch.rand((batch_size, channels, input_h, input_w), dtype=torch.bfloat16)
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )

    # --- ttnn input: NCHW -> NHWC -> [1, 1, N*H*W, C] ---
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_h * input_w, channels).contiguous()

    tensor_height = batch_size * input_h * input_w  # 1024
    tensor_width = channels  # 64
    assert tensor_height % 32 == 0

    # FORCE a SINGLE core: the whole height shard lives on core (0,0). 1024 rows = 32 tiles, 64ch =
    # 2 width tiles -> 64 tiles ~= 128 KB, comfortably fits one core's L1.
    num_cores = 1
    shard_height = tensor_height  # all rows on the one core
    grid = device.compute_with_storage_grid_size()
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
    got = ttnn.to_torch(out).float().reshape(1, 1, batch_size * output_h * output_w, channels)

    assert_with_pcc(golden_flat, got, pcc=0.99)
