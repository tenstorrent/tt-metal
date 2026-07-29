# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
craq-sim repro: Quasar MAX_POOL2D hits an unimplemented instruction in the simulator.

  ERROR: UnimplementedFunctionality: tensix_execute_unpacr0_stride

WHAT / WHY
----------
The Quasar pool compute (compute_pool_2d.cpp `tilizeA_B_reduce_init`) performs a combined
tilize-A + column-reduce, which the LLK implements with the STRIDED unpack sequence
`_llk_unpack_reduce_col_tilizeA_strided_` (tt_llk_quasar/llk_lib/llk_unpack_tilize.h /
llk_api/llk_unpack_reduce_col_tilizeA_strided.h). That sequence emits the UNPACR0_STRIDE
Tensix instruction.

craq-sim (tensix.cpp) stubs out UNPACR0_STRIDE (and every other *_STRIDE variant:
PACR_STRIDE, UNPACR0/1/2_STRIDE, UNPACR_DEST_STRIDE) via UNIMPLEMENTED_TENSIX_INST, so any
Quasar pool/reduce that tilizes-and-reduces aborts here. This is independent of the LLK
tile-shape validators (a separate, already-tracked family); it is a missing simulator
instruction, so it reproduces with kernel asserts on OR off.

HOW TO RUN (functional simulator)
---------------------------------
Kernel asserts must be OFF so execution reaches the instruction rather than tripping the
Quasar LLK tile-shape validators first:

  unset TT_METAL_LLK_ASSERTS
  unset TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest tests/ttnn/unit_tests/operations/test_craqsim_unpacr_stride_maxpool.py

Expected today: the run aborts with
  UnimplementedFunctionality: tensix_execute_unpacr0_stride
A simulator that implements UNPACR_STRIDE would instead complete the pool.

Minimal config: single core, tiny input. The trigger is the pool reduce itself, not the
spatial size — any MAX_POOL2D (or AVG_POOL2D) on Quasar exercises the strided tilize+reduce.
"""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_craqsim_unpacr_stride_maxpool(mesh_device):
    device = mesh_device

    batch_size = 1
    channels = 64
    input_h = 8
    input_w = 8

    tensor_height = batch_size * input_h * input_w  # 64
    tensor_width = channels  # 64

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, tensor_height, tensor_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x_torch = torch.rand((1, 1, tensor_height, tensor_width), dtype=torch.bfloat16)
    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    # The pool reduce (tilizeA_B_reduce) emits UNPACR0_STRIDE, which craq-sim does not implement.
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

    ttnn.synchronize_device(device)
    assert out is not None
