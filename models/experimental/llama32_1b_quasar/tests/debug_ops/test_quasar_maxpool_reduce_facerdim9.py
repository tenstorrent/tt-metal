# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro for the LLK team: Quasar max_pool2d compute kernel asserts on a non-power-of-2
`face_r_dim` derived from a 3x3 pooling window.

SYMPTOM
-------
Running any 3x3 max_pool2d on Quasar, the pool compute kernel (NEO) trips two LLK asserts:

  1. tt_metal/tt-llk/tt_llk_quasar/common/inc/ckernel_trisc_common.h:117  validate_buffer_desc
         LLK_ASSERT(y_dim == 16 || 8 || 4 || 2 || 1, "y_dim must be powers of 2 <= 16")
     (hit while configuring the input tile's buffer descriptor for tilize/unpack)

  2. tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_math_reduce.h:355  _llk_math_reduce_init_
         LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape),
                    "Invalid tensor shape for tile-dependent op")
     (tt_metal/tt-llk/common/tensor_shape.h:87 requires face_r_dim in {1,2,4,8,16})

ROOT CAUSE
----------
The generic pool factory packs the pooling window rows into the input tile's face-row dimension:

  ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/pool_multi_core_program_factory.cpp:648
      const uint32_t window_size_hw = kernel_h * kernel_w;      // 3*3 = 9
      const uint32_t raw_face_r     = std::min(window_size_hw, 16u);  // = 9  (NOT a power of 2)
      ... FaceGeometry{ .face_r_dim = raw_face_r /* 9 */, .num_faces = 2 }

So the input CB tile has face_r_dim = 9. In the buffer descriptor this becomes y_dim = face_r_dim = 9
(ckernel_trisc_common.h:398), and the reduce-init reads the same shape. Both Quasar validators only
accept power-of-2 face rows {1,2,4,8,16}, so 9 fails.

WHY WORMHOLE / BLACKHOLE DO NOT HIT THIS
----------------------------------------
The WH/BH pool factory uses the IDENTICAL value:
  ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp:514
      const uint32_t raw_face_r = std::min(window_size_hw, 16u);   // also 9 for a 3x3 window
i.e. face_r_dim = 9 is the INTENDED design — a partial-face reduce over exactly the 9 valid window
rows (no padding). WH's `_llk_math_reduce_init_` never calls the validator, and the WH unpack path
has no pow2 y_dim gate. Quasar ADDED both asserts. The validator's own comment
(common/tensor_shape.h) says: "Will start relaxing this constraint once we test larger tensor shapes."

The trigger is purely kernel_size == [3,3] (window_size_hw = 9); spatial size / batch / core count do
not matter. This test uses a single core and a tiny input to keep it fast and deterministic.

QUESTION FOR LLK
----------------
Can the Quasar reduce + TDMA buffer descriptor support a non-power-of-2 face_r_dim (specifically the
pool window size, 1..16) as WH/BH do? If yes, relaxing validate_buffer_desc /
validate_tensor_shape_tile_dependent_ops_ to allow face_r_dim in [1,16] fixes maxpool with no metal
change. If the HW TDMA engine genuinely cannot encode y_dim = 9, the pool must instead round
face_r_dim up to 16 and pad the extra window rows with the pool identity (-inf for max / 0 for avg).

RUN
---
  CHIP_ARCH / device selected for Quasar; e.g. on the functional simulator:
    TT_METAL_SIMULATOR=~/sim/libttsim.so \
    TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest tests/ttnn/unit_tests/operations/test_quasar_maxpool_reduce_facerdim9.py

A pool that supported face_r_dim=9 would return normally; today the compute kernel asserts as above.
"""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("kernel_size", [[3, 3]], ids=["k3x3_window9"])
def test_quasar_maxpool_facerdim9(mesh_device, kernel_size):
    device = mesh_device

    # Minimal single-core config. The assert depends ONLY on kernel_size=[3,3]
    # (window_size_hw = 9 -> face_r_dim = 9), so the spatial size is kept small.
    batch_size = 1
    channels = 64  # 2 tiles wide
    input_h = 8
    input_w = 8

    tensor_height = batch_size * input_h * input_w  # 64
    tensor_width = channels  # 64

    # Height-shard the whole tensor onto a single core (core (0,0)).
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

    # 3x3 window -> window_size_hw = 9 -> input-CB tile face_r_dim = 9 -> LLK asserts on the compute NEO.
    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
    )

    # Never reached today: the pool compute kernel asserts (face_r_dim=9) before completion.
    ttnn.synchronize_device(device)
    assert out is not None
