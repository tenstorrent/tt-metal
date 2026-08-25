# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the pool reduce **SrcA face-geometry** fix.

Fix site: ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/pool_multi_core_program_factory.cpp
    input_face_geometry = {face_r_dim = tt::constants::FACE_HEIGHT (16), num_faces = 4}   (was pow2(window))

WHY THIS TEST FAILS WITHOUT THE FIX AND PASSES WITH IT
------------------------------------------------------
The Quasar reduce-col strided tilize (_llk_unpack_reduce_col_tilizeA_strided_) accepts the reduce input
(SrcA) in only two shapes: a full 32x32 FOUR-FACE tile (total_row_dim==32 && total_col_dim==32) or a tiny
Nx32 tile (num_faces==2, face_r_dim<=8). A four-face tile therefore MUST have face_r_dim==16 (2 face-rows
x 16 = 32 rows) -- the only way z_dim==4 satisfies validate_buffer_desc's "y_dim must be 16 when z_dim is 4".

The OLD factory set face_r_dim = pow2(min(kernel_h*kernel_w, 16)):
  * windows >= 9 round to 16 -> a valid full tile. The resnet50 stem (3x3, window 9) and global-avg
    (7x7, window 49) both land here, so they were UNAFFECTED by the bug and by the fix (16 -> 16).
  * windows <= 8 give face_r_dim < 16 -> a four-face tile with total_row_dim < 32, which trips BOTH the
    four-face "only supports 32x32 tiles" LLK assert AND validate_buffer_desc's z=4 => y=16 assert.

So the SMALL-window cases below (window <= 8, i.e. face_r_dim 4 / 8 under the old code):
  * WITHOUT the fix  -> trip the LLK assert at reduce init (hard fail with TT_METAL_LLK_ASSERTS on; a
    stale-L1 leak / wrong output with asserts off).
  * WITH the fix     -> SrcA is a valid 32x32 four-face tile (face_r_dim=16), the full-tile padding holds
    the pool identity in the rows past the true window, and the reduce is correct -> PASS.

The 3x3 control case (window 9 -> face_r_dim 16 either way) passes BOTH with and without the fix; it is
included to prove the fix leaves the resnet path unchanged.

Checks per case (same as test_max_pool2d_correctness.py): (1) hard leak invariant out.max <= input.max,
(2) PCC vs torch >= 0.99.

HOW TO RUN (assert-on is the point -- without the fix the small-window cases trip the LLK assert):
    TT_METAL_LLK_ASSERTS=1 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_srca_face_geometry.py
(add TT_METAL_SIMULATOR=~/sim/libttsim.so for craq-sim.)
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (in_h, in_w, channels, kernel, stride, padding, id). N*H*W is tile-aligned (multiple of 32). 64c = 2 tiles.
# `face_r_dim WITHOUT the fix` = pow2(min(kernel_h*kernel_w, 16)); the fix makes it 16 for all of them.
SMALL_WINDOW_CONFIGS = [
    # window 4 -> old face_r_dim 4 (z=4/y=4): FAILS without the fix, passes with it.
    (32, 32, 64, (2, 2), (2, 2), (0, 0), "2x2_w4_facerdim4"),
    # window 8 -> old face_r_dim 8 (z=4/y=8): FAILS without the fix, passes with it.
    (32, 32, 64, (2, 4), (2, 4), (0, 0), "2x4_w8_facerdim8"),
]
CONTROL_CONFIGS = [
    # window 9 -> face_r_dim 16 either way: passes both -- proves the fix doesn't disturb the resnet path.
    (32, 32, 64, (3, 3), (2, 2), (1, 1), "3x3_w9_facerdim16_control"),
]


def _run_pool(mesh_device, is_max, in_h, in_w, channels, kernel, stride, padding):
    device = mesh_device
    torch.manual_seed(0)
    batch = 1

    out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1
    out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1

    # Random input in [0,1): a correct pool output is also in [0,1); any out-of-range value is a stale-L1 leak.
    x_nchw = torch.rand((batch, channels, in_h, in_w), dtype=torch.bfloat16)
    input_max = x_nchw.float().max().item()

    pool_fn = torch.nn.functional.max_pool2d if is_max else torch.nn.functional.avg_pool2d
    golden_nchw = pool_fn(x_nchw.float(), kernel_size=list(kernel), stride=list(stride), padding=list(padding))

    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * in_h * in_w, channels).contiguous()
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * out_h * out_w, channels).contiguous()

    tensor_height = batch * in_h * in_w
    assert tensor_height % 32 == 0, "test sizes must be tile-aligned in N*H*W"

    # Grid-adaptive HEIGHT sharding: largest core count that evenly tile-divides the height and fits the grid
    # (same pattern as test_max_pool2d_correctness.py), so this runs unchanged on the 2-core emulator or a full part.
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

    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    if is_max:
        out = ttnn.experimental.quasar.max_pool2d(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=list(kernel),
            stride=list(stride),
            padding=list(padding),
            dilation=[1, 1],
        )
    else:
        out = ttnn.experimental.quasar.avg_pool2d(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=list(kernel),
            stride=list(stride),
            padding=list(padding),
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
            ),
        )
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).float().reshape(1, 1, batch * out_h * out_w, channels)

    # (1) HARD leak invariant: a correct max/avg pool over [0,1) input never exceeds the input max.
    got_max = got.max().item()
    assert got_max <= input_max + 1e-2, (
        f"pool leaked stale L1: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(kernel={kernel}, ch={channels}, {in_h}x{in_w})"
    )
    # (2) PCC vs torch golden.
    assert_with_pcc(golden_flat, got, pcc=0.99)


# max_pool2d over every case (small windows exercise the fix; the 3x3 control proves no regression).
@pytest.mark.parametrize(
    "in_h,in_w,channels,kernel,stride,padding",
    [c[:6] for c in SMALL_WINDOW_CONFIGS + CONTROL_CONFIGS],
    ids=[c[6] for c in SMALL_WINDOW_CONFIGS + CONTROL_CONFIGS],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_max_pool2d_srca_face_geometry(mesh_device, in_h, in_w, channels, kernel, stride, padding):
    _run_pool(mesh_device, True, in_h, in_w, channels, kernel, stride, padding)


# avg_pool2d over the small-window (unpadded) cases only -- padded avg has torch count_include_pad ambiguity,
# and the goal here is the SrcA face geometry, which is pool-type-independent.
@pytest.mark.parametrize(
    "in_h,in_w,channels,kernel,stride,padding",
    [c[:6] for c in SMALL_WINDOW_CONFIGS],
    ids=[c[6] for c in SMALL_WINDOW_CONFIGS],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_avg_pool2d_srca_face_geometry(mesh_device, in_h, in_w, channels, kernel, stride, padding):
    _run_pool(mesh_device, False, in_h, in_w, channels, kernel, stride, padding)
