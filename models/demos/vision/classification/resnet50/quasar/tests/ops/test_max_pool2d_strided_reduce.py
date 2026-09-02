# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar strided reduce-col tilize repro at the OP level (model-infra copy of the tt-llk unit test).

WHY THIS EXISTS
---------------
The Quasar max_pool2d reduce is driven by `unpack_tilizeA_B_block` ->
`_llk_unpack_reduce_col_tilizeA_strided_` (the `UNPACR0_STRIDE` primitive), which on the sim/emulator
reads L1 at the WRONG row stride and feeds garbage into the reduce (random-input maxpool -> PCC ~0.03,
per-stick means below the single-element floor). The definitive repro is the tt-llk unit test
`tt_metal/tt-llk/tests/python_tests/quasar/test_unpack_reduce_col_tilizeA_strided_quasar.py`, but that
runs in the tt-llk two-phase harness (its own helpers/golden generators, CHIP_ARCH=quasar) which isn't
available in every setup.

This is the FUNCTIONAL equivalent in the model pytest infra: it drives the SAME primitive through
`max_pool2d` on a FULLY DETERMINISTIC input (a normalized index ramp, distinct per (h, w, channel)), so
the golden is exactly torch.max_pool2d and any mis-strided row read shows up as a readable, reproducible
mismatch (unlike random input). Pair it with:
  - test_max_pool2d_const_channel.py  (row-INVARIANT -> should PASS if the bug is purely the row stride)
  - test_max_pool2d_channel_sweep.py  (random -> fails)
Together they localize the defect to the UNPACR_STRIDE row stride.

INTERPRETATION
--------------
  * PASS -> the strided reduce-col tilize handles per-row-varying data correctly (bug elsewhere / fixed).
  * FAIL (PCC low, got.max may exceed input.max) -> the strided read is misaligned; the printed
    sample rows show WHICH input rows the reduce actually picked up.

RUN (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_strided_reduce.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar
from tests.ttnn.utils_for_testing import assert_with_pcc

# Small deterministic sweep (channel tiles = channels/32), mirroring the tt-llk dim sweep spirit.
CHANNELS = [32, 64, 128]

IN_H = IN_W = 32
KERNEL, STRIDE, PADDING = (3, 3), (2, 2), (1, 1)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("channels", CHANNELS, ids=[f"{c}c" for c in CHANNELS])
def test_quasar_max_pool2d_strided_reduce(mesh_device, channels):
    device = mesh_device
    batch = 1
    out_h = (IN_H - KERNEL[0] + 2 * PADDING[0]) // STRIDE[0] + 1
    out_w = (IN_W - KERNEL[1] + 2 * PADDING[1]) // STRIDE[1] + 1

    # Fully deterministic ramp: value is a distinct increasing function of (h, w, c), normalized to (0, 1).
    # Every window element differs, and the max is a specific known element -> a mis-strided row read
    # produces a specific wrong value, not just noise.
    n_elems = IN_H * IN_W * channels
    idx = torch.arange(n_elems, dtype=torch.float32)
    ramp = ((idx + 1.0) / (n_elems + 1.0)).view(IN_H, IN_W, channels)  # [h, w, c] in (0,1)
    x_nchw = ramp.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)  # [1, c, h, w]

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

    # Readable diagnostic: first output stick, channel 0..3 — golden vs got. On a mis-strided read the
    # got values are specific wrong ramp entries (reveals which input rows the reduce picked up).
    print(
        f"[strided_reduce ch={channels}] out_stick0 ch0..3 golden={golden_flat[0,0,0,:4].tolist()} "
        f"got={got[0,0,0,:4].tolist()}  got.max={got_max:.4f} input.max={input_max:.4f}"
    )

    assert got_max <= input_max + 1e-2, (
        f"pool leaked stale L1 / mis-strided read: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(ch={channels}={channels // 32} tiles, deterministic ramp)"
    )
    assert_with_pcc(golden_flat, got, pcc=0.99)
