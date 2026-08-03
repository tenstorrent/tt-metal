# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC (throwaway): broad SHAPE SWEEP for quasar max_pool2d with a DETERMINISTIC input.

Input: channel c holds value (c+1) everywhere, so for ANY window the max over channel c is (c+1)
=> EVERY output stick must equal [1, 2, ..., C]. Clean structural oracle: reveals pack/routing/reduce
bugs (wrong channels, wrong/zero sticks, split-reader odd-stick failures). Varies window (full-tile vs
partial face), channels (tiles), spatial size and stride/pad so many code paths + shardings are hit.
(Does NOT catch value-inflation leaks — those need random input; see test_max_pool2d_correctness.py.)
"""
import pytest
import torch

import ttnn

# (in_h, in_w, C, kernel, stride, padding, id)
CONFIGS = [
    # --- full-tile window (window_size >= 32 => num_faces=4, no partial face) ---
    (4, 8, 64, (4, 8), (4, 8), (0, 0), "4x8_full_1stick"),
    (32, 16, 64, (8, 4), (8, 4), (0, 0), "8x4_full_16stick"),  # multi-stick/core
    (16, 32, 64, (8, 4), (8, 4), (0, 0), "8x4_full_MULTICORE"),  # full-tile window FORCED >1 stick/core
    # --- one full face (16 rows) ---
    (16, 16, 64, (4, 4), (4, 4), (0, 0), "4x4_16rows"),
    # --- partial face windows (the resnet-shaped / small-window path) ---
    (16, 16, 64, (2, 2), (2, 2), (0, 0), "2x2_16x16_64c"),
    (16, 16, 64, (3, 3), (2, 2), (1, 1), "3x3_16x16_64c"),
    (8, 8, 64, (3, 3), (2, 2), (1, 1), "3x3_8x8_64c"),
    (16, 16, 64, (3, 3), (1, 1), (1, 1), "3x3_s1_16x16_64c"),  # stride 1 -> many sticks
    (16, 16, 64, (5, 5), (2, 2), (0, 0), "5x5_16x16_64c"),
    # --- channel-count sweep (tiles wide) ---
    (16, 16, 32, (3, 3), (2, 2), (1, 1), "3x3_32c_1tile"),
    (16, 16, 96, (3, 3), (2, 2), (1, 1), "3x3_96c_3tile"),
    (16, 16, 128, (3, 3), (2, 2), (1, 1), "3x3_128c_4tile"),
]


@pytest.mark.timeout(240)
@pytest.mark.parametrize("in_h,in_w,C,kernel,stride,padding", [c[:6] for c in CONFIGS], ids=[c[6] for c in CONFIGS])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_scratch_pack_dprint(mesh_device, in_h, in_w, C, kernel, stride, padding):
    device = mesh_device
    batch = 1
    dilation = (1, 1)
    H = batch * in_h * in_w
    assert H % 32 == 0, "input N*H*W must be tile-aligned"
    out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1
    out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1
    n_out = batch * out_h * out_w

    x_nchw = torch.zeros((batch, C, in_h, in_w), dtype=torch.bfloat16)
    for c in range(C):
        x_nchw[0, c, :, :] = float(c + 1)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, H, C).contiguous()

    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    htiles = H // 32
    ncores = max(c for c in range(1, max_cores + 1) if htiles % c == 0)
    # [DEBUG] force multi-stick-per-core (cap cores so out_nhw/core > 1) to test a full-tile
    # (no-tail-clear) window under the split reader with several sticks per core.
    if n_out > 4:
        ncores = max(c for c in range(1, min(2, max_cores) + 1) if htiles % c == 0)
    shard_h = (htiles // ncores) * 32
    cg = ttnn.num_cores_to_corerangeset(ncores, grid, True)
    mc = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, C),
        core_grid=cg,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, mc)
    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch,
        input_h=in_h,
        input_w=in_w,
        channels=C,
        kernel_size=list(kernel),
        stride=list(stride),
        padding=list(padding),
        dilation=list(dilation),
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float().reshape(-1, C)[:n_out]

    # Every output stick must equal [1..C].
    bad_sticks = []
    for s in range(n_out):
        row = got[s].tolist()
        nbad = sum(1 for c in range(C) if abs(row[c] - (c + 1)) > 1e-3)
        if nbad:
            bad_sticks.append((s, nbad))
    ncores_used = ncores
    print(
        f"\n[{in_h}x{in_w} k{kernel} s{stride} p{padding} C{C}] out={n_out} sticks, cores={ncores_used}, "
        f"sticks/core~{n_out // max(ncores_used,1)}: "
        + ("ALL CORRECT" if not bad_sticks else f"{len(bad_sticks)}/{n_out} BAD -> {bad_sticks[:8]}")
    )
    assert (
        not bad_sticks
    ), f"{len(bad_sticks)}/{n_out} sticks wrong (first bad stick0 got={got[bad_sticks[0][0]].tolist()[:8]}...)"
