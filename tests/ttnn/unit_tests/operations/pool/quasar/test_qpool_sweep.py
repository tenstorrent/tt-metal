# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar pool CHANNEL-WIDTH sweep — one device session, many C values, per-case verdicts.

Covers the DEST-width regimes from WH/BH testing practice: sub-face (8), one face (16),
1/2/3/4 tiles (32..128), full DST widths (256/512/768), 1.5 DST widths (384), and partial-tile
combinations (40 = 32+8, 144 = 128+16, 280 = 256+24, 392 = 384+8). C > 128 exercises the wide
reduction (in_nblocks_c > 1) — currently EXPECTED RED at num_threads > 1 until the wide-C
threading rework (c-block-as-work-unit) lands.

Input layout per case: TILE for C % 32 == 0 (the harness's proven path), ROW_MAJOR otherwise
(TILE sharding can't express partial-tile shard widths; RM needs only C % 8 == 0 for 16B rows).

Each case prints its banner BEFORE running, so on a hang the log names the case in flight.
OOM is caught per-case and reported as its own verdict instead of aborting the sweep.

Run via the sibling run_qpool.sh sweep.
"""

import os
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_qpool_debug import _build_input, _dump_mismatches

# =============================== CONFIG — edit me ===============================
C_VALUES = [8, 16, 32, 40, 64, 96, 128, 144, 256, 280, 384, 392, 512, 768]
BATCH = 1
IN_H, IN_W = 16, 8  # base for C < 128 (4 cores); C >= 128 auto-drops to 8x4 (see _run_case)
KERNEL = (3, 3)
STRIDE = (2, 2)
PADDING = (1, 1)
PATTERN = "random"
SEED = 0
PCC_THRESHOLD = 0.99
# =================================================================================

SIM_MAX_STICKS = 128


def _run_case(device, channels):
    # craq-sim DFB bug (refined 2026-08-28): the halo's NoC writes stall when C x halo volume
    # crosses ~the 32KB class (16x8x128 @4 cores stalls; 8x4x128 @1 core passes exact; the
    # original 16x16x64 @1 core stall is the same volume class). Keep the 4-core base for small C
    # (multi-core coverage), drop to the 32-stick single-core base for C >= 128. Threading loses
    # no coverage (threads are intra-cluster; halo exchange is the only cross-core traffic).
    in_h, in_w = (IN_H, IN_W) if channels < 128 else (8, 4)
    batch = BATCH
    kernel, stride, padding = list(KERNEL), list(STRIDE), list(PADDING)
    out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1
    out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1
    tensor_height = batch * in_h * in_w
    tiled_input = channels % 32 == 0

    x_nhwc = _build_input(PATTERN, batch, in_h, in_w, channels, SEED, 0).to(torch.bfloat16)
    input_max = x_nhwc.float().max().item()
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nhwc.permute(0, 3, 1, 2).float(), kernel_size=kernel, stride=stride, padding=padding
    )
    golden = golden_nchw.permute(0, 2, 3, 1).reshape(batch * out_h * out_w, channels).contiguous()

    grid = device.compute_with_storage_grid_size()
    height_tiles = tensor_height // 32
    num_cores = max(c for c in range(1, grid.x * grid.y + 1) if height_tiles % c == 0)
    shard_height = (height_tiles // num_cores) * 32
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, channels),
        core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    print(
        f"\nQPOOL-SWEEP: C={channels} layout={'TILE' if tiled_input else 'ROW_MAJOR'} "
        f"in={batch}x{in_h}x{in_w} k={kernel} s={stride} p={padding} cores={num_cores}",
        flush=True,
    )

    x = ttnn.from_torch(
        x_nhwc.reshape(1, 1, tensor_height, channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT if tiled_input else ttnn.ROW_MAJOR_LAYOUT,
    )
    x = x.to(device, mem_config)
    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch,
        input_h=in_h,
        input_w=in_w,
        channels=channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=[1, 1],
    )
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float().reshape(batch * out_h * out_w, channels)
    x.deallocate()
    out.deallocate()

    got_max = got.max().item()
    if got_max > input_max + 1e-2:
        return f"LEAK out.max={got_max:.4f} > in.max={input_max:.4f}"
    max_diff = (got - golden).abs().max().item()
    close = torch.allclose(got, golden, rtol=0.01, atol=0.01)
    pcc = None
    if golden.std() > 0 and got.std() > 0:
        pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
    if not close or (pcc is not None and pcc < PCC_THRESHOLD):
        _dump_mismatches(got, golden, out_h, out_w, channels, 4)
        return f"MISMATCH max_abs_diff={max_diff:.6f}" + (f" pcc={pcc:.6f}" if pcc is not None else "")
    return "PASS" + (f" (pcc={pcc:.6f})" if pcc is not None else "")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_c_sweep(mesh_device):
    tensor_height = BATCH * IN_H * IN_W
    assert tensor_height % 32 == 0
    if os.environ.get("TT_METAL_SIMULATOR") and tensor_height > SIM_MAX_STICKS:
        pytest.fail(f"base shape {tensor_height} sticks > {SIM_MAX_STICKS}: hits the open craq-sim DFB bug")
    for c in C_VALUES:
        assert c % 8 == 0, f"C={c}: ROW_MAJOR sharding needs 16B-aligned rows (C % 8 == 0 for bf16)"

    results = {}
    for c in C_VALUES:
        try:
            results[c] = _run_case(mesh_device, c)
        except RuntimeError as e:
            msg = str(e)
            if "Out of Memory" in msg or "beyond max L1" in msg or "OOM" in msg:
                results[c] = f"OOM: {msg.splitlines()[0][:120]}"
            else:
                results[c] = f"ERROR: {msg.splitlines()[0][:120]}"
        print(f"QPOOL-SWEEP: C={c}: {results[c]}", flush=True)

    print("\nQPOOL-SWEEP SUMMARY:")
    for c in C_VALUES:
        print(f"  C={c:4d}  {results[c]}")
    failures = {c: r for c, r in results.items() if not r.startswith("PASS")}
    assert not failures, f"{len(failures)}/{len(C_VALUES)} C values failed: {sorted(failures)}"
