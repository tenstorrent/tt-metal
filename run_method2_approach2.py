#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Standalone script for Method 2 Approach 2 — on-device row-group spatial packing.

Runs both BEV YUV conv2d configurations, compares against a PyTorch 1x1 conv2d
golden reference, and reports PCC for each.

Usage:
    python3 run_method2_approach2.py 2>&1 | tee method2_approach2_run.log
"""

import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
import ttnn

TILE_WIDTH = 32

# ---- configs ---------------------------------------------------------------
CONFIGS = [
    (1, 3, 3, 1536, 1536, "conv2d_1 — Block A (1×3×1536×1536)"),
    (1, 3, 3, 1280, 2304, "conv2d_2 — Block C (1×3×1280×2304)"),
]


# ---- helpers ---------------------------------------------------------------


def spatial_pack_factor(in_channels: int) -> int:
    return TILE_WIDTH // math.gcd(in_channels, TILE_WIDTH)


def make_packed_weight(torch_weight: torch.Tensor, in_channels: int, out_channels: int, K: int) -> torch.Tensor:
    """Row-group packing: W_packed[c*K+k, oc*K+k] = W[oc,c] for all k."""
    W_orig = torch_weight.reshape(out_channels, in_channels).float().numpy()
    W_block = np.zeros((in_channels * K, out_channels * K), dtype=np.float32)
    for k in range(K):
        W_block[k::K, k::K] = W_orig.T
    return torch.from_numpy(W_block).to(torch.bfloat16)


def make_packed_bias(torch_bias: torch.Tensor, out_channels: int, K: int) -> torch.Tensor:
    """bias_packed[oc*K+k] = bias[oc]  →  repeat_interleave."""
    return torch_bias.reshape(out_channels).repeat_interleave(K)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()


# ---- main run ---------------------------------------------------------------


def run_config(device, batch, in_channels, out_channels, input_height, input_width, label):
    print(f"\n{'='*70}")
    print(f"  Config: {label}")
    print(f"{'='*70}")

    dram = ttnn.DRAM_MEMORY_CONFIG
    K = spatial_pack_factor(in_channels)
    packed_ic = in_channels * K
    packed_oc = out_channels * K
    packed_h = input_height // K
    packed_sp = packed_h * input_width

    print(f"  K={K}  packed_ic={packed_ic}  packed_oc={packed_oc}  " f"packed_h={packed_h}  packed_sp={packed_sp}")

    assert input_height % K == 0, f"input_height={input_height} not divisible by K={K}"

    # Random tensors
    torch.manual_seed(0)
    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(out_channels, dtype=torch.bfloat16)

    # PyTorch golden reference
    golden = F.conv2d(torch_input.float(), torch_weight.float(), bias=torch_bias.float())
    print(f"\n  Golden (torch 1×1 conv2d):")
    print(f"    shape: {tuple(golden.shape)}")
    print(f"    [0,0,0,:4]: {golden[0,0,0,:4].tolist()}")

    # Build packed weight and bias
    torch_w_packed = make_packed_weight(torch_weight, in_channels, out_channels, K)
    torch_b_packed = make_packed_bias(torch_bias, out_channels, K)

    # Sanity-check weight in Python
    x_py = (
        torch_input.reshape(batch, packed_ic, packed_h, input_width)
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(1, 1, packed_sp, packed_ic)
        .float()
    )
    out_py = (
        (x_py @ torch_w_packed.float() + torch_b_packed.float())
        .reshape(batch, packed_h, input_width, packed_oc)
        .permute(0, 3, 1, 2)
        .contiguous()
        .reshape(batch, out_channels, input_height, input_width)
    )
    pcc_py = pcc(out_py, golden)
    print(f"\n  Python host simulation PCC : {pcc_py:.6f}")

    # ---- Device computation (Approach 2: on-device reshape + permute) ----
    tt_w = ttnn.from_torch(
        torch_w_packed.reshape(1, 1, packed_ic, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )
    tt_b = ttnn.from_torch(
        torch_b_packed.reshape(1, 1, 1, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )

    # Pack entirely on HOST then send to device (bypass device permute to isolate issue)
    torch_packed_flat = (
        torch_input.reshape(batch, packed_ic, packed_h, input_width)
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(1, 1, packed_sp, packed_ic)
    )
    tt_tile = ttnn.from_torch(
        torch_packed_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
    )

    # --- Intermediate check: packed input on device vs Python ---
    tile_from_device = ttnn.to_torch(tt_tile).float()
    tile_from_python = x_py  # [1, 1, packed_sp, packed_ic]
    pcc_tile = pcc(tile_from_device, tile_from_python)
    print(f"\n  Intermediate PCC (packed input: device vs python): {pcc_tile:.6f}")
    print(f"    Device tile[0,0,0,:4]: {tile_from_device[0,0,0,:4].tolist()}")
    print(f"    Python tile[0,0,0,:4]: {tile_from_python[0,0,0,:4].tolist()}")

    # Step 6: single matmul — reads only 17.7 MB
    tt_out_packed = ttnn.linear(tt_tile, tt_w, bias=tt_b, memory_config=dram)
    ttnn.deallocate(tt_tile)
    ttnn.deallocate(tt_w)
    ttnn.deallocate(tt_b)

    # --- Intermediate check: linear output on device vs Python ---
    out_packed_dev = ttnn.to_torch(ttnn.to_layout(tt_out_packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)).float()
    out_packed_py = tile_from_python @ torch_w_packed.float() + torch_b_packed.float()
    pcc_linear = pcc(out_packed_dev, out_packed_py)
    print(f"\n  Intermediate PCC (linear output: device vs python): {pcc_linear:.6f}")
    print(f"    Device out[0,0,0,:4]: {out_packed_dev[0,0,0,:4].tolist()}")
    print(f"    Python out[0,0,0,:4]: {out_packed_py[0,0,0,:4].tolist()}")

    # Unpack entirely on HOST to confirm linear output is correct
    out_packed_host = ttnn.to_torch(
        ttnn.to_layout(tt_out_packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    )  # [1,1,packed_sp,packed_oc]
    ttnn.deallocate(tt_out_packed)
    result = (
        out_packed_host.reshape(batch, packed_h, input_width, packed_oc)  # [N,H/K,W,OC*K]
        .permute(0, 3, 1, 2)
        .contiguous()  # [N,OC*K,H/K,W]
        .reshape(batch, out_channels, input_height, input_width)
    )  # [N,OC,H,W]

    pcc_device = pcc(result, golden)

    print(f"\n  Device result (Method 2 Approach 2):")
    print(f"    shape: {tuple(result.shape)}")
    print(f"    dtype: {result.dtype}")
    print(f"    [0,0,0,:4]: {result[0,0,0,:4].tolist()}")
    print(f"    Golden [0,0,0,:4]: {golden[0,0,0,:4].tolist()}")

    # Per-channel PCC
    print(f"\n  Per-channel PCC:")
    for oc in range(out_channels):
        ch_pcc = pcc(result[0, oc], golden[0, oc])
        print(f"    OC={oc}: PCC={ch_pcc:.6f}")

    print(f"\n  Overall PCC : {pcc_device:.6f}  " f"{'PASS ✓' if pcc_device >= 0.99 else 'FAIL ✗'}")

    return pcc_device


def main():
    print("Method 2 Approach 2 — On-Device Row-Group Spatial Packing")
    print("=" * 70)
    print("Hardware: Wormhole N150")
    print(f"Torch: {torch.__version__}  Configs: {len(CONFIGS)}")

    device = ttnn.open_device(device_id=0)
    results = []
    try:
        for cfg in CONFIGS:
            batch, ic, oc, h, w, label = cfg
            p = run_config(device, batch, ic, oc, h, w, label)
            results.append((label, p))
    finally:
        ttnn.close_device(device)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for label, p in results:
        status = "PASS ✓" if p >= 0.99 else "FAIL ✗"
        print(f"  {label:<45}  PCC={p:.6f}  {status}")
        if p < 0.99:
            all_pass = False

    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
