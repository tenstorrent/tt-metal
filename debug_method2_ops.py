#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Op-by-op comparison: device vs torch equivalent for each step in the
Method 2 Approach 2 pipeline.

Pattern:
  reshape -> permute -> reshape -> to_layout(TILE) -> linear
          -> to_layout(ROW) -> reshape -> permute -> reshape

Usage:
    python3 debug_method2_ops.py 2>&1 | tee debug_method2_ops.log
"""

import math, sys
import numpy as np
import torch
import torch.nn.functional as F
import ttnn

TILE_WIDTH = 32


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    if a_f.std() < 1e-8 or b_f.std() < 1e-8:
        return 1.0 if torch.allclose(a_f, b_f, atol=1e-3) else 0.0
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()


def check(step: str, dev_tensor, torch_ref: torch.Tensor, dram):
    dev_host = ttnn.to_torch(
        ttnn.to_layout(dev_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        if dev_tensor.layout == ttnn.TILE_LAYOUT
        else dev_tensor
    ).float()
    ref = torch_ref.float()
    p = pcc(dev_host, ref)
    match = "MATCH ✓" if p >= 0.999 else f"MISMATCH ✗  <-- FIRST DIVERGENCE"
    print(f"  Step {step:<50}  shape={tuple(dev_host.shape)}  PCC={p:.6f}  {match}")
    if p < 0.999:
        print(f"    Device [0,...,:4]: {dev_host.flatten()[:4].tolist()}")
        print(f"    Torch  [0,...,:4]: {ref.flatten()[:4].tolist()}")
    return p >= 0.999


def run(device, batch, ic, oc, H, W, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    dram = ttnn.DRAM_MEMORY_CONFIG
    K = TILE_WIDTH // math.gcd(ic, TILE_WIDTH)  # 32
    packed_ic = ic * K  # 96
    packed_oc = oc * K  # 96
    packed_h = H // K  # 48 (1536) or 40 (1280)
    packed_sp = packed_h * W

    print(f"  K={K}  packed_ic={packed_ic}  packed_oc={packed_oc}  " f"packed_h={packed_h}  packed_sp={packed_sp}")

    torch.manual_seed(0)
    x_orig = torch.randn(batch, ic, H, W, dtype=torch.bfloat16)
    w_orig = torch.randn(oc, ic, 1, 1, dtype=torch.bfloat16)
    b_orig = torch.randn(oc, dtype=torch.bfloat16)

    # Reference conv2d output
    golden = F.conv2d(x_orig.float(), w_orig.float(), bias=b_orig.float())

    # Build packed weight W_packed[c*K+k, oc*K+k] = W[oc,c]
    W_orig_np = w_orig.reshape(oc, ic).float().numpy()
    W_block = np.zeros((packed_ic, packed_oc), dtype=np.float32)
    for k in range(K):
        W_block[k::K, k::K] = W_orig_np.T
    w_packed = torch.from_numpy(W_block).to(torch.bfloat16)

    # Packed bias: b_packed[oc*K+k] = b[oc]
    b_packed = b_orig.repeat_interleave(K)

    # ----------------------------------------------------------------
    # Send input to device as ROW_MAJOR
    # ----------------------------------------------------------------
    tt_x = ttnn.from_torch(x_orig, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram)
    print(f"\n  INPUT  device={tuple(ttnn.to_torch(tt_x).shape)}" f"  torch={tuple(x_orig.shape)}")

    all_ok = True

    # ---- STEP 1: reshape [N,C,H,W] -> [N,C*K,H/K,W] ----------------
    tt1 = ttnn.reshape(tt_x, (batch, packed_ic, packed_h, W))
    ref1 = x_orig.reshape(batch, packed_ic, packed_h, W)
    ok = check("1: reshape [N,C,H,W]->[N,C*K,H/K,W]", tt1, ref1, dram)
    all_ok &= ok

    # ---- STEP 2: permute NCHW -> NHWC [N,H/K,W,C*K] ----------------
    tt2 = ttnn.permute(tt1, dims=(0, 2, 3, 1), memory_config=dram)
    ref2 = ref1.permute(0, 2, 3, 1).contiguous()
    ok = check("2: permute [N,C*K,H/K,W]->[N,H/K,W,C*K]", tt2, ref2, dram)
    all_ok &= ok
    ttnn.deallocate(tt1)

    # ---- STEP 3: reshape [N,H/K,W,C*K] -> [1,1,N*H/K*W,C*K] -------
    tt3 = ttnn.reshape(tt2, (batch, 1, packed_sp, packed_ic))
    ref3 = ref2.reshape(batch, 1, packed_sp, packed_ic)
    ok = check("3: reshape [N,H/K,W,C*K]->[1,1,packed_sp,C*K]", tt3, ref3, dram)
    all_ok &= ok
    ttnn.deallocate(tt2)

    # ---- STEP 4: to_layout TILE -------------------------------------
    tt4 = ttnn.to_layout(tt3, ttnn.TILE_LAYOUT, memory_config=dram)
    ref4 = ref3  # same data, just tiled
    ok = check("4: to_layout TILE", tt4, ref4, dram)
    all_ok &= ok
    ttnn.deallocate(tt3)

    # ---- STEP 5: linear [C*K -> OC*K] ------------------------------
    tt_w = ttnn.from_torch(
        w_packed.reshape(1, 1, packed_ic, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )
    tt_b = ttnn.from_torch(
        b_packed.reshape(1, 1, 1, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )
    tt5 = ttnn.linear(tt4, tt_w, bias=tt_b, memory_config=dram)
    ref5 = (ref4.float() @ w_packed.float() + b_packed.float()).to(torch.bfloat16)
    ok = check("5: linear [C*K->OC*K]", tt5, ref5, dram)
    all_ok &= ok
    ttnn.deallocate(tt4)
    ttnn.deallocate(tt_w)
    ttnn.deallocate(tt_b)

    # ---- STEP 6: to_layout ROW_MAJOR --------------------------------
    tt6 = ttnn.to_layout(tt5, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    ref6 = ref5
    ok = check("6: to_layout ROW_MAJOR", tt6, ref6, dram)
    all_ok &= ok
    ttnn.deallocate(tt5)

    # ---- STEP 7: reshape [1,1,packed_sp,OC*K] -> [N,H/K,W,OC*K] ---
    # reshape returns a VIEW of tt6's buffer → do NOT deallocate tt6 until
    # after permute (Step 8) has finished consuming the view (tt7).
    tt7 = ttnn.reshape(tt6, (batch, packed_h, W, packed_oc))
    ref7 = ref6.reshape(batch, packed_h, W, packed_oc)
    ok = check("7: reshape [1,1,sp,OC*K]->[N,H/K,W,OC*K]", tt7, ref7, dram)
    all_ok &= ok

    # ---- STEP 8: permute NHWC -> NCHW [N,OC*K,H/K,W] ---------------
    tt8 = ttnn.permute(tt7, dims=(0, 3, 1, 2), memory_config=dram)
    ref8 = ref7.permute(0, 3, 1, 2).contiguous()
    # Now safe to release: permute has consumed tt7 (which viewed tt6's buffer)
    ttnn.deallocate(tt6)
    ttnn.deallocate(tt7)
    ok = check("8: permute [N,H/K,W,OC*K]->[N,OC*K,H/K,W]", tt8, ref8, dram)
    all_ok &= ok

    # ---- STEP 9: reshape [N,OC*K,H/K,W] -> [N,OC,H,W] -------------
    # reshape returns a VIEW of tt8's buffer → consume (to_torch) before deallocating tt8.
    tt9 = ttnn.reshape(tt8, (batch, oc, H, W))
    ref9 = ref8.reshape(batch, oc, H, W)
    ok = check("9: reshape [N,OC*K,H/K,W]->[N,OC,H,W]", tt9, ref9, dram)
    all_ok &= ok
    ttnn.deallocate(tt8)

    # ---- Final PCC vs golden ----------------------------------------
    result = ttnn.to_torch(tt9).float()
    ttnn.deallocate(tt9)
    final_pcc = pcc(result, golden)
    print(f"\n  Final PCC vs conv2d golden: {final_pcc:.6f}  " f"{'PASS ✓' if final_pcc >= 0.99 else 'FAIL ✗'}")
    return all_ok and final_pcc >= 0.99


def main():
    print("Method 2 Approach 2 — Op-by-Op Device vs Torch Comparison")
    print("=" * 70)

    CONFIGS = [
        (1, 3, 3, 1536, 1536, "conv2d_1 — Block A (1×3×1536×1536)"),
        (1, 3, 3, 1280, 2304, "conv2d_2 — Block C (1×3×1280×2304)"),
    ]

    device = ttnn.open_device(device_id=0)
    all_pass = True
    try:
        for cfg in CONFIGS:
            batch, ic, oc, H, W, label = cfg
            ok = run(device, batch, ic, oc, H, W, label)
            all_pass &= ok
    finally:
        ttnn.close_device(device)

    print(f"\n{'='*70}")
    print(f"  {'ALL PASS ✓' if all_pass else 'SOME STEPS FAILED ✗'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
