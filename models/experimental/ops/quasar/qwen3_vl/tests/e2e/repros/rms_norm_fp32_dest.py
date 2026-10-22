# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: text rms_norm ([1,1,128,2560]) with fp32_dest_acc_en=<argv[1]> (craq-sim aborts the process for True)."""
import sys

import torch
import ttnn


def main():
    fp32 = sys.argv[1] == "True"
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    x, w = torch.randn(1, 1, 128, 2560), torch.randn(2560)
    tx = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    tw = ttnn.from_torch(w.bfloat16().reshape(1, 1, 80, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
    )
    got = ttnn.to_torch(ttnn.rms_norm(tx, epsilon=1e-6, weight=tw, compute_kernel_config=ckc)).float()
    xb = x.bfloat16().float()
    ref = xb * torch.rsqrt(xb.pow(2).mean(-1, keepdim=True) + 1e-6) * w.bfloat16().float()
    p = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
    print(f"rms_norm fp32_dest_acc_en={fp32} pcc={p:.5f}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
