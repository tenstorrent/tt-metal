# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: which part of layer_norm (LoFi, bf16 dest) is wrong: plain, gamma, beta, gamma+beta."""
import torch
import ttnn


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    x = torch.randn(1, 1, 64, 1024)
    w, b = torch.randn(1024), torch.randn(1024)
    tx = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    rm = lambda t: ttnn.from_torch(t.bfloat16().reshape(1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    tw, tb = rm(w), rm(b)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    norm = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-6)
    for name, kw, ref in [
        ("plain", {}, norm),
        ("gamma", {"weight": tw}, norm * w),
        ("beta", {"bias": tb}, norm + b),
        ("gamma+beta", {"weight": tw, "bias": tb}, norm * w + b),
    ]:
        got = ttnn.to_torch(ttnn.layer_norm(tx, epsilon=1e-6, compute_kernel_config=ckc, **kw)).float()
        print(
            f"{name:10s} pcc={pcc(got, ref):.5f} pcc_vs_x={pcc(got, x):.5f} absmax={got.abs().max().item():.3g}",
            flush=True,
        )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
