# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: LM head linear (one 1280-wide vocab split) with FP8_E4M3 weights and bf16 activations, vs bf16 weights."""
import torch
import ttnn


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    x, w = torch.randn(1, 1, 32, 2560), torch.randn(2560, 1280) / 50
    tx = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = x.bfloat16().float() @ w.bfloat16().float()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    for name, wdtype in (("bf16 weights", ttnn.bfloat16), ("fp8_e4m3 weights", ttnn.DataType.FP8_E4M3)):
        try:
            tw = ttnn.from_torch(
                w.bfloat16().float() if wdtype == ttnn.DataType.FP8_E4M3 else w.bfloat16(),
                dtype=wdtype,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
            )
            out = ttnn.linear(
                tx, tw, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
            )
            got = ttnn.to_torch(out).float()
            p = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
            print(f"{name:18s} w.dtype={tw.dtype} pcc={p:.5f}", flush=True)
        except Exception as e:
            lines = [ln.strip() for ln in str(e).splitlines() if ln.strip() and not ln.strip().startswith("---")]
            print(f"{name:18s} FAIL: {' | '.join(lines[:3])[:200]}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
