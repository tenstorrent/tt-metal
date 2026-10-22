# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: LM head linear as the model calls it (one 1280-wide vocab split), bf16 everywhere, program_config=None."""
import torch
import ttnn


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid={g.x}x{g.y}", flush=True)
    x, w = torch.randn(1, 1, 32, 2560), torch.randn(2560, 1280) / 50
    tx, tw = (ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for t in (x, w))
    print(f"dtypes: x={tx.dtype} w={tw.dtype}", flush=True)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    for name, kw in (("dtype=bf16", {"dtype": ttnn.bfloat16}), ("no dtype", {})):
        try:
            out = ttnn.linear(tx, tw, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
            ref = x.bfloat16().float() @ w.bfloat16().float()
            got = ttnn.to_torch(out).float()
            p = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
            print(f"{name:10s} pcc={p:.5f} out_dtype={out.dtype}", flush=True)
        except Exception as e:
            msg = next((ln.strip() for ln in str(e).splitlines() if "not supported" in ln), str(e).splitlines()[0])
            print(f"{name:10s} FAIL: {msg[:140]}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
