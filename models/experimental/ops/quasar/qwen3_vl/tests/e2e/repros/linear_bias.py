# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: vision MLP FC1 ([1,1,256,1024] x [1024,4096] + bias, gelu), base ttnn.linear vs ttnn.experimental.quasar.linear."""
import time

import torch
import ttnn


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid={g.x}x{g.y}", flush=True)
    x, w, b = torch.randn(1, 1, 256, 1024), torch.randn(1024, 4096) / 32, torch.randn(4096)
    up = lambda t: ttnn.from_torch(t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    tx, tw, tb = up(x), up(w), up(b.reshape(1, -1))
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    xb, wb, bb = (t.bfloat16().float() for t in (x, w, b))
    variants = {
        "no bias": ({}, xb @ wb),
        "bias": ({"bias": tb}, xb @ wb + bb),
        "bias+gelu": ({"bias": tb, "activation": "gelu"}, torch.nn.functional.gelu(xb @ wb + bb)),
    }
    for op_name, op in (("base", ttnn.linear), ("experimental.quasar", ttnn.experimental.quasar.linear)):
        for name, (kw, ref) in variants.items():
            t0 = time.time()
            try:
                out = op(tx, tw, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
                print(
                    f"{op_name:20s} {name:10s} pcc={pcc(ttnn.to_torch(out).float(), ref):.5f} {time.time() - t0:.1f}s",
                    flush=True,
                )
            except Exception as e:
                msg = next((ln.strip() for ln in str(e).splitlines() if "not supported" in ln), str(e).splitlines()[0])
                print(f"{op_name:20s} {name:10s} FAIL: {msg[:110]}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
