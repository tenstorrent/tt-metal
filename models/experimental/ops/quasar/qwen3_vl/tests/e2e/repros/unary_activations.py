# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: base unary activations the model needs (gelu for the vision MLP, silu for the text MLP)."""
import torch
import ttnn


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid={g.x}x{g.y}", flush=True)
    x = torch.randn(1, 1, 256, 4096)
    tx = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    xb = x.bfloat16().float()
    cases = {
        "gelu": (lambda: ttnn.gelu(tx), torch.nn.functional.gelu(xb)),
        "gelu approx": (lambda: ttnn.gelu(tx, fast_and_approximate_mode=True), torch.nn.functional.gelu(xb)),
        "silu": (lambda: ttnn.silu(tx), torch.nn.functional.silu(xb)),
        "exp": (lambda: ttnn.exp(tx), torch.exp(xb)),
    }
    for name, (fn, ref) in cases.items():
        try:
            print(f"{name:12s} pcc={pcc(ttnn.to_torch(fn()).float(), ref):.5f}", flush=True)
        except Exception as e:
            msg = next((ln.strip() for ln in str(e).splitlines() if "not supported" in ln), str(e).splitlines()[0])
            print(f"{name:12s} FAIL: {msg[:110]}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
