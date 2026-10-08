# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: the vision LayerNorm call as the model makes it ([1,1,256,1024], gamma/beta TILE [1,32,1024]) vs row-major gamma."""
import torch
import ttnn


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    x = torch.randn(1, 1, 256, 1024) * 3 + 0.5
    w, b = torch.randn(1024), torch.randn(1024)
    ref = torch.nn.functional.layer_norm(
        x.bfloat16().float(), (1024,), w.bfloat16().float(), b.bfloat16().float(), eps=1e-6
    )
    tx = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tile = lambda t: ttnn.from_torch(
        t.view(1, 1, 1024).expand(1, 32, 1024).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev
    )
    rm = lambda t: ttnn.from_torch(t.bfloat16().reshape(1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    plain = ttnn.to_torch(ttnn.layer_norm(tx, epsilon=1e-6, compute_kernel_config=ckc)).float()
    r = torch.nn.functional.layer_norm(x.bfloat16().float(), (1024,), eps=1e-6)
    print(f"{'no gamma/beta':24s} pcc={pcc(plain, r):.5f} max_abs={(plain - r).abs().max().item():.3f}", flush=True)
    for name, mk in (("model: TILE [1,32,1024]", tile), ("row-major [1,1,32,32]", rm)):
        got = ttnn.to_torch(
            ttnn.layer_norm(tx, epsilon=1e-6, weight=mk(w), bias=mk(b), compute_kernel_config=ckc)
        ).float()
        print(f"{name:24s} pcc={pcc(got, ref):.5f} max_abs={(got - ref).abs().max().item():.3f}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
