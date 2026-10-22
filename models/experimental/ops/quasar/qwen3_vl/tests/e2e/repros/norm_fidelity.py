# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: layer_norm / rms_norm compile and PCC per math fidelity (bf16 dest acc)."""
import sys

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
    tw = ttnn.from_torch(w.bfloat16().reshape(1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    tb = ttnn.from_torch(b.bfloat16().reshape(1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    ref_ln = torch.nn.functional.layer_norm(x, (1024,), w, b, eps=1e-6)
    ref_rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    for fid in sys.argv[1:] or ["LoFi", "HiFi2", "HiFi4"]:
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        for name, op, ref, extra in [
            ("layer_norm", ttnn.layer_norm, ref_ln, {"bias": tb}),
            ("rms_norm", ttnn.rms_norm, ref_rms, {}),
        ]:
            try:
                out = op(tx, epsilon=1e-6, weight=tw, compute_kernel_config=ckc, **extra)
                print(f"{name:10s} {fid:5s} pcc={pcc(ttnn.to_torch(out).float(), ref):.5f}", flush=True)
            except Exception as e:
                msg = "static assert LoFi" if "Math fidelity must be LoFi" in str(e) else str(e).splitlines()[0][:100]
                print(f"{name:10s} {fid:5s} FAIL: {msg}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
