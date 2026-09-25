# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm [M, 4096] bf16 device-time variants (compute config), with accuracy vs torch."""

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


@pytest.mark.parametrize("M", [640, 2048])
def test_rmsnorm_variants(device, M):
    torch.manual_seed(0)
    x = torch.randn(1, 1, M, 4096) * 0.05
    w = torch.rand(4096) + 0.5
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tw = ttnn.from_torch(w.reshape(1, 1, -1, 32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    arch = device.arch()
    variants = {
        "hifi4_fp32": ttnn.init_device_compute_kernel_config(arch, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        "hifi2_fp32": ttnn.init_device_compute_kernel_config(arch, math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True),
        "hifi4_bf16": ttnn.init_device_compute_kernel_config(arch, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
        "lofi_fp32": ttnn.init_device_compute_kernel_config(arch, math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
    }
    for name, ckc in variants.items():
        for it in range(3):
            if it:
                signpost(f"rms_M{M}_{name}_start")
            o = ttnn.rms_norm(tx, epsilon=1e-6, weight=tw, compute_kernel_config=ckc)
            ttnn.synchronize_device(device)
            if it:
                signpost(f"rms_M{M}_{name}_end")
        got = ttnn.to_torch(o).float()
        pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
        print(f"RMS M{M} {name} pcc={pcc:.6f} maxerr={(ref - got).abs().max().item():.4f}")
        o.deallocate(True)
