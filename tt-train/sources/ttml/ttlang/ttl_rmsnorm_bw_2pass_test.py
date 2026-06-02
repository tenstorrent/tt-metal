# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare TT-Lang 2-pass RMSNorm backward (``ttl_rmsnorm_bw_2pass``) to a PyTorch reference.
"""
from __future__ import annotations

import torch
import ttnn

TILE = 32

import ttl_rmsnorm_bw_2pass as ttl_mod  # noqa: E402


def _torch_ref(x, gamma_1d, eps, dL):
    x = x.float().clone().requires_grad_(True)
    g = gamma_1d.float().clone().requires_grad_(True)
    dL = dL.float()
    var = x.pow(2).mean(-1, keepdim=True)
    rms = (var + eps).sqrt()
    y = x / rms * g
    y.backward(dL)
    return x.grad.to(torch.bfloat16), g.grad.to(torch.bfloat16), rms.detach().to(torch.bfloat16)


def _run_case(
    device: ttnn.Device,
    rows: int,
    cols: int,
    label: str,
    rtol: float,
    atol: float,
) -> None:
    eps = 1e-5 if cols // TILE > 1 else 0.0078125
    torch.manual_seed(0)
    x = torch.randn(rows, cols, dtype=torch.bfloat16)
    g1d = torch.randn(cols, dtype=torch.bfloat16)
    with torch.no_grad():
        var = x.float().pow(2).mean(-1, keepdim=True)
        rms = (var + eps).sqrt()
        y = x.float() / rms * g1d.float()
    dL = ((2.0 / y.numel()) * y).to(torch.bfloat16)

    dL_dx_ref, dL_dg_ref, _ = _torch_ref(x, g1d, eps, dL)
    # Match run_rmsnorm_bw_2pass: ttnn.sum(..., dim=[0], keepdim=True) -> [1, C].
    dL_dg_ref = dL_dg_ref.unsqueeze(0)

    rows_p = -(-rows // TILE) * TILE
    cols_p = -(-cols // TILE) * TILE
    x_p = ttl_mod.pad(x, rows_p, cols_p)
    g_p = ttl_mod.pad(g1d.unsqueeze(0), 1, cols_p)
    rms_p = ttl_mod_pad(rms.to(torch.bfloat16), rows_p, 1)
    dL_p = ttl_mod.pad(dL, rows_p, cols_p)

    k = ttl_mod.make_kernel()
    out_da = ttl_mod.to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), device)
    out_dg = ttl_mod.to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), device)
    out_da, dg_reduced_ttnn = ttl_mod.run_rmsnorm_bw_2pass(
        device,
        k,
        ttl_mod.to_dev(x_p, device),
        ttl_mod.to_dev(g_p, device),
        ttl_mod.to_dev(rms_p, device),
        ttl_mod.to_dev(dL_p, device),
        out_da,
        out_dg,
    )
    ttnn.synchronize_device(device)

    da = ttnn.to_torch(out_da)[:rows, :cols]
    dg_reduced = ttnn.to_torch(dg_reduced_ttnn)[:, :cols]

    torch.testing.assert_close(da, dL_dx_ref, rtol=rtol, atol=atol)
    print(f"OK {label}: dL/dx matches torch ref (rtol={rtol}, atol={atol}).")
    torch.testing.assert_close(dg_reduced, dL_dg_ref, rtol=rtol, atol=atol)
    print(f"OK {label}: dL/dgamma matches torch ref (rtol={rtol}, atol={atol}).")


def main() -> None:
    # Taken from tests/ops/rmsnorm_op_test.cpp
    rtol, atol = 1.0e-3, 2.0e-3
    device = ttnn.open_device(device_id=0)

    try:
        _run_case(device, 64, 384, "64x384", rtol, atol)
        _run_case(device, 4096, 4096, "4096x4096", rtol, atol)
        _run_case(device, 8192, 8192, "8192x8192", rtol, atol)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
