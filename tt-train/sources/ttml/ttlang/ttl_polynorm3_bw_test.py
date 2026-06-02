# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare TT-Lang PolyNorm3 backward (``ttl_polynorm3_bw``) to a PyTorch reference.
"""
from __future__ import annotations

import torch
import ttnn

TILE = 32

import ttl_polynorm3_bw  # noqa: E402


def polynorm3_forward_torch(x: torch.Tensor, w0, w1, w2, b: float, eps: float) -> torch.Tensor:
    """Torch PolyNorm3 forward used for reference ``backward``::

    out = w0 * RmsNorm(x^3) + w1 * RmsNorm(x^2) + w2 * RmsNorm(x) + b
    """
    xf = x.float()

    def rmsnorm_row(t: torch.Tensor) -> torch.Tensor:
        # One scalar sum/mean per row over W, then divide by row RMS.
        var = t.pow(2).mean(dim=-1, keepdim=True)
        rms = (var + eps).sqrt()
        return t / rms

    w0t = w0 if isinstance(w0, torch.Tensor) else torch.tensor(w0, dtype=xf.dtype, device=xf.device)
    w1t = w1 if isinstance(w1, torch.Tensor) else torch.tensor(w1, dtype=xf.dtype, device=xf.device)
    w2t = w2 if isinstance(w2, torch.Tensor) else torch.tensor(w2, dtype=xf.dtype, device=xf.device)
    bt = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=xf.dtype, device=xf.device)
    return w0t * rmsnorm_row(xf**3) + w1t * rmsnorm_row(xf**2) + w2t * rmsnorm_row(xf) + bt


def torch_polynorm_bw(x_bf16: torch.Tensor, dout_bf16: torch.Tensor, w0, w1, w2, b: float, eps: float):
    """Torch PolyNorm3 backward"""
    x = x_bf16.float().clone().detach().requires_grad_(True)
    w0t = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
    w1t = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
    w2t = torch.tensor(w2, dtype=torch.float32, requires_grad=True)
    y = polynorm3_forward_torch(x, w0t, w1t, w2t, b, eps)
    (y * dout_bf16.float()).sum().backward()
    return x.grad, w0t.grad, w1t.grad, w2t.grad


def _torch_polynorm_bw_grad_x_ref(
    x_t: torch.Tensor, dout_t: torch.Tensor, w0, w1, w2, b: float, eps: float
) -> torch.Tensor:
    """Autograd reference for ``grad_x``, shape ``(H, W)``.

    Uses ``torch_polynorm_bw`` (full backward through the PolyNorm forward). This is separate
    from the row-partial refs because ``grad_x`` is a dense per-element tensor, while weight/bias
    partials from the device are packed as one ``(H, TILE)`` face per row in ``grad_packed``.
    """
    gx, _g0, _g1, _g2 = torch_polynorm_bw(x_t, dout_t, w0, w1, w2, b, eps)
    return gx.float()


def _torch_polynorm_bw_packed_row_refs(
    x_t: torch.Tensor, dout_t: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic row-partial refs for ``grad_packed``, each block shape ``(H, TILE)``.

    Each row scalar (``dL/db``, ``dL/dw_k``) is expanded across TILE lanes to match the tile
    faces the kernel writes. Closed-form ``Σ dout·RmsNorm(t)`` per row — not taken from
    autograd weight ``.grad`` scalars, which would be global sums.
    """
    xf = x_t.float()
    df = dout_t.float()

    def rmsnorm_row(t: torch.Tensor) -> torch.Tensor:
        var = t.pow(2).mean(dim=-1, keepdim=True)
        return t / (var + eps).sqrt()

    rx = rmsnorm_row(xf)
    rx2 = rmsnorm_row(xf * xf)
    x3 = xf * xf * xf
    rx3 = rmsnorm_row(x3)

    db_block = df.sum(dim=-1, keepdim=True).expand(-1, TILE)
    gw2_block = (rx * df).sum(dim=-1, keepdim=True).expand(-1, TILE)
    gw1_block = (rx2 * df).sum(dim=-1, keepdim=True).expand(-1, TILE)
    gw0_block = (rx3 * df).sum(dim=-1, keepdim=True).expand(-1, TILE)
    return db_block.float(), gw0_block.float(), gw1_block.float(), gw2_block.float()


def _decode_slot_from_packed(gp_dev: torch.Tensor, height: int, slot: int) -> torch.Tensor:
    """Return decoded packed slot block from ``(height, 4 * TILE)`` host tensor.

    ``slot`` must be in ``{0,1,2,3}``; returned tensor shape is ``(height, TILE)``.
    """
    if gp_dev.shape[0] != height or gp_dev.shape[1] != 4 * TILE:
        raise ValueError(f"expected gp_dev shape ({height}, {4 * TILE}), got {tuple(gp_dev.shape)}")
    if slot < 0 or slot > 3:
        raise ValueError(f"slot must be in [0, 3], got {slot}")
    # Four contiguous blocks (dw0, dw1, dw2, db), each of shape (height, TILE).
    return gp_dev[:, slot * TILE : (slot + 1) * TILE].float()


def _make_case() -> tuple[float, float, float, float]:
    """Host fixture aligned with c++ PolyNorm3 tests."""
    w0, w1, w2 = 0.2, 0.3, 0.5
    b = 0.1
    return w0, w1, w2, b


def polynorm3_bw_smoke_test(device, height: int, width: int) -> None:
    """Smoke test for the polynorm3 backward pass"""
    assert height % TILE == 0 and width % TILE == 0, (height, width)
    eps = 1e-5
    w0, w1, w2, b = _make_case()
    x_t = torch.empty((height, width), dtype=torch.bfloat16)
    x_t.uniform_(-1.0, 1.0)

    # Old tolerances from c++ PolyNorm3 tests, current are (5e-3, 5e-3)
    POLY_ATOL = 8e-2
    POLY_RTOL = 8e-2

    with torch.no_grad():
        dout_t = polynorm3_forward_torch(x_t, w0, w1, w2, b, eps).to(torch.bfloat16)

    g_x_ref = _torch_polynorm_bw_grad_x_ref(x_t, dout_t, w0, w1, w2, b, eps)
    db_row_ref, gw0_row_ref, gw1_row_ref, gw2_row_ref = _torch_polynorm_bw_packed_row_refs(x_t, dout_t, eps)

    wstrip = torch.zeros(TILE, 3 * TILE, dtype=torch.bfloat16)
    wstrip[:, 0:TILE] = w2
    wstrip[:, TILE : 2 * TILE] = w1
    wstrip[:, 2 * TILE : 3 * TILE] = w0

    eps_t = torch.full((TILE, TILE), eps, dtype=torch.bfloat16)

    x_tt = ttl_polynorm3_bw._to_dev_f32(device, x_t.float())
    dout_tt = ttl_polynorm3_bw._to_dev_f32(device, dout_t.float())
    w_tt = ttl_polynorm3_bw._to_dev_f32(device, wstrip.float())
    ep_tt = ttl_polynorm3_bw._to_dev_f32(device, eps_t.float())

    gx_tt = ttl_polynorm3_bw._to_dev_f32(device, torch.zeros(height, width, dtype=torch.float32))
    gp_tt = ttl_polynorm3_bw._to_dev_f32(device, torch.zeros(height, 4 * TILE, dtype=torch.float32))

    ttl_polynorm3_bw.polynorm3_bw(x_tt, dout_tt, w_tt, ep_tt, gx_tt, gp_tt)
    ttnn.synchronize_device(device)

    gx_dev = ttnn.to_torch(gx_tt).to(torch.bfloat16)
    gp_dev = ttnn.to_torch(gp_tt).float()
    g_x_ref_bf = g_x_ref.to(torch.bfloat16)

    torch.testing.assert_close(
        gx_dev,
        g_x_ref_bf,
        rtol=POLY_RTOL,
        atol=POLY_ATOL,
    )
    dw0_block = _decode_slot_from_packed(gp_dev, height, slot=0)
    dw1_block = _decode_slot_from_packed(gp_dev, height, slot=1)
    dw2_block = _decode_slot_from_packed(gp_dev, height, slot=2)
    db_block = _decode_slot_from_packed(gp_dev, height, slot=3)

    gw0_ref_block = gw0_row_ref.to(torch.float32)
    gw1_ref_block = gw1_row_ref.to(torch.float32)
    gw2_ref_block = gw2_row_ref.to(torch.float32)
    db_ref_block = db_row_ref.to(torch.float32)

    torch.testing.assert_close(
        dw0_block,
        gw0_ref_block,
        rtol=POLY_RTOL,
        atol=POLY_ATOL,
    )
    torch.testing.assert_close(
        dw1_block,
        gw1_ref_block,
        rtol=POLY_RTOL,
        atol=POLY_ATOL,
    )
    torch.testing.assert_close(
        dw2_block,
        gw2_ref_block,
        rtol=POLY_RTOL,
        atol=POLY_ATOL,
    )
    torch.testing.assert_close(
        db_block,
        db_ref_block,
        rtol=POLY_RTOL,
        atol=POLY_ATOL,
    )
    print(f"OK: polynorm3_bw {height}x{width}")


def main() -> None:
    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0)
    try:
        polynorm3_bw_smoke_test(device, 8192, 8192)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
