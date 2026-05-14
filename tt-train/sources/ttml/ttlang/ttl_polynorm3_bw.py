# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
PolyNorm3 backward — TT-Lang
"""
from __future__ import annotations

import torch
import ttnn
import ttl

TILE = 32


@ttl.operation(grid="auto", fp32_dest_acc_en=True)
def polynorm3_bw(x, dout, weight_strip, eps_tile, grad_x, grad_packed):
    rows = x.shape[0] // TILE
    cols = x.shape[1] // TILE
    one = (1, 1)
    n_row_elems = x.shape[1]
    inv_n_row = 1.0 / float(n_row_elems)

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    # How many cores in the grid
    n_cores = grid_rows * grid_cols

    rows_per_node = -(-rows // n_cores)
    # Input buffers
    x_tile = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    dout_tile = ttl.make_dataflow_buffer_like(dout, shape=one, block_count=2)
    gx_accum_tile = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=one, block_count=1)

    # Pass-1 strip accumulators
    inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    scalar_1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    scalar_2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    scalar_3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    # Weights
    w0_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)
    w1_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)
    w2_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)

    # coeff_k = inv_rms_k^3 * (scalar_k * w_k) * (1/N)
    coeff1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # wk * inv_rms_k
    w2_inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w1_inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w0_inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # Output buffers
    dl_dw0_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw1_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw2_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_db_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    # Helper buffers for the partial computation
    t1_scratch = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    red_dfb = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        # Pass-1 strip accumulations
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                with inv_rms_x.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with inv_rms_x2.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with inv_rms_x3.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with scalar_1.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with scalar_2.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with scalar_3.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with red_dfb.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))

                for _local_col in range(cols):
                    with (
                        # inputs
                        x_tile.wait() as xv,
                        dout_tile.wait() as dv,
                        # intermediate results
                        inv_rms_x.wait() as s2,
                        inv_rms_x2.wait() as s4,
                        inv_rms_x3.wait() as s6,
                        scalar_1.wait() as sxd,
                        scalar_2.wait() as sx2d,
                        scalar_3.wait() as sx3d,
                        red_dfb.wait() as sd,
                    ):
                        ninv = ttl.math.fill(xv, inv_n_row)
                        reduce_tile = ttl.math.fill(ninv, 1.0)
                        x2v = xv * xv
                        x3v = x2v * xv
                        x4v = x2v * x2v
                        x6v = x4v * x2v
                        ns2 = s2 + (x2v @ reduce_tile)
                        ns4 = s4 + (x4v @ reduce_tile)
                        ns6 = s6 + (x6v @ reduce_tile)
                        nsxd = sxd + ((xv * dv) @ reduce_tile)
                        nsx2d = sx2d + ((x2v * dv) @ reduce_tile)
                        nsx3d = sx3d + ((x3v * dv) @ reduce_tile)
                        nsd = sd + (dv @ reduce_tile)
                        with inv_rms_x.reserve() as o:
                            o.store(ns2)
                        with inv_rms_x2.reserve() as o:
                            o.store(ns4)
                        with inv_rms_x3.reserve() as o:
                            o.store(ns6)
                        with scalar_1.reserve() as o:
                            o.store(nsxd)
                        with scalar_2.reserve() as o:
                            o.store(nsx2d)
                        with scalar_3.reserve() as o:
                            o.store(nsx3d)
                        with red_dfb.reserve() as o:
                            o.store(nsd)

                with red_dfb.wait() as sd:
                    with dl_db_row.reserve() as o:
                        o.store(sd)

                with (
                    inv_rms_x.wait() as sum_x2,
                    inv_rms_x2.wait() as sum_x4,
                    inv_rms_x3.wait() as sum_x6,
                    eps_dfb.wait() as epsv,
                ):
                    with inv_rms_x.reserve() as o:
                        o.store(ttl.math.rsqrt(sum_x2 * ttl.math.fill(sum_x2, inv_n_row) + epsv))
                    with inv_rms_x2.reserve() as o:
                        o.store(ttl.math.rsqrt(sum_x4 * ttl.math.fill(sum_x4, inv_n_row) + epsv))
                    with inv_rms_x3.reserve() as o:
                        o.store(ttl.math.rsqrt(sum_x6 * ttl.math.fill(sum_x6, inv_n_row) + epsv))

                with w0_dfb.wait() as w0v, w1_dfb.wait() as w1v, w2_dfb.wait() as w2v:
                    # 1/N_row via fill(iv, inv_n_row) stored in ninv block.
                    with inv_rms_x.wait() as iv, scalar_1.wait() as sv:
                        with t1_scratch.reserve() as ninv:
                            ninv.store(ttl.math.fill(ninv, inv_n_row))
                        ninv = t1_scratch.wait()
                        with coeff1.reserve() as o:
                            o.store(iv * iv * iv * (sv * w2v) * ninv)
                        with w2_inv_rms_x.reserve() as o:
                            o.store(iv * w2v)
                        with dl_dw2_row.reserve() as o:
                            o.store(iv * sv)
                    with inv_rms_x2.wait() as iv, scalar_2.wait() as sv:
                        with t1_scratch.reserve() as ninv:
                            ninv.store(ttl.math.fill(ninv, inv_n_row))
                        ninv = t1_scratch.wait()
                        with coeff2.reserve() as o:
                            o.store(iv * iv * iv * (sv * w1v) * ninv)
                        with w1_inv_rms_x2.reserve() as o:
                            o.store(iv * w1v)
                        with dl_dw1_row.reserve() as o:
                            o.store(iv * sv)
                    with inv_rms_x3.wait() as iv, scalar_3.wait() as sv:
                        with t1_scratch.reserve() as ninv:
                            ninv.store(ttl.math.fill(ninv, inv_n_row))
                        ninv = t1_scratch.wait()
                        with coeff3.reserve() as o:
                            o.store(iv * iv * iv * (sv * w0v) * ninv)
                        with w0_inv_rms_x3.reserve() as o:
                            o.store(iv * w0v)
                        with dl_dw0_row.reserve() as o:
                            o.store(iv * sv)

                c1 = coeff1.wait()
                c2 = coeff2.wait()
                c3 = coeff3.wait()
                w0t = w0_inv_rms_x3.wait()
                w1t = w1_inv_rms_x2.wait()
                w2t = w2_inv_rms_x.wait()

                # Pass-2 - grad_x computation
                for _local_col in range(cols):
                    xv = x_tile.wait()
                    dv = dout_tile.wait()
                    x2v = xv * xv
                    x3v = x2v * xv
                    t1 = dv * w2t - xv * c1
                    with t1_scratch.reserve() as o:
                        o.store(t1)
                    t2 = (dv * w1t - x2v * c2) * (xv + xv)
                    with t1_scratch.wait() as acc, red_dfb.reserve() as o:
                        o.store(acc + t2)
                    t3 = (dv * w0t - x3v * c3) * (x2v + x2v + x2v)
                    with red_dfb.wait() as acc, gx_accum_tile.reserve() as o:
                        o.store(acc + t3)

    @ttl.datamovement()
    def dm_read():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                # Pass-1
                for local_col in range(cols):
                    with x_tile.reserve() as b:
                        ttl.copy(x[row, local_col], b).wait()
                    with dout_tile.reserve() as b:
                        ttl.copy(dout[row, local_col], b).wait()

                # Pass-2
                with eps_dfb.reserve() as b:
                    ttl.copy(eps_tile[0, 0], b).wait()
                with w0_dfb.reserve() as b:
                    ttl.copy(weight_strip[0, 2], b).wait()
                with w1_dfb.reserve() as b:
                    ttl.copy(weight_strip[0, 1], b).wait()
                with w2_dfb.reserve() as b:
                    ttl.copy(weight_strip[0, 0], b).wait()

                for local_col in range(cols):
                    with x_tile.reserve() as b:
                        ttl.copy(x[row, local_col], b).wait()
                    with dout_tile.reserve() as b:
                        ttl.copy(dout[row, local_col], b).wait()

    @ttl.datamovement()
    def dm_write():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                for local_col in range(cols):
                    with gx_accum_tile.wait() as b:
                        ttl.copy(b, grad_x[row, local_col]).wait()
                with dl_db_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 3]).wait()
                with dl_dw0_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 0]).wait()
                with dl_dw1_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 1]).wait()
                with dl_dw2_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 2]).wait()


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


def _to_dev_f32(dev, t: torch.Tensor):
    return ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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

    x_tt = _to_dev_f32(device, x_t.float())
    dout_tt = _to_dev_f32(device, dout_t.float())
    w_tt = _to_dev_f32(device, wstrip.float())
    ep_tt = _to_dev_f32(device, eps_t.float())

    gx_tt = _to_dev_f32(device, torch.zeros(height, width, dtype=torch.float32))
    gp_tt = _to_dev_f32(device, torch.zeros(height, 4 * TILE, dtype=torch.float32))

    polynorm3_bw(x_tt, dout_tt, w_tt, ep_tt, gx_tt, gp_tt)
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


def main():
    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0)
    try:
        polynorm3_bw_smoke_test(device, 8192, 8192)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
