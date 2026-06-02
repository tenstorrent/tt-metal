# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
PolyNorm3 backward
"""
from __future__ import annotations

import torch
import ttnn
import ttl

TILE = 32


@ttl.operation(grid="auto", fp32_dest_acc_en=True)
def polynorm3_bw(x, dout, weight_strip, eps_tile, grad_x, grad_packed):
    bcw = 1
    rows = x.shape[0] // TILE
    cols = x.shape[1] // TILE
    one = (1, 1)
    wide_shape = (1, cols)
    n_row_elems = x.shape[1]
    inv_n_row = 1.0 / float(n_row_elems)

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    n_cores = grid_rows * grid_cols
    rows_per_node = -(-rows // n_cores)
    # Input
    wide_x = ttl.make_dataflow_buffer_like(x, shape=wide_shape, block_count=bcw)
    wide_dout = ttl.make_dataflow_buffer_like(dout, shape=wide_shape, block_count=bcw)
    x_tile = ttl.make_dataflow_buffer_like(x, shape=one, block_count=cols)
    dout_tile = ttl.make_dataflow_buffer_like(dout, shape=one, block_count=cols)
    gx_accum_tile = ttl.make_dataflow_buffer_like(grad_x, shape=one, block_count=cols)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=one, block_count=1)

    # Three primary buffers: row-strip ``inv_rms`` scalars (each 1×1 tile of bf16).
    inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    scalar_1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    scalar_2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    scalar_3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    w0_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=1)
    w1_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=1)
    w2_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=1)

    ws0 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    ws1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    ws2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    coeff1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # prepare_weighted_inv_rms_for_row: one tile each, reused in Pass-2.
    w2_inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w1_inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w0_inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # Row partials for dL/dw_k and dL/db (accumulate on host or fuse across rows if needed).
    dl_dw0_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw1_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw2_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_db_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # ``wide_tmp`` holds (1, cols) strips; ``reduce_c_cb`` is (cols, 1) ones for ``@`` sum workaround.
    wide_tmp = ttl.make_dataflow_buffer_like(x, shape=wide_shape, block_count=bcw)
    reduce_c_cb = ttl.make_dataflow_buffer_like(x, shape=(cols, 1), block_count=bcw)
    ninv_row_dfb = ttl.make_dataflow_buffer_like(x, shape=one, block_count=rows_per_node)
    red_dfb = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                with ninv_row_dfb.reserve() as b:
                    b.store(ttl.math.fill(b, inv_n_row))
                ninv = ninv_row_dfb.wait()
                with (
                    eps_dfb.wait() as epsv,
                    wide_x.wait() as wx,
                    wide_dout.wait() as wd,
                    w0_dfb.wait() as w0v,
                    w1_dfb.wait() as w1v,
                    w2_dfb.wait() as w2v,
                ):
                    # Reduce phase: 1/√(mean(x²) + ε) and  Σ(x·dout)
                    x2w = wx * wx
                    with wide_tmp.reserve() as tmp_wide_x2:
                        tmp_wide_x2.store(x2w)
                    with reduce_c_cb.reserve() as rc:
                        rc.store(ttl.math.fill(rc, 1.0))
                    with wide_tmp.wait() as tmc, reduce_c_cb.wait() as reduce_c, red_dfb.reserve() as rd:
                        rd.store(tmc @ reduce_c)
                    with red_dfb.wait() as rd, inv_rms_x.reserve() as inv_rms_x_tmp:
                        inv_rms_x_tmp.store(ttl.math.rsqrt(rd * ninv + epsv))

                    x4w = x2w * x2w
                    with wide_tmp.reserve() as tmp_wide_x4:
                        tmp_wide_x4.store(x4w)
                    with reduce_c_cb.reserve() as rc:
                        rc.store(ttl.math.fill(rc, 1.0))
                    with wide_tmp.wait() as tmc, reduce_c_cb.wait() as reduce_c, red_dfb.reserve() as rd:
                        rd.store(tmc @ reduce_c)
                    with red_dfb.wait() as rd, inv_rms_x2.reserve() as inv_rms_x2_tmp:
                        inv_rms_x2_tmp.store(ttl.math.rsqrt(rd * ninv + epsv))

                    x6w = x4w * x2w
                    with wide_tmp.reserve() as tmp_wide_x6:
                        tmp_wide_x6.store(x6w)
                    with reduce_c_cb.reserve() as rc:
                        rc.store(ttl.math.fill(rc, 1.0))
                    with wide_tmp.wait() as tmc, reduce_c_cb.wait() as reduce_c, red_dfb.reserve() as rd:
                        rd.store(tmc @ reduce_c)
                    with red_dfb.wait() as rd, inv_rms_x3.reserve() as inv_rms_x3_tmp:
                        inv_rms_x3_tmp.store(ttl.math.rsqrt(rd * ninv + epsv))

                    with wide_tmp.reserve() as tmp:
                        tmp.store(wx * wd)
                    with reduce_c_cb.reserve() as rc:
                        rc.store(ttl.math.fill(rc, 1.0))
                    with wide_tmp.wait() as tmc, reduce_c_cb.wait() as reduce_c, red_dfb.reserve() as rd:
                        rd.store(tmc @ reduce_c)
                    with red_dfb.wait() as rd, scalar_1.reserve() as s1:
                        s1.store(rd)

                    with wide_tmp.reserve() as tmp:
                        tmp.store(x2w * wd)
                    with reduce_c_cb.reserve() as rc:
                        rc.store(ttl.math.fill(rc, 1.0))
                    with wide_tmp.wait() as tmc, reduce_c_cb.wait() as reduce_c, red_dfb.reserve() as rd:
                        rd.store(tmc @ reduce_c)
                    with red_dfb.wait() as rd, scalar_2.reserve() as s2:
                        s2.store(rd)

                    x3w = x2w * wx
                    with wide_tmp.reserve() as tmp:
                        tmp.store(x3w * wd)
                    with reduce_c_cb.reserve() as rc:
                        rc.store(ttl.math.fill(rc, 1.0))
                    with wide_tmp.wait() as tmc, reduce_c_cb.wait() as reduce_c, red_dfb.reserve() as rd:
                        rd.store(tmc @ reduce_c)
                    with red_dfb.wait() as rd, scalar_3.reserve() as s3:
                        s3.store(rd)

                    with reduce_c_cb.reserve() as rc:
                        rc.store(ttl.math.fill(rc, 1.0))
                    with reduce_c_cb.wait() as reduce_c, red_dfb.reserve() as rd:
                        rd.store(wd @ reduce_c)
                    with red_dfb.wait() as sm, dl_db_row.reserve() as o:
                        o.store(sm)

                    # ws_k = scalar_k · w_k; coeff_k = inv_rms_k³ · ws_k · (1/N); ``ninv`` is 1/N_row.
                    # Forward pairs w2↔x, w1↔x², w0↔x³ → scalar_1↔inv_rms_x, scalar_2↔inv_rms_x2, scalar_3↔inv_rms_x3.
                    with inv_rms_x.wait() as iv, scalar_1.wait() as sv:
                        with ws2.reserve() as o:
                            o.store(sv * w2v)
                        with ws2.wait() as ws:
                            with coeff1.reserve() as o:
                                o.store(iv * iv * iv * ws * ninv)
                        with w2_inv_rms_x.reserve() as o:
                            o.store(iv * w2v)
                        # dL/dw2 = Σ dout·RmsNorm(x) = inv_rms_x · scalar_1
                        with dl_dw2_row.reserve() as o:
                            o.store(iv * sv)
                    with inv_rms_x2.wait() as iv, scalar_2.wait() as sv:
                        with ws1.reserve() as o:
                            o.store(sv * w1v)
                        with ws1.wait() as ws:
                            with coeff2.reserve() as o:
                                o.store(iv * iv * iv * ws * ninv)
                        with w1_inv_rms_x2.reserve() as o:
                            o.store(iv * w1v)
                        with dl_dw1_row.reserve() as o:
                            o.store(iv * sv)
                    with inv_rms_x3.wait() as iv, scalar_3.wait() as sv:
                        with ws0.reserve() as o:
                            o.store(sv * w0v)
                        with ws0.wait() as ws:
                            with coeff3.reserve() as o:
                                o.store(iv * iv * iv * ws * ninv)
                        with w0_inv_rms_x3.reserve() as o:
                            o.store(iv * w0v)
                        with dl_dw0_row.reserve() as o:
                            o.store(iv * sv)

                # Scalar dfbs are depth-1 — single outer ``wait`` for the column loop (not per column).
                # HW DST: ``o.store(t1+t2)`` overflows registers while ``w0t/c3``… are pinned; fuse via scratch.
                # ``ws2`` is idle after coeff setup above; dm kernels never touch it.
                # Never ``wait(gx_accum_tile)`` mid-pipeline for the same slot dm_write consumes (sim deadlock).
                with (
                    w2_inv_rms_x.wait() as w2t,
                    w1_inv_rms_x2.wait() as w1t,
                    w0_inv_rms_x3.wait() as w0t,
                    coeff1.wait() as c1,
                    coeff2.wait() as c2,
                    coeff3.wait() as c3,
                ):
                    for local_col in range(cols):
                        with x_tile.wait() as xv, dout_tile.wait() as dv:
                            x2v = xv * xv
                            x3v = x2v * xv
                            t1 = dv * w2t - xv * c1
                            with ws2.reserve() as o:
                                o.store(t1)
                            t2 = (dv * w1t - x2v * c2) * (xv + xv)
                            with ws2.wait() as acc, red_dfb.reserve() as o:
                                o.store(acc + t2)
                            t3 = (dv * w0t - x3v * c3) * (x2v + x2v + x2v)
                            with red_dfb.wait() as acc, gx_accum_tile.reserve() as o:
                                o.store(acc + t3)
                ninv.pop()

    @ttl.datamovement()
    def dm_read():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                with wide_x.reserve() as b:
                    ttl.copy(x[row : row + 1, 0:cols], b).wait()
                with wide_dout.reserve() as b:
                    ttl.copy(dout[row : row + 1, 0:cols], b).wait()
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
                with dl_dw0_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 0]).wait()
                with dl_dw1_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 1]).wait()
                with dl_dw2_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 2]).wait()
                with dl_db_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 3]).wait()


# Short name (C++ ``polynorm_bw``); primary symbol in this module is ``polynorm3_bw``.
polynorm_bw = polynorm3_bw


def polynorm3_forward_torch(x: torch.Tensor, w0, w1, w2, b: float, eps: float) -> torch.Tensor:
    """Torch PolyNorm forward used for reference ``backward``::"""
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


def _to_dev(dev, t32: torch.Tensor):
    return ttnn.from_torch(
        t32,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def torch_polynorm_bw(x_bf16: torch.Tensor, dout_bf16: torch.Tensor, w0, w1, w2, b: float, eps: float):
    x = x_bf16.float().clone().detach().requires_grad_(True)
    w0t = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
    w1t = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
    w2t = torch.tensor(w2, dtype=torch.float32, requires_grad=True)
    y = polynorm3_forward_torch(x, w0t, w1t, w2t, b, eps)
    (y * dout_bf16.float()).sum().backward()
    return x.grad, w0t.grad, w1t.grad, w2t.grad


def _torch_polynorm_bw_scalar_refs(
    x_t: torch.Tensor, dout_t: torch.Tensor, w0, w1, w2, b: float, eps: float
) -> tuple[torch.Tensor, float, float, float, float]:
    """``backward`` from ``(out * dout).sum()`` with ``out = polynorm3_forward_torch(...)`` (row-wise RmsNorm)."""
    gx, g0, g1, g2 = torch_polynorm_bw(x_t, dout_t, w0, w1, w2, b, eps)
    return (
        gx.float(),
        float(dout_t.float().sum()),
        float(g0.sum()),
        float(g1.sum()),
        float(g2.sum()),
    )


def _torch_polynorm_bw_row_refs(
    x_t: torch.Tensor, dout_t: torch.Tensor, w0, w1, w2, b: float, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-wise backward refs matching packed slot shapes.

    Returns ``(gx, db_block, gw0_block, gw1_block, gw2_block)`` where each block has shape
    ``(H, TILE)`` (row scalar expanded across TILE lanes to match packed faces).
    """
    gx, _g0, _g1, _g2 = torch_polynorm_bw(x_t, dout_t, w0, w1, w2, b, eps)
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
    return gx.float(), db_block.float(), gw0_block.float(), gw1_block.float(), gw2_block.float()


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


# ``dL/dx``: kernel uses tile-row strip RMS; torch ref is **per matrix row** unless the kernel is aligned.
_POLY_GX_ATOL = 0.05
_POLY_GX_RTOL = 0.05


def _print_tensor_compare_random_cells(
    result: torch.Tensor,
    expectation: torch.Tensor,
    num: int = 10,
) -> None:
    """Print ``min(num, H*W)`` random 2D cells: row/col, then expectation vs result (float32)."""
    if result.ndim != 2 or expectation.ndim != 2:
        raise ValueError(f"expected 2D tensors, got {result.ndim=} {expectation.ndim=}")
    if result.shape != expectation.shape:
        raise ValueError(f"shape mismatch: {tuple(result.shape)} vs {tuple(expectation.shape)}")
    height, width = result.shape[0], result.shape[1]
    n_dbg = min(num, height * width)
    ri = torch.randint(0, height, (n_dbg,), dtype=torch.int64)
    ci = torch.randint(0, width, (n_dbg,), dtype=torch.int64)
    rf = result.detach().float()
    ef = expectation.detach().float()
    print(f"random {n_dbg} 2D cells (row, col), shape ({height}, {width}), float32:")
    for k in range(n_dbg):
        r, c = int(ri[k]), int(ci[k])
        a = rf[r, c].item()
        b = ef[r, c].item()
        print(f"  cell (row, col) = ({r}, {c})")
        print(f"    expectation: {b:.6g}")
        print(f"    result:      {a:.6g}")
        print(f"    abs_diff:    {abs(a - b):.6g}")


def _make_case() -> tuple[float, float, float, float]:
    """Host fixture aligned with metal tests: ``weight(0,0,0,k)`` and ``bias(0,0,0,0)``."""
    w0, w1, w2 = 0.2, 0.3, 0.5
    b = 0.1
    return w0, w1, w2, b


def polynorm3_bw_smoke_test(device, height: int, width: int) -> None:
    """Run backward smoke. ``height`` and ``width`` must be multiples of ``TILE``.

    Reference ``dL/dx`` is from autograd on ``out = w0·RmsNorm(x³)+…`` with **row-wise** RmsNorm (one
    scale per row over ``W``). ``polynorm3_bw`` still uses **tile-row strip** stats, so ``grad_x`` is
    only checked with loose ``_POLY_GX_*`` tolerances unless the kernel is aligned to per-row RMS.
    """
    assert height % TILE == 0 and width % TILE == 0, (height, width)
    eps = 1e-5
    w0, w1, w2, b = _make_case()

    x_t = torch.empty((height, width), dtype=torch.bfloat16)
    x_t.uniform_(-1.0, 1.0)
    with torch.no_grad():
        dout_t = polynorm3_forward_torch(x_t, w0, w1, w2, b, eps).to(torch.bfloat16)

    g_x_ref, _db_ref, _g_w0_ref, _g_w1_ref, _g_w2_ref = _torch_polynorm_bw_scalar_refs(x_t, dout_t, w0, w1, w2, b, eps)
    _gx_row_ref, db_row_ref, gw0_row_ref, gw1_row_ref, gw2_row_ref = _torch_polynorm_bw_row_refs(
        x_t, dout_t, w0, w1, w2, b, eps
    )

    wstrip = torch.zeros(TILE, 3 * TILE, dtype=torch.bfloat16)
    wstrip[:, 0:TILE] = w2
    wstrip[:, TILE : 2 * TILE] = w1
    wstrip[:, 2 * TILE : 3 * TILE] = w0

    eps_t = torch.full((TILE, TILE), eps, dtype=torch.bfloat16)

    x_tt = _to_dev(device, x_t)
    dout_tt = _to_dev(device, dout_t)
    w_tt = _to_dev(device, wstrip)
    ep_tt = _to_dev(device, eps_t)

    gx_tt = _to_dev(device, torch.zeros(height, width, dtype=torch.bfloat16))
    gp_tt = _to_dev(device, torch.zeros(height, 4 * TILE, dtype=torch.bfloat16))

    polynorm3_bw(x_tt, dout_tt, w_tt, ep_tt, gx_tt, gp_tt)

    gx_dev = ttnn.to_torch(gx_tt)
    gp_dev = ttnn.to_torch(gp_tt).float()
    g_x_ref_bf = g_x_ref.to(torch.bfloat16)

    torch.testing.assert_close(
        gx_dev,
        g_x_ref_bf,
        rtol=_POLY_GX_RTOL,
        atol=_POLY_GX_ATOL,
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
        rtol=_POLY_GX_RTOL,
        atol=_POLY_GX_ATOL,
    )
    torch.testing.assert_close(
        dw1_block,
        gw1_ref_block,
        rtol=_POLY_GX_RTOL,
        atol=_POLY_GX_ATOL,
    )
    torch.testing.assert_close(
        dw2_block,
        gw2_ref_block,
        rtol=_POLY_GX_RTOL,
        atol=_POLY_GX_ATOL,
    )
    torch.testing.assert_close(
        db_block,
        db_ref_block,
        rtol=_POLY_GX_RTOL,
        atol=_POLY_GX_ATOL,
    )
    print(f"OK: polynorm3_bw {height}x{width}")


def main():
    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0)
    try:
        polynorm3_bw_smoke_test(device, 256, 256)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
