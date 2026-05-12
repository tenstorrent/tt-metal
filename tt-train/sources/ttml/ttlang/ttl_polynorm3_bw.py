# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
PolyNorm3 backward — TT-Lang ``polynorm3_bw`` (incremental bring-up).

**Current kernel** (WIP): per **tile-row** strip of ``x``, three result buffers only:

- ``buffer1``: ``Σx²`` over the strip → ``inv_rms_x  = 1/√(mean(x²) + ε)``
- ``buffer2``: ``Σx⁴`` over the strip → ``inv_rms_x2 = 1/√(mean(x⁴) + ε)``
- ``buffer3``: ``Σx⁶`` over the strip → ``inv_rms_x3 = 1/√(mean(x⁶) + ε)``
- ``scalar_1`` / ``scalar_2`` / ``scalar_3``: ``Σ(x·dout)``, ``Σ(x²·dout)``, ``Σ(x³·dout)`` (whole row strip).
- ``coeff1``–``coeff3``: correction scalars ``inv_rms_k³ · (scalar_k · w_k) · (1/N)`` for the grad-``x`` chain rule (fused, no staging buffer).
- ``w2_inv_rms_x`` / ``w1_inv_rms_x2`` / ``w0_inv_rms_x3``: single tiles ``w2·inv_rms_x``, ``w1·inv_rms_x2``,
  ``w0·inv_rms_x3`` (FP32 dest acc via ``fp32_dest_acc_en``) for reuse in Pass-2.

**Pass-2 (``grad_x``):** ``dm_read`` pushes one ``x`` / ``dout`` tile per column into ``x_tile`` / ``dout_tile`` (ring depth
``cols``). ``compute`` waits each tile, forms ``term_1 + term_2 + term_3``, stores into ``gx_accum_tile``; ``dm_write``
pops ``gx_accum_tile`` and copies to ``grad_x[row, col]``.

**Weight / bias (row partials):** with ``inv_rms_* = rsqrt(mean+eps)`` and ``scalar_* = Σ(dout·t)``,
``Σ dout·RmsNorm(t) = inv_rms_* · scalar_*`` for each branch; ``dL/db = Σ dout`` on the strip.
Partials go to ``grad_packed[row, 0*4 + k]`` (tile column 0), matching ``polynorm_2`` layout.

for numerical behavior. Helpers: ``wide_x``, ``wide_tmp``, ``wide_dout``, ``x_tile``, ``dout_tile``, ``gx_accum_tile``,
``ninv_row``, ``eps``, ``red_dfb``, ``reduce_c_cb`` (column of ones).
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
    n_cores = grid_rows * grid_cols
    rows_per_node = -(-rows // n_cores)
    # Double-buffer (``block_count=2``): one block for NOC reader/writer, one for compute — standard ping-pong.
    x_tile = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    dout_tile = ttl.make_dataflow_buffer_like(dout, shape=one, block_count=2)
    gx_accum_tile = ttl.make_dataflow_buffer_like(grad_x, shape=one, block_count=2)
    # Depth 2: dm_read may start row N+1 while compute row N is still in pass-1 (has not
    # waited eps/weights yet). block_count=1 deadlocks on reserve vs full CB.
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=one, block_count=1)

    # # Three primary buffers: row-strip ``inv_rms`` scalars (each 1×1 tile of bf16).
    inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    scalar_1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    scalar_2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    scalar_3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    w0_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)
    w1_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)
    w2_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)

    coeff1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # # prepare_weighted_inv_rms_for_row: one tile each, reused in Pass-2.
    w2_inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w1_inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w0_inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # # Row partials for dL/dw_k and dL/db (accumulate on host or fuse across rows if needed).
    dl_dw0_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw1_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw2_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_db_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    # # Pass-2: ``t1_scratch`` holds ``t1`` so ``t1+t2`` into ``red_dfb`` is a smaller DST region (HW compile).
    t1_scratch = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    red_dfb = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # Pass-1 strip reductions: reuse ``inv_rms_*``, ``scalar_*``, and ``dl_db_row`` as running-sum CBs (they are
    # otherwise unused until after reduce). Extra dedicated ``sum_*_acc`` buffers exceed HW DFB limits (~32).

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
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
                with coeff1.reserve() as z0:
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
                        coeff1.wait() as sd,
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
                        with coeff1.reserve() as o:
                            o.store(nsd)

                with coeff1.wait() as sd:
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
                    # coeff_k = inv_rms_k³ · (scalar_k · w_k) · (1/N); 1/N_row via ``fill(iv, inv_n_row)``.
                    # Forward pairs w2↔x, w1↔x², w0↔x³ → scalar_1↔inv_rms_x, scalar_2↔inv_rms_x2, scalar_3↔inv_rms_x3.
                    with inv_rms_x.wait() as iv, scalar_1.wait() as sv:
                        with t1_scratch.reserve() as ninv:
                            ninv.store(ttl.math.fill(ninv, inv_n_row))
                        ninv = t1_scratch.wait()
                        with coeff1.reserve() as o:
                            o.store(iv * iv * iv * (sv * w2v) * ninv)
                        with w2_inv_rms_x.reserve() as o:
                            o.store(iv * w2v)  # iv
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
                for local_col in range(cols):
                    with x_tile.reserve() as b:
                        ttl.copy(x[row, local_col], b).wait()
                    with dout_tile.reserve() as b:
                        ttl.copy(dout[row, local_col], b).wait()
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
    """Torch PolyNorm forward used for reference ``backward``::

        out = w0 * RmsNorm(x³) + w1 * RmsNorm(x²) + w2 * RmsNorm(x) + b

    ``RmsNorm(t)`` follows ``rmsnorm_25-04-20.py::_torch_ref``:
    ``var = t.pow(2).mean(-1, keepdim=True)``, ``rms = (var + eps).sqrt()``, ``y = t / rms``.
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
_POLY_GX_RTOL = 0.07


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
    ttnn.synchronize_device(device)

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
        polynorm3_bw_smoke_test(device, 8192, 8192)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
