# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Column-split rmsnorm backward with gather+mcast for scale all-reduce.
"""
from __future__ import annotations

import torch
import ttnn
import ttl

TILE = 32
GRID_X_MAX = 11
GRID_Y_MAX = 10


def pick_config(ht: int, wt: int, grid_x: int = GRID_X_MAX, grid_y: int = GRID_Y_MAX):
    """Max cores; pad via cpc/rpc = ceil(n / max_grid)."""
    col_cores = min(grid_x, wt)
    cpc = -(-wt // col_cores)
    wt_padded = col_cores * cpc
    row_cores = min(grid_y, ht)
    rpc = -(-ht // row_cores)
    ht_padded = row_cores * rpc
    return col_cores, row_cores, cpc, rpc, ht_padded, wt_padded


def make_kernel_for_shape(rows: int, cols: int, block_count: int = 2):
    ht = rows // TILE
    wt = cols // TILE
    col_cores, row_cores, cpc, rpc, ht_padded, wt_padded = pick_config(ht, wt)
    k = make_kernel(col_cores, row_cores, cpc, rpc, elem_c=cols, block_count=block_count)
    return k, (col_cores, row_cores, cpc, rpc, ht_padded, wt_padded)


def make_kernel(col_cores: int, row_cores: int, cpc: int, rpc: int, elem_c: int, block_count: int = 2):
    print (f"rpc={rpc}")
    kd = 26
    rb = rpc // kd if (rpc % kd == 0 and rpc >= kd) else rpc
    n_row_strip = rpc // rb
    print (f"rb={rb} n_row_strip={n_row_strip}")
    bc = max(block_count, 2)

    @ttl.operation(grid=(col_cores, row_cores))
    def rmsnorm_bw_colsplit(inp, gamma, rms, dL, dL_da_out, dL_dg_out):
        ht = inp.shape[0] // TILE
        wt = inp.shape[1] // TILE
        if wt != col_cores * cpc:
            raise ValueError(f"wt={wt} != col_cores*cpc ({col_cores}*{cpc})")
        if ht != row_cores * rpc:
            raise ValueError(f"ht={ht} != row_cores*rpc ({row_cores}*{rpc})")

        inv_c = 1.0 / elem_c

        gather_net = ttl.PipeNet([
            ttl.Pipe((x, y), (0, y))
            for x in range(1, col_cores)
            for y in range(row_cores)
        ])
        bcast_net = ttl.PipeNet([
            ttl.Pipe((0, y), (slice(1, col_cores), y))
            for y in range(row_cores)
        ])

        block_shape = (rb, cpc)
        row_shape = (rb, 1)
        scalar_shape = (1, 1)

        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=bc)
        gam_cb = ttl.make_dataflow_buffer_like(gamma, shape=block_shape, block_count=bc)
        rms_cb = ttl.make_dataflow_buffer_like(rms, shape=block_shape, block_count=bc)
        dL_cb = ttl.make_dataflow_buffer_like(dL, shape=block_shape, block_count=bc)

        # Compute-internal: fill() needs a (1,1) block target for the reduce scaler.
        scaler_cb = ttl.make_dataflow_buffer_like(inp, shape=scalar_shape, block_count=bc)
        partial_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=bc)

        recv_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=col_cores)
        sum_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=bc)
        bcast_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=bc)

        out_da_cb = ttl.make_dataflow_buffer_like(dL_da_out, shape=block_shape, block_count=bc)
        out_dg_cb = ttl.make_dataflow_buffer_like(dL_dg_out, shape=block_shape, block_count=bc)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)

            scaler_cb.reserve().store(ttl.math.fill(scaler_cb.reserve(), inv_c))
            scaler = scaler_cb.wait()

            for _ri in range(n_row_strip):
                iv = inp_cb.wait()
                recip = ttl.math.recip(rms_cb.wait())
                dLw = dL_cb.wait()
                gained = recip * gam_cb.wait() * dLw

                partial_cb.reserve().store(ttl.math.reduce_sum(iv * gained, scaler, dims=[1]))

                if core_x == 0:
                    scale = partial_cb.wait()
                    for _ in range(col_cores - 1):
                        scale = scale + recv_cb.wait()
                    sum_cb.reserve().store(scale)
                    out_da_cb.reserve().store(
                        gained - ttl.math.broadcast(scale, iv, dims=[1]) * iv * recip * recip
                    )
                    out_dg_cb.reserve().store(iv * recip * dLw)
                else:
                    out_da_cb.reserve().store(
                        gained - ttl.math.broadcast(bcast_cb.wait(), iv, dims=[1]) * iv * recip * recip
                    )
                    out_dg_cb.reserve().store(iv * recip * dLw)

        @ttl.datamovement()
        def dm_read():
            core_x, core_y = ttl.node(dims=2)
            col_start = core_x * cpc
            row_start = core_y * rpc

            for ri in range(n_row_strip):
                r0 = row_start + ri * rb
                i_blk = inp_cb.reserve()
                g_blk = gam_cb.reserve()
                r_blk = rms_cb.reserve()
                d_blk = dL_cb.reserve()
                i_tx = ttl.copy(inp[r0 : r0 + rb, col_start : col_start + cpc], i_blk)
                g_tx = ttl.copy(gamma[r0 : r0 + rb, col_start : col_start + cpc], g_blk)
                r_tx = ttl.copy(rms[r0 : r0 + rb, col_start : col_start + cpc], r_blk)
                d_tx = ttl.copy(dL[r0 : r0 + rb, col_start : col_start + cpc], d_blk)
                i_tx.wait()
                g_tx.wait()
                r_tx.wait()
                d_tx.wait()

                if core_x > 0:
                    blk = partial_cb.wait()
                    def send(pipe):
                        tx = ttl.copy(blk, pipe)
                        tx.wait()
                    gather_net.if_src(send)

                    def recv(pipe):
                        b = bcast_cb.reserve()
                        tx = ttl.copy(pipe, b)
                        tx.wait()
                    bcast_net.if_dst(recv)
                else:
                    def recv(pipe):
                        b = recv_cb.reserve()
                        tx = ttl.copy(pipe, b)
                        tx.wait()
                    gather_net.if_dst(recv)

                    blk = sum_cb.wait()
                    def send(pipe):
                        tx = ttl.copy(blk, pipe)
                        tx.wait()
                    bcast_net.if_src(send)

        @ttl.datamovement()
        def dm_write():
            core_x, core_y = ttl.node(dims=2)
            col_start = core_x * cpc
            row_start = core_y * rpc
            for ri in range(n_row_strip):
                r0 = row_start + ri * rb
                da_blk = out_da_cb.wait()
                da_tx = ttl.copy(da_blk, dL_da_out[r0 : r0 + rb, col_start : col_start + cpc])
                da_tx.wait()
                dg_blk = out_dg_cb.wait()
                dg_tx = ttl.copy(dg_blk, dL_dg_out[r0 : r0 + rb, col_start : col_start + cpc])
                dg_tx.wait()

    return rmsnorm_bw_colsplit


def _torch_ref(x, gamma_1d, eps, dL):
    x = x.float().clone().requires_grad_(True)
    g = gamma_1d.float().clone().requires_grad_(True)
    dL = dL.float()
    var = x.pow(2).mean(-1, keepdim=True)
    rms = (var + eps).sqrt()
    y = x / rms * g
    y.backward(dL)
    return x.grad.to(torch.bfloat16), g.grad.to(torch.bfloat16), rms.detach().to(torch.bfloat16)


def _to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _pad(t, rows_padded, cols_padded):
    rows, cols = t.shape
    pad_r = rows_padded - rows
    pad_c = cols_padded - cols
    if pad_r == 0 and pad_c == 0:
        return t
    return torch.nn.functional.pad(t, (0, pad_c, 0, pad_r), value=0.0)


def _run_case(device, rows, cols, label):
    eps = 1e-5 if cols // TILE > 1 else 0.0078125
    torch.manual_seed(0)
    x = torch.randn(rows, cols, dtype=torch.bfloat16)
    g1d = torch.randn(cols, dtype=torch.bfloat16)
    g2d = g1d.unsqueeze(0).expand(rows, cols).contiguous()
    with torch.no_grad():
        var = x.float().pow(2).mean(-1, keepdim=True)
        rms = (var + eps).sqrt()
        y = x.float() / rms * g2d.float()
    dL = ((2.0 / y.numel()) * y).to(torch.bfloat16)

    dL_dx_ref, dL_dg_ref, _ = _torch_ref(x, g1d, eps, dL)
    rms_2d = rms.to(torch.bfloat16).expand(-1, cols).contiguous()

    k, cfg = make_kernel_for_shape(rows, cols)
    cc, rc, cpc, rpc, ht_p, wt_p = cfg
    rb = rpc // 2 if (rpc % 2 == 0 and rpc >= 2) else rpc
    rows_p, cols_p = ht_p * TILE, wt_p * TILE
    x_p = _pad(x, rows_p, cols_p)
    g_p = _pad(g2d, rows_p, cols_p)
    rms_p = _pad(rms_2d, rows_p, cols_p)
    dL_p = _pad(dL, rows_p, cols_p)
    out_da = _to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), device)
    out_dg = _to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), device)
    k(_to_dev(x_p, device), _to_dev(g_p, device), _to_dev(rms_p, device), _to_dev(dL_p, device), out_da, out_dg)

    da = ttnn.to_torch(out_da)[:rows, :cols]
    dg = ttnn.to_torch(out_dg)[:rows, :cols]
    _n_dbg = min(10, rows * cols)
    _ri = torch.randint(0, rows, (_n_dbg,), dtype=torch.int64)
    _ci = torch.randint(0, cols, (_n_dbg,), dtype=torch.int64)
    print(f"{label}: random {_n_dbg} cells da vs dL_dx_ref (float32):")
    for _k in range(_n_dbg):
        _r, _c = int(_ri[_k]), int(_ci[_k])
        _a = da[_r, _c].float().item()
        _b = dL_dx_ref[_r, _c].float().item()
        print(f"  [{_r},{_c}]  da={_a:.6g}  ref={_b:.6g}  abs_diff={abs(_a - _b):.6g}")
    torch.testing.assert_close(da, dL_dx_ref, rtol=0.15, atol=0.15)
    torch.testing.assert_close(dg.sum(dim=0), dL_dg_ref, rtol=0.15, atol=0.15)
    print(
        f"OK: {label}  cfg={cc}x{rc} cpc={cpc} rpc={rpc} L1_rb={rb} "
        f"rb={rb} -> {cc * rc}c  padded=({rows_p},{cols_p})"
    )


if __name__ == "__main__":
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 90112)
    try:
        # _run_case(device, 256, 384, "[256,384]")
        # _run_case(device, 256, 2048, "[256,2048]")
        _run_case(device, 4096, 4096, "[4096,4096]")
        print("colsplit: all checks passed.")
    finally:
        ttnn.close_device(device)
