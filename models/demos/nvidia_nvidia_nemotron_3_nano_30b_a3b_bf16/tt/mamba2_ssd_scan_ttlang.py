# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-lang kernel for Mamba2 SSD chunked-scan (single-head, h_prev=0).

Runs in the tt-lang Python simulator or on Tenstorrent hardware — the
conditional import at module level selects the right backend automatically.

Set TT_LANG_PYTHON_PATH=/home/ttuser/ssinghal/tt-lang/python before importing.

Algorithm per chunk c, per head h
----------------------------------
  L          = exp(log_L)                              [C,C]
  QK         = C_mat @ transpose(B)                   [C,N]@[N,C] → [C,C]
  y_intra    = (L * QK) @ x_dt                        [C,D]
  y_cross    = C_mat @ transpose(h_prev)              [C,D]
  gamma      = exp(log_gamma)  (column vec [C,1])
  y_cross_sc = y_cross * broadcast(gamma → [C,D])
  y[c]       = y_intra + y_cross_sc + broadcast(D_skip,[C,D]) * x
  delta      = exp(log_delta)  (column vec [C,1])
  x_dt_sc    = x_dt * broadcast(delta → [C,D])
  h_prev     = broadcast(exp(log_gscalar),[D,N]) * h_prev
               + transpose(x_dt_sc) @ B              [D,N]
"""
import os as _os
import sys as _sys

_tt_lang_path = _os.environ.get("TT_LANG_PYTHON_PATH", "")
if _tt_lang_path and _tt_lang_path not in _sys.path:
    _sys.path.insert(0, _tt_lang_path)

try:
    import ttl  # noqa: F401 — hardware Metal kernel

    import ttnn  # noqa: F401
except ImportError:
    from sim import ttl, ttnn  # type: ignore[no-redef]  # noqa: F401

# Tile counts (all in tile units, each tile = 32×32 elements)
_XDT_SHAPE = (2, 2)  # [C=64, D=64]
_B_SHAPE = (2, 4)  # [C=64, N=128]
_L_SHAPE = (2, 2)  # [C=64, C=64]
_H_SHAPE = (2, 4)  # [D=64, N=128]
_GAMMA_SHAPE = (2, 1)  # [C=64, 1-tile-col] (column vector, filled across tile cols)
_SCALAR_SHAPE = (1, 1)  # scalar (value filled across entire 32×32 tile)


def make_mamba2_ssd_scan_kernel(n_chunks: int):
    """Return a tt-lang kernel that processes one head across n_chunks chunks.

    n_chunks is captured by closure — it is NOT a tensor argument.

    Kernel inputs (all ttnn.Tensor, TILE_LAYOUT, bfloat16):
        log_L        [n_chunks*C, C]  elements — lower-triangular log-decay per chunk
        x_dt         [n_chunks*C, D]  elements — x*dt per chunk
        B            [n_chunks*C, N]  elements — SSM B matrices per chunk
        C_mat        [n_chunks*C, N]  elements — SSM C matrices per chunk
        x            [n_chunks*C, D]  elements — raw x (for D_skip residual)
        log_gamma    [n_chunks*C, 32] elements — per-row A_cumsum[i], broadcast-filled
        log_delta    [n_chunks*C, 32] elements — per-row A_last-A_cumsum[s], broadcast-filled
        log_gscalar  [n_chunks*32,32] elements — per-chunk A_cumsum[-1], tile-filled
        h_in         [D, N]           elements — initial state
        D_skip_t     [32, 32]         elements — D skip coefficient, tile-filled
        y_out        [n_chunks*C, D]  elements — output (written in-place)
        h_out        [D, N]           elements — final state (written once at end)
    """

    @ttl.operation(grid=(1, 1))
    def _kernel(
        log_L: ttnn.Tensor,
        x_dt: ttnn.Tensor,
        B: ttnn.Tensor,
        C_mat: ttnn.Tensor,
        x: ttnn.Tensor,
        log_gamma: ttnn.Tensor,
        log_delta: ttnn.Tensor,
        log_gscalar: ttnn.Tensor,
        h_in: ttnn.Tensor,
        D_skip_t: ttnn.Tensor,
        y_out: ttnn.Tensor,
        h_out: ttnn.Tensor,
    ) -> None:
        # Per-chunk DFBs: block_count=2 enables compute-DMA pipelining —
        # NCRISC loads chunk c+1 while TRISC computes chunk c.
        logl_dfb = ttl.make_dataflow_buffer_like(log_L, shape=_L_SHAPE, block_count=2)
        xdt_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(B, shape=_B_SHAPE, block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(C_mat, shape=_B_SHAPE, block_count=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=_XDT_SHAPE, block_count=2)
        lg_dfb = ttl.make_dataflow_buffer_like(log_gamma, shape=_GAMMA_SHAPE, block_count=2)
        ld_dfb = ttl.make_dataflow_buffer_like(log_delta, shape=_GAMMA_SHAPE, block_count=2)
        lgs_dfb = ttl.make_dataflow_buffer_like(log_gscalar, shape=_SCALAR_SHAPE, block_count=2)
        # One-time DFBs: single block (no pipelining benefit)
        hinit_dfb = ttl.make_dataflow_buffer_like(h_in, shape=_H_SHAPE, block_count=1)
        dskip_dfb = ttl.make_dataflow_buffer_like(D_skip_t, shape=_SCALAR_SHAPE, block_count=1)
        # Output DFBs: single block (BRISC drains immediately after TRISC pushes)
        y_dfb = ttl.make_dataflow_buffer_like(y_out, shape=_XDT_SHAPE, block_count=1)
        hout_dfb = ttl.make_dataflow_buffer_like(h_out, shape=_H_SHAPE, block_count=1)

        @ttl.compute()
        def compute() -> None:
            hinit_blk = hinit_dfb.wait()
            dskip_blk = dskip_dfb.wait()
            h_prev = hinit_blk
            hinit_popped = False

            for chunk_idx in range(n_chunks):
                logl = logl_dfb.wait()
                xdt = xdt_dfb.wait()
                b = b_dfb.wait()
                c_blk = c_dfb.wait()
                x_blk = x_dfb.wait()
                lg = lg_dfb.wait()
                ld = ld_dfb.wait()
                lgs = lgs_dfb.wait()
                y = y_dfb.reserve()

                L = ttl.math.exp(logl)
                QK = c_blk @ ttl.block.transpose(b)
                y_intra = (L * QK) @ xdt

                y_cross = c_blk @ ttl.block.transpose(h_prev)
                gamma = ttl.math.exp(lg)
                y_cross_sc = y_cross * ttl.block.broadcast(gamma, dims=[-1], shape=_XDT_SHAPE)

                d_bcast = ttl.block.broadcast(dskip_blk, dims=[0, -1], shape=_XDT_SHAPE)
                y.store(y_intra + y_cross_sc + d_bcast * x_blk)
                y.push()

                delta = ttl.math.exp(ld)
                x_dt_sc = xdt * ttl.block.broadcast(delta, dims=[-1], shape=_XDT_SHAPE)
                x_dt_sc_T = ttl.block.transpose(x_dt_sc)
                g_last = ttl.math.exp(lgs)
                g_bcast = ttl.block.broadcast(g_last, dims=[0, -1], shape=_H_SHAPE)
                h_next = g_bcast * h_prev + x_dt_sc_T @ b

                if not hinit_popped:
                    hinit_blk.pop()
                    hinit_popped = True

                h_prev = h_next
                logl.pop()
                xdt.pop()
                b.pop()
                c_blk.pop()
                x_blk.pop()
                lg.pop()
                ld.pop()
                lgs.pop()

            dskip_blk.pop()
            hout = hout_dfb.reserve()
            hout.store(h_prev)
            hout.push()

        @ttl.datamovement()
        def read() -> None:
            with hinit_dfb.reserve() as h_blk:
                ttl.copy(h_in[0:2, 0:4], h_blk).wait()
            with dskip_dfb.reserve() as ds_blk:
                ttl.copy(D_skip_t[0:1, 0:1], ds_blk).wait()

            for chunk in range(n_chunks):
                r = chunk * 2
                with (
                    logl_dfb.reserve() as logl,
                    xdt_dfb.reserve() as xdt,
                    b_dfb.reserve() as b,
                    c_dfb.reserve() as c_b,
                    x_dfb.reserve() as x_b,
                    lg_dfb.reserve() as lg,
                    ld_dfb.reserve() as ld,
                    lgs_dfb.reserve() as lgs,
                ):
                    ttl.copy(log_L[r : r + 2, 0:2], logl).wait()
                    ttl.copy(x_dt[r : r + 2, 0:2], xdt).wait()
                    ttl.copy(B[r : r + 2, 0:4], b).wait()
                    ttl.copy(C_mat[r : r + 2, 0:4], c_b).wait()
                    ttl.copy(x[r : r + 2, 0:2], x_b).wait()
                    ttl.copy(log_gamma[r : r + 2, 0:1], lg).wait()
                    ttl.copy(log_delta[r : r + 2, 0:1], ld).wait()
                    ttl.copy(log_gscalar[chunk : chunk + 1, 0:1], lgs).wait()

        @ttl.datamovement()
        def write() -> None:
            for chunk in range(n_chunks):
                r = chunk * 2
                with y_dfb.wait() as y:
                    ttl.copy(y, y_out[r : r + 2, 0:2]).wait()
            with hout_dfb.wait() as hout:
                ttl.copy(hout, h_out[0:2, 0:4]).wait()

    return _kernel
