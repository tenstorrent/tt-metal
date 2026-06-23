# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-lang sim implementation of the Mamba2 SSD chunked-scan (single-head, h_prev=0).

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

from sim import ttl, ttnn

# Tile counts (all in tile units, each tile = 32×32 elements)
_XDT_SHAPE = (2, 2)  # [C=64, D=64]
_B_SHAPE = (2, 4)  # [C=64, N=128]
_L_SHAPE = (2, 2)  # [C=64, C=64]
_H_SHAPE = (2, 4)  # [D=64, N=128]
_GAMMA_SHAPE = (2, 1)  # [C=64, 1-tile-col] (column vector, filled across tile cols)
_SCALAR_SHAPE = (1, 1)  # scalar (value filled across entire 32×32 tile)


def make_mamba2_ssd_scan_kernel(n_chunks: int):
    """Return a tt-lang sim kernel that processes one head across n_chunks chunks.

    n_chunks is captured by closure — it is NOT a tensor argument.

    Kernel inputs (all SimTensor, TILE_LAYOUT):
        log_L        [n_chunks*C, C]  elements — lower-triangular log-decay per chunk
        x_dt         [n_chunks*C, D]  elements — x*dt per chunk
        B            [n_chunks*C, N]  elements — SSM B matrices per chunk
        C_mat        [n_chunks*C, N]  elements — SSM C matrices per chunk
        x            [n_chunks*C, D]  elements — raw x (for D_skip residual)
        log_gamma    [n_chunks*C, 32] elements — per-row log(gamma[i])=A_cumsum[i], broadcast-filled
        log_delta    [n_chunks*C, 32] elements — per-row log(delta[s])=A_last-A_cumsum[s], broadcast-filled
        log_gscalar  [n_chunks*32,32] elements — per-chunk log(gamma_last)=A_cumsum[C-1], tile-filled
        h_in         [D, N]           elements — initial state (zeros for h_prev=0)
        D_skip_t     [32, 32]         elements — D skip coefficient, tile-filled
        y_out        [n_chunks*C, D]  elements — output (pre-allocated zeros, written in-place)
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
        # ── DFBs (all shapes in tile units) ───────────────────────────────
        logl_dfb = ttl.make_dataflow_buffer_like(log_L, shape=_L_SHAPE)
        xdt_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE)
        b_dfb = ttl.make_dataflow_buffer_like(B, shape=_B_SHAPE)
        c_dfb = ttl.make_dataflow_buffer_like(C_mat, shape=_B_SHAPE)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=_XDT_SHAPE)
        lg_dfb = ttl.make_dataflow_buffer_like(log_gamma, shape=_GAMMA_SHAPE)
        ld_dfb = ttl.make_dataflow_buffer_like(log_delta, shape=_GAMMA_SHAPE)
        lgs_dfb = ttl.make_dataflow_buffer_like(log_gscalar, shape=_SCALAR_SHAPE)
        hinit_dfb = ttl.make_dataflow_buffer_like(h_in, shape=_H_SHAPE)
        dskip_dfb = ttl.make_dataflow_buffer_like(D_skip_t, shape=_SCALAR_SHAPE)
        y_dfb = ttl.make_dataflow_buffer_like(y_out, shape=_XDT_SHAPE)
        hout_dfb = ttl.make_dataflow_buffer_like(h_out, shape=_H_SHAPE)

        # ── Compute ───────────────────────────────────────────────────────
        @ttl.compute()
        def compute() -> None:
            # Load one-time tensors before chunk loop via direct .wait().
            # We keep hinit_blk as a separate reference so we can pop() it
            # explicitly after the first chunk consumes it as an arithmetic
            # source (mark_assign_src_complete fires then, enabling pop).
            hinit_blk = hinit_dfb.wait()
            dskip_blk = dskip_dfb.wait()
            h_prev = hinit_blk  # starts as the wait-block; reassigned each iter
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

                # ── Intra-chunk (Steps A-D) ───────────────────────────
                L = ttl.math.exp(logl)  # [C,C]
                QK = c_blk @ ttl.block.transpose(b)  # [C,N]@[N,C]→[C,C]
                y_intra = (L * QK) @ xdt  # [C,D]

                # ── Cross-chunk (Steps E-F) ───────────────────────────
                y_cross = c_blk @ ttl.block.transpose(h_prev)  # [C,N]@[N,D]→[C,D]
                gamma = ttl.math.exp(lg)  # (2,1) tile
                y_cross_sc = y_cross * ttl.block.broadcast(gamma, dims=[-1], shape=_XDT_SHAPE)

                # ── Output (Step G) ───────────────────────────────────
                d_bcast = ttl.block.broadcast(dskip_blk, dims=[0, -1], shape=_XDT_SHAPE)
                y.store(y_intra + y_cross_sc + d_bcast * x_blk)
                y.push()

                # ── State update (Step H) ─────────────────────────────
                delta = ttl.math.exp(ld)
                x_dt_sc = xdt * ttl.block.broadcast(delta, dims=[-1], shape=_XDT_SHAPE)
                x_dt_sc_T = ttl.block.transpose(x_dt_sc)  # [D,C]
                g_last = ttl.math.exp(lgs)  # (1,1)
                g_bcast = ttl.block.broadcast(g_last, dims=[0, -1], shape=_H_SHAPE)
                # h_prev was used as arithmetic source above (in the matmul),
                # which fires assign_src → POP is now allowed.
                h_next = g_bcast * h_prev + x_dt_sc_T @ b  # [D,N]

                # Pop the initial hinit_blk after chunk 0 uses it;
                # for subsequent chunks h_prev is already a temp block.
                if not hinit_popped:
                    hinit_blk.pop()
                    hinit_popped = True

                h_prev = h_next  # now a temporary block; no pop needed

                # Pop chunk wait-blocks
                logl.pop()
                xdt.pop()
                b.pop()
                c_blk.pop()
                x_blk.pop()
                lg.pop()
                ld.pop()
                lgs.pop()

            # Pop the D_skip one-time block after all chunks
            dskip_blk.pop()

            hout = hout_dfb.reserve()
            hout.store(h_prev)
            hout.push()

        # ── Read ──────────────────────────────────────────────────────────
        @ttl.datamovement()
        def read() -> None:
            # One-time loads (before chunk loop — matches compute() order)
            with hinit_dfb.reserve() as h_blk:
                ttl.copy(h_in[0:2, 0:4], h_blk).wait()  # [D,N] = (2,4) tiles
            with dskip_dfb.reserve() as ds_blk:
                ttl.copy(D_skip_t[0:1, 0:1], ds_blk).wait()

            for chunk in range(n_chunks):
                r = chunk * 2  # tile-row offset: each chunk = 2 tile rows (C=64/32=2)
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
                    ttl.copy(log_L[r : r + 2, 0:2], logl).wait()  # (2,2) tiles
                    ttl.copy(x_dt[r : r + 2, 0:2], xdt).wait()  # (2,2)
                    ttl.copy(B[r : r + 2, 0:4], b).wait()  # (2,4)
                    ttl.copy(C_mat[r : r + 2, 0:4], c_b).wait()  # (2,4)
                    ttl.copy(x[r : r + 2, 0:2], x_b).wait()  # (2,2)
                    ttl.copy(log_gamma[r : r + 2, 0:1], lg).wait()  # (2,1) col-vec
                    ttl.copy(log_delta[r : r + 2, 0:1], ld).wait()  # (2,1) col-vec
                    # log_gscalar: 1 tile per chunk (tile row = chunk index)
                    ttl.copy(log_gscalar[chunk : chunk + 1, 0:1], lgs).wait()

        # ── Write ─────────────────────────────────────────────────────────
        @ttl.datamovement()
        def write() -> None:
            for chunk in range(n_chunks):
                r = chunk * 2
                with y_dfb.wait() as y:
                    ttl.copy(y, y_out[r : r + 2, 0:2]).wait()
            with hout_dfb.wait() as hout:
                ttl.copy(hout, h_out[0:2, 0:4]).wait()

    return _kernel
