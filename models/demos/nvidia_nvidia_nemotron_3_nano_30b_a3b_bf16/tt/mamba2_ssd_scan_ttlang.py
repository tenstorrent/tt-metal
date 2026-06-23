# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-lang kernel for Mamba2 SSD chunked-scan — multi-core, all heads in parallel.

Runs in the tt-lang Python simulator or on Tenstorrent hardware — the
conditional import at module level selects the right backend automatically.

Set TT_LANG_PYTHON_PATH=/home/ttuser/ssinghal/tt-lang/python before importing.

Each of the 64 cores (8×8 grid) processes one attention head. The caller
stacks all heads' tensors along the tile-row dimension; each core uses
ttl.node() to index into its own head's slice.

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

_GRID = (8, 8)  # 64 cores for 64 heads


def make_mamba2_ssd_scan_kernel(n_chunks: int, num_heads: int = 64):
    """Return a tt-lang kernel that processes all heads across n_chunks chunks.

    Dispatches one head per core on an 8×8 grid (64 cores).
    n_chunks and num_heads are captured by closure — NOT tensor arguments.

    Kernel inputs (all ttnn.Tensor, TILE_LAYOUT, bfloat16):
        log_L        [num_heads*n_chunks*C, C]  — heads stacked along tile-rows
        x_dt         [num_heads*n_chunks*C, D]
        B            [num_heads*n_chunks*C, N]  — B pre-expanded from G groups to num_heads
        C_mat        [num_heads*n_chunks*C, N]
        x            [num_heads*n_chunks*C, D]
        log_gamma    [num_heads*n_chunks*C, 32] — per-row A_cumsum[i], broadcast-filled
        log_delta    [num_heads*n_chunks*C, 32] — per-row A_last-A_cumsum[s], broadcast-filled
        log_gscalar  [num_heads*n_chunks*32,32] — per-chunk A_cumsum[-1], tile-filled;
                                                   head h at tile rows [h*n_chunks:(h+1)*n_chunks]
        h_in         [num_heads*D, N]           — stacked initial states
        D_skip_t     [num_heads*32, 32]         — per-head D scalar, one tile per head
        y_out        [num_heads*n_chunks*C, D]  — output (written in-place)
        h_out        [num_heads*D, N]           — final states
    """
    assert (
        num_heads == _GRID[0] * _GRID[1]
    ), f"num_heads={num_heads} must equal grid {_GRID[0]}×{_GRID[1]}={_GRID[0]*_GRID[1]}"
    grid_c = _GRID[1]  # used inside closures for head_idx = node_r * grid_c + node_c

    @ttl.operation(grid=_GRID)
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
        # Per-chunk DFBs: block_count=2 — NCRISC loads chunk c+1 while TRISC computes c
        logl_dfb = ttl.make_dataflow_buffer_like(log_L, shape=_L_SHAPE, block_count=2)
        xdt_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(B, shape=_B_SHAPE, block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(C_mat, shape=_B_SHAPE, block_count=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=_XDT_SHAPE, block_count=2)
        lg_dfb = ttl.make_dataflow_buffer_like(log_gamma, shape=_GAMMA_SHAPE, block_count=2)
        ld_dfb = ttl.make_dataflow_buffer_like(log_delta, shape=_GAMMA_SHAPE, block_count=2)
        lgs_dfb = ttl.make_dataflow_buffer_like(log_gscalar, shape=_SCALAR_SHAPE, block_count=2)
        # One-time DFBs: single block (loaded once per head, no pipelining benefit)
        hinit_dfb = ttl.make_dataflow_buffer_like(h_in, shape=_H_SHAPE, block_count=1)
        dskip_dfb = ttl.make_dataflow_buffer_like(D_skip_t, shape=_SCALAR_SHAPE, block_count=1)
        # Output DFBs: single block (BRISC drains immediately after TRISC pushes)
        y_dfb = ttl.make_dataflow_buffer_like(y_out, shape=_XDT_SHAPE, block_count=1)
        hout_dfb = ttl.make_dataflow_buffer_like(h_out, shape=_H_SHAPE, block_count=1)

        # ── Compute (head-agnostic — reads from DFBs filled by read()) ────────
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

                # Intra-chunk
                L = ttl.math.exp(logl)
                QK = c_blk @ ttl.block.transpose(b)
                y_intra = (L * QK) @ xdt

                # Cross-chunk
                y_cross = c_blk @ ttl.block.transpose(h_prev)
                gamma = ttl.math.exp(lg)
                y_cross_sc = y_cross * ttl.block.broadcast(gamma, dims=[-1], shape=_XDT_SHAPE)

                # Output
                d_bcast = ttl.block.broadcast(dskip_blk, dims=[0, -1], shape=_XDT_SHAPE)
                y.store(y_intra + y_cross_sc + d_bcast * x_blk)
                y.push()

                # State update
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

        # ── Read (per-core: each core reads its head's slice) ─────────────────
        @ttl.datamovement()
        def read() -> None:
            node_r, node_c = ttl.node(dims=2)
            h_idx = node_r * grid_c + node_c  # head index: 0..63

            # Tile-row bases for this head's data in the full-batch input tensors.
            # Per-chunk tensors (log_L, x_dt, B, C_mat, x, log_gamma, log_delta):
            #   each head has n_chunks chunks × 2 tile-rows per chunk (C=64 → 2 tiles)
            h_chunk_r = h_idx * n_chunks * 2
            # log_gscalar: 1 tile-row per chunk → head h at [h*n_chunks : (h+1)*n_chunks]
            h_scalar_r = h_idx * n_chunks
            # h_in: 2 tile-rows per head (D=64 → 2 tiles)
            h_state_r = h_idx * 2
            # D_skip_t: 1 tile-row per head (1 scalar tile, TILE×TILE elements)
            h_dskip_r = h_idx

            with hinit_dfb.reserve() as h_blk:
                ttl.copy(h_in[h_state_r : h_state_r + 2, 0:4], h_blk).wait()
            with dskip_dfb.reserve() as ds_blk:
                ttl.copy(D_skip_t[h_dskip_r : h_dskip_r + 1, 0:1], ds_blk).wait()

            for chunk in range(n_chunks):
                r = h_chunk_r + chunk * 2  # global tile-row for this chunk
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
                    ttl.copy(
                        log_gscalar[h_scalar_r + chunk : h_scalar_r + chunk + 1, 0:1],
                        lgs,
                    ).wait()

        # ── Write (per-core: each core writes its head's slice) ───────────────
        @ttl.datamovement()
        def write() -> None:
            node_r, node_c = ttl.node(dims=2)
            h_idx = node_r * grid_c + node_c

            h_chunk_r = h_idx * n_chunks * 2
            h_state_r = h_idx * 2

            for chunk in range(n_chunks):
                r = h_chunk_r + chunk * 2
                with y_dfb.wait() as y:
                    ttl.copy(y, y_out[r : r + 2, 0:2]).wait()
            with hout_dfb.wait() as hout:
                ttl.copy(hout, h_out[h_state_r : h_state_r + 2, 0:4]).wait()

    return _kernel
