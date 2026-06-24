# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-lang kernel for Mamba2 SSD chunked-scan — multi-core, all heads in parallel.

Runs in the tt-lang Python simulator or on Tenstorrent hardware — the
conditional import at module level selects the right backend automatically.

Set TT_LANG_PYTHON_PATH=<tt-lang-repo>/python before importing.

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

    _IS_SIM = False
except ImportError:
    from sim import ttl, ttnn  # type: ignore[no-redef]  # noqa: F401

    _IS_SIM = True

# Tile counts (all in tile units, each tile = 32×32 elements)
_XDT_SHAPE = (2, 2)  # [C=64, D=64]
_B_SHAPE = (2, 4)  # [C=64, N=128]
_L_SHAPE = (2, 2)  # [C=64, C=64]
_H_SHAPE = (2, 4)  # [D=64, N=128]
_GAMMA_SHAPE = (2, 1)  # [C=64, 1-tile-col] (column vector, filled across tile cols)
_SCALAR_SHAPE = (1, 1)  # scalar (value filled across entire 32×32 tile)

_GRID = (8, 8)  # 64 cores for 64 heads

# Module-level cache: avoids re-JIT on repeated calls with same (n_chunks, num_heads, n_groups)
# program_caching KB record: "compute hash ONCE at construction; O(1) cache hit per call"
_KERNEL_CACHE: dict = {}


def make_mamba2_ssd_scan_kernel(n_chunks: int, num_heads: int = 64, n_groups: int = 8):
    """Return a tt-lang kernel that processes all heads across n_chunks chunks.

    Dispatches one head per core on an 8×8 grid (64 cores).
    n_chunks and num_heads are captured by closure — NOT tensor arguments.

    Kernel inputs (all ttnn.Tensor, TILE_LAYOUT, bfloat16):
        log_L        [num_heads*n_chunks*C, C]  — heads stacked along tile-rows
        x_dt         [num_heads*n_chunks*C, D]
        B            [n_groups*n_chunks*C, N]   — group format (NOT head-expanded); each core reads g=h//(H/G)
        C_mat        [n_groups*n_chunks*C, N]   — group format
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
    key = (n_chunks, num_heads, n_groups)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    grid_c = _GRID[1]  # used inside closures for head_idx = node_r * grid_c + node_c
    n_heads_per_group = num_heads // n_groups  # = 8 for H=64, G=8

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
        # ── External DFBs (DM threads fill/drain these) ──────────────────────
        # Per-chunk DFBs: block_count=2 — NCRISC loads chunk c+1 while TRISC computes c
        logl_dfb = ttl.make_dataflow_buffer_like(log_L, shape=(2, 2), block_count=2)  # [C,C]
        xdt_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D]
        b_dfb = ttl.make_dataflow_buffer_like(B, shape=(2, 4), block_count=2)  # [C,N]
        c_dfb = ttl.make_dataflow_buffer_like(C_mat, shape=(2, 4), block_count=2)  # [C,N]
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(2, 2), block_count=2)  # [C,D]
        lg_dfb = ttl.make_dataflow_buffer_like(log_gamma, shape=(2, 1), block_count=2)  # [C,1]
        ld_dfb = ttl.make_dataflow_buffer_like(log_delta, shape=(2, 1), block_count=2)  # [C,1]
        lgs_dfb = ttl.make_dataflow_buffer_like(log_gscalar, shape=(1, 1), block_count=2)  # scalar
        # One-time DFBs: single block
        hinit_dfb = ttl.make_dataflow_buffer_like(h_in, shape=(2, 4), block_count=1)  # [D,N]
        dskip_dfb = ttl.make_dataflow_buffer_like(D_skip_t, shape=(1, 1), block_count=1)  # scalar
        # Output DFBs: single block (BRISC drains immediately after TRISC pushes)
        y_dfb = ttl.make_dataflow_buffer_like(y_out, shape=(2, 2), block_count=1)  # [C,D]
        hout_dfb = ttl.make_dataflow_buffer_like(h_out, shape=(2, 4), block_count=1)  # [D,N]

        # ── Compute-local intermediate DFBs (DM threads do NOT touch these) ─
        # block_count=2 for ping-pong between iterations
        h_state_dfb = ttl.make_dataflow_buffer_like(h_in, shape=(2, 4), block_count=2)  # [D,N] h_prev
        bt_dfb = ttl.make_dataflow_buffer_like(B, shape=(4, 2), block_count=2)  # [N,C] b transposed
        ht_dfb = ttl.make_dataflow_buffer_like(h_in, shape=(4, 2), block_count=2)  # [N,D] h_prev transposed
        L_dfb = ttl.make_dataflow_buffer_like(log_L, shape=(2, 2), block_count=2)  # [C,C] exp(logl)
        QK_dfb = ttl.make_dataflow_buffer_like(log_L, shape=(2, 2), block_count=2)  # [C,C] c@bt
        LQK_dfb = ttl.make_dataflow_buffer_like(log_L, shape=(2, 2), block_count=2)  # [C,C] L*QK
        yi_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D] y_intra
        yc_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D] y_cross
        gamma_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D] exp(bcast(lg))
        ycs_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D] y_cross_sc
        db_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D] bcast(dskip)
        delta_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D] exp(bcast(ld))
        xsc_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [C,D] xdt*delta
        xsc_T_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=(2, 2), block_count=2)  # [D,C] xsc transposed
        g_dfb = ttl.make_dataflow_buffer_like(h_in, shape=(2, 4), block_count=2)  # [D,N] exp(bcast(lgs))

        # ── Compute (head-agnostic — reads from DFBs filled by read()) ────────
        @ttl.compute()
        def compute() -> None:
            # Seed h_prev ping-pong DFB with initial state loaded by read()
            with hinit_dfb.wait() as hinit, h_state_dfb.reserve() as h_init:
                h_init.store(hinit)

            # Keep D_skip alive across all chunks (same scalar tile every iteration)
            dskip = dskip_dfb.wait()

            for chunk_idx in range(n_chunks):
                # Keep per-chunk inputs alive across multiple within-iteration stages.
                # Hardware constraint: inputs to broadcast/transpose must be CB-attached;
                # explicit wait() (without context manager) keeps the CB slot live until
                # the matching pop() at the end of the iteration.
                logl = logl_dfb.wait()
                xdt = xdt_dfb.wait()
                b = b_dfb.wait()
                c_blk = c_dfb.wait()
                x_blk = x_dfb.wait()
                lg = lg_dfb.wait()
                ld = ld_dfb.wait()
                lgs = lgs_dfb.wait()
                # h_prev from ping-pong DFB — CB-attached for transpose AND state update
                h_prev = h_state_dfb.wait()

                # ── Two-stage transposes ────────────────────────────────────────
                # Hardware requires transpose input to be CB-attached; store to
                # intermediate DFB first, then use the DFB value downstream.
                with bt_dfb.reserve() as bt:
                    bt.store(ttl.block.transpose(b))

                with ht_dfb.reserve() as ht:
                    ht.store(ttl.block.transpose(h_prev))

                # ── Intra-chunk: y_intra = (exp(logl) * (c @ b.T)) @ xdt ───────
                with L_dfb.reserve() as L_out:
                    L_out.store(ttl.math.exp(logl))

                with bt_dfb.wait() as bt, QK_dfb.reserve() as qk:
                    qk.store(c_blk @ bt)

                with L_dfb.wait() as L, QK_dfb.wait() as qk, LQK_dfb.reserve() as lqk:
                    lqk.store(L * qk)

                with LQK_dfb.wait() as lqk, yi_dfb.reserve() as yi:
                    yi.store(lqk @ xdt)

                # ── Cross-chunk: y_cross_sc = (c @ h.T) * exp(bcast(lg)) ────────
                with ht_dfb.wait() as ht, yc_dfb.reserve() as yc:
                    yc.store(c_blk @ ht)

                # Fused broadcast+exp: lg is CB-attached, result stored directly to DFB
                with gamma_dfb.reserve() as g_out:
                    g_out.store(ttl.math.exp(ttl.block.broadcast(lg, dims=[-1], shape=(2, 2))))

                with yc_dfb.wait() as yc, gamma_dfb.wait() as gamma, ycs_dfb.reserve() as ycs:
                    ycs.store(yc * gamma)

                # ── Output: y = yi + ycs + bcast(dskip) * x ─────────────────────
                # dskip is CB-attached (waited before loop), re-broadcast each chunk
                with db_dfb.reserve() as db:
                    db.store(ttl.block.broadcast(dskip, dims=[-1, -2], shape=(2, 2)))

                # Fused 4-input chain: (yi + ycs) + (db * x_blk) — all CB-attached
                with (
                    yi_dfb.wait() as yi,
                    ycs_dfb.wait() as ycs,
                    db_dfb.wait() as db,
                    y_dfb.reserve() as y,
                ):
                    y.store(yi + ycs + db * x_blk)

                # ── State update: h = exp(bcast(lgs)) * h + xdt_sc.T @ b ─────────
                # Fused broadcast+exp for delta
                with delta_dfb.reserve() as d_out:
                    d_out.store(ttl.math.exp(ttl.block.broadcast(ld, dims=[-1], shape=(2, 2))))

                with delta_dfb.wait() as delta, xsc_dfb.reserve() as xsc:
                    xsc.store(xdt * delta)

                with xsc_dfb.wait() as xsc, xsc_T_dfb.reserve() as xsc_T:
                    xsc_T.store(ttl.block.transpose(xsc))

                # Fused broadcast+exp for g scalar (1,1) → (2,4)
                with g_dfb.reserve() as g_out:
                    g_out.store(ttl.math.exp(ttl.block.broadcast(lgs, dims=[-1, -2], shape=(2, 4))))

                # Fused: g_bcast * h_prev + xsc_T @ b (all inputs CB-attached)
                with (
                    g_dfb.wait() as g_bcast,
                    xsc_T_dfb.wait() as xsc_T,
                    h_state_dfb.reserve() as h_new,
                ):
                    h_new.store(g_bcast * h_prev + xsc_T @ b)

                # Pop all manually-waited DFB slots
                h_prev.pop()
                logl.pop()
                xdt.pop()
                b.pop()
                c_blk.pop()
                x_blk.pop()
                lg.pop()
                ld.pop()
                lgs.pop()

            dskip.pop()

            # Write final h state to output DFB for write() to drain
            with h_state_dfb.wait() as h_last, hout_dfb.reserve() as hout:
                hout.store(h_last)

        # ── Read (per-core: each core reads its head's slice) ─────────────────
        @ttl.datamovement()
        def read() -> None:
            node_r, node_c = ttl.node(dims=2)
            h_idx = node_r * grid_c + node_c  # head index: 0..63

            # Tile-row bases for this head's data in the full-batch input tensors.
            # Per-chunk head-specific tensors (log_L, x_dt, x, log_gamma, log_delta, y_out):
            #   each head has n_chunks chunks × 2 tile-rows per chunk (C=64 → 2 tiles)
            h_chunk_r = h_idx * n_chunks * 2
            # log_gscalar: 1 tile-row per chunk → head h at [h*n_chunks : (h+1)*n_chunks]
            h_scalar_r = h_idx * n_chunks
            # h_in: 2 tile-rows per head (D=64 → 2 tiles)
            h_state_r = h_idx * 2
            # D_skip_t: 1 tile-row per head (1 scalar tile, TILE×TILE elements)
            h_dskip_r = h_idx

            # B/C group format (bandwidth opt: 8x less DRAM vs head-expanded):
            #   g_idx = h_idx // (H / G);  B/C stored as [G*n_chunks*C, N]
            g_idx = h_idx // n_heads_per_group
            g_chunk_r = g_idx * n_chunks * 2

            with hinit_dfb.reserve() as h_blk:
                ttl.copy(h_in[h_state_r : h_state_r + 2, 0:4], h_blk).wait()
            with dskip_dfb.reserve() as ds_blk:
                ttl.copy(D_skip_t[h_dskip_r : h_dskip_r + 1, 0:1], ds_blk).wait()

            for chunk in range(n_chunks):
                r = h_chunk_r + chunk * 2  # global tile-row for head-specific inputs
                r_g = g_chunk_r + chunk * 2  # global tile-row for group-format B/C
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
                    ttl.copy(B[r_g : r_g + 2, 0:4], b).wait()
                    ttl.copy(C_mat[r_g : r_g + 2, 0:4], c_b).wait()
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

    _KERNEL_CACHE[key] = _kernel
    return _kernel
