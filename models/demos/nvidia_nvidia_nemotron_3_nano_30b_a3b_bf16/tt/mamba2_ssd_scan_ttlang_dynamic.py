# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prototype: single tt-lang kernel for all ISLs via runtime n_chunks tensor arg.

DESIGN
------
n_chunks is passed as a 1×1 TILE_LAYOUT bfloat16 tensor whose element [0, 0]
equals float(n_chunks).  All three coroutines read this value at runtime and
use `for _ in range(int(n)):` instead of closure-baked `for _ in range(n_chunks):`.

A single factory `make_mamba2_ssd_scan_kernel_dynamic()` (no n_chunks param)
populates `_KERNEL_CACHE_DYN` with ONE entry regardless of ISL —
one Python kernel object, one ELF on hardware.

FLOW PER COROUTINE
------------------
All three coroutines read n_chunks directly from the kernel argument SimTensor
via the Python closure — no control DFBs needed:

    n = int(n_chunks_t._tensor[0, 0].item())

read():
  1. n = int(n_chunks_t._tensor[0, 0])
  2. Load one-time h_in / D_skip_t.
  3. Loop `for chunk in range(n):` — load per-chunk tiles into per-input DFBs.

compute():
  1. n = int(n_chunks_t._tensor[0, 0])
  2. Wait on hinit_dfb / dskip_dfb (one-time).
  3. Loop `for _ in range(n):` — process per-chunk DFBs, store y and h.

write():
  1. n = int(n_chunks_t._tensor[0, 0])
  2. Loop `for chunk in range(n):` — drain y_dfb to y_out.
  3. Drain hout_dfb → h_out.

SCALAR READ MECHANISM
---------------------
  SIM: `n_chunks_t._tensor[0, 0].item()` — direct PyTorch access on the
       SimTensor closure variable. Works because all three coroutines are
       Python closures that share the kernel's argument scope.

  HW: Device tensors have no `._tensor` attribute. Hardware equivalents:
       - DM threads: copy tile to local DFB → ttl.raw_element_read(blk, 0, 0)
       - Compute thread: cannot use raw_element_read (DM-only); needs
         either while-loop support or get_common_arg_val extension.

HARDWARE STATUS (as of 2026-06-24)
------------------------------------
  SIM ✓   Works today — all three coroutines read n at runtime via block
          backing-tensor access; the loop bounds are runtime Python ints.

  HW  ✗   Three compiler/API gaps remain:

  1. Scalar read in DM threads (NCRISC / BRISC): need `ttl.raw_element_read`
     for hardware, but it's absent from sim. In hardware mode the DM coroutines
     would call ttl.raw_element_read(nc_blk, 0, 0), which returns a bf16 SSA
     value. Then visit_For needs arith.FPToSIOp(i32, bf16_val) before the
     existing arith.IndexCastOp to produce an index type.

  2. Scalar read in compute thread (TRISC): raw_element_read is DM-only on
     hardware. Options:
       a. Add `while` loop support to pykernel compiler (scf.WhileOp).
       b. Expose n_chunks as a `get_common_arg_val` runtime arg by extending
          @ttl.operation to accept optional scalar runtime args alongside
          ttnn.Tensor args.

  3. Tile-row offsets: h_chunk_r = h_idx * n * 2 — uses runtime `n` as a
     multiplier to compute DRAM tensor addresses. This works fine in DM
     threads since address arithmetic is already runtime (get_common_arg_val
     for tensor base addresses). Needs testing with compiler.

Until these gaps are closed this file is SIM-ONLY.  The existing
mamba2_ssd_scan_ttlang.py (one factory entry per n_chunks) remains the
hardware production kernel.
"""
import os as _os
import sys as _sys

_tt_lang_path = _os.environ.get("TT_LANG_PYTHON_PATH", "")
if _tt_lang_path and _tt_lang_path not in _sys.path:
    _sys.path.insert(0, _tt_lang_path)

try:
    from sim import ttl, ttnn  # type: ignore[assignment]

    _IS_SIM = True
except ImportError:
    import ttl  # type: ignore[no-redef]  # noqa: F401

    import ttnn  # type: ignore[no-redef]  # noqa: F401

    _IS_SIM = False

assert _IS_SIM, "mamba2_ssd_scan_ttlang_dynamic.py is SIM-ONLY. " "Use mamba2_ssd_scan_ttlang.py for hardware."

# Tile shapes (tile units, each tile = 32×32 elements)
_XDT_SHAPE = (2, 2)  # [C=64, D=64]
_B_SHAPE = (2, 4)  # [C=64, N=128]
_L_SHAPE = (2, 2)  # [C=64, C=64]
_H_SHAPE = (2, 4)  # [D=64, N=128]
_GAMMA_SHAPE = (2, 1)  # [C=64, 1-tile-col] column vector
_SCALAR_SHAPE = (1, 1)  # scalar

_GRID = (8, 8)  # 64 cores for 64 heads

# Single cache entry for all n_chunks — key is (num_heads, n_groups) only.
_KERNEL_CACHE_DYN: dict = {}


def make_mamba2_ssd_scan_kernel_dynamic(num_heads: int = 64, n_groups: int = 8):
    """Return a tt-lang SIM kernel accepting any n_chunks via runtime tensor arg.

    n_chunks is NOT a closure parameter; it is read at runtime from
    `n_chunks_t[0, 0]`.  One cache entry serves all ISLs.

    SIM ONLY — see module docstring for hardware compiler gaps.

    Kernel signature:
        kernel(n_chunks_t, log_L, x_dt, B, C_mat, x,
               log_gamma, log_delta, log_gscalar, h_in, D_skip_t, y_out, h_out)
    """
    key = (num_heads, n_groups)
    if key in _KERNEL_CACHE_DYN:
        return _KERNEL_CACHE_DYN[key]

    grid_c = _GRID[1]
    n_heads_per_group = num_heads // n_groups  # 8 for H=64, G=8

    @ttl.operation(grid=_GRID)
    def _kernel(
        n_chunks_t: ttnn.Tensor,  # [32,32] bf16 — element [0,0] = float(n_chunks)
        log_L: ttnn.Tensor,  # [H*n*C, C]  variable
        x_dt: ttnn.Tensor,
        B: ttnn.Tensor,  # [G*n*C, N]  group format
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
        # ── External data DFBs ───────────────────────────────────────────────────
        logl_dfb = ttl.make_dataflow_buffer_like(log_L, shape=_L_SHAPE, block_count=2)
        xdt_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(B, shape=_B_SHAPE, block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(C_mat, shape=_B_SHAPE, block_count=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=_XDT_SHAPE, block_count=2)
        lg_dfb = ttl.make_dataflow_buffer_like(log_gamma, shape=_GAMMA_SHAPE, block_count=2)
        ld_dfb = ttl.make_dataflow_buffer_like(log_delta, shape=_GAMMA_SHAPE, block_count=2)
        lgs_dfb = ttl.make_dataflow_buffer_like(log_gscalar, shape=_SCALAR_SHAPE, block_count=2)
        hinit_dfb = ttl.make_dataflow_buffer_like(h_in, shape=_H_SHAPE, block_count=1)
        dskip_dfb = ttl.make_dataflow_buffer_like(D_skip_t, shape=_SCALAR_SHAPE, block_count=1)
        y_dfb = ttl.make_dataflow_buffer_like(y_out, shape=_XDT_SHAPE, block_count=1)
        hout_dfb = ttl.make_dataflow_buffer_like(h_out, shape=_H_SHAPE, block_count=1)

        # ── Compute-local intermediate DFBs ──────────────────────────────────────
        h_state_dfb = ttl.make_dataflow_buffer_like(h_in, shape=_H_SHAPE, block_count=2)
        bt_dfb = ttl.make_dataflow_buffer_like(B, shape=(4, 2), block_count=2)
        ht_dfb = ttl.make_dataflow_buffer_like(h_in, shape=(4, 2), block_count=2)
        L_dfb = ttl.make_dataflow_buffer_like(log_L, shape=_L_SHAPE, block_count=2)
        QK_dfb = ttl.make_dataflow_buffer_like(log_L, shape=_L_SHAPE, block_count=2)
        LQK_dfb = ttl.make_dataflow_buffer_like(log_L, shape=_L_SHAPE, block_count=2)
        yi_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        yc_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        gamma_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        ycs_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        db_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        delta_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        xsc_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        xsc_T_dfb = ttl.make_dataflow_buffer_like(x_dt, shape=_XDT_SHAPE, block_count=2)
        g_dfb = ttl.make_dataflow_buffer_like(h_in, shape=_H_SHAPE, block_count=2)

        # ── Compute ───────────────────────────────────────────────────────────────
        @ttl.compute()
        def compute() -> None:
            # SIM: read n_chunks from the kernel arg SimTensor via Python closure.
            # HW gap: device tensor has no ._tensor; needs while-loop support or
            # get_common_arg_val extension for TRISC.
            n = int(n_chunks_t._tensor[0, 0].item())

            with hinit_dfb.wait() as hinit, h_state_dfb.reserve() as h_init:
                h_init.store(hinit)

            dskip = dskip_dfb.wait()

            for chunk_idx in range(n):  # runtime bound — Python int in sim
                logl = logl_dfb.wait()
                xdt = xdt_dfb.wait()
                b = b_dfb.wait()
                c_blk = c_dfb.wait()
                x_blk = x_dfb.wait()
                lg = lg_dfb.wait()
                ld = ld_dfb.wait()
                lgs = lgs_dfb.wait()
                h_prev = h_state_dfb.wait()

                with bt_dfb.reserve() as bt:
                    bt.store(ttl.block.transpose(b))
                with ht_dfb.reserve() as ht:
                    ht.store(ttl.block.transpose(h_prev))

                with L_dfb.reserve() as L_out:
                    L_out.store(ttl.math.exp(logl))
                with bt_dfb.wait() as bt, QK_dfb.reserve() as qk:
                    qk.store(c_blk @ bt)
                with L_dfb.wait() as L, QK_dfb.wait() as qk, LQK_dfb.reserve() as lqk:
                    lqk.store(L * qk)
                with LQK_dfb.wait() as lqk, yi_dfb.reserve() as yi:
                    yi.store(lqk @ xdt)

                with ht_dfb.wait() as ht, yc_dfb.reserve() as yc:
                    yc.store(c_blk @ ht)
                with gamma_dfb.reserve() as g_out:
                    g_out.store(ttl.math.exp(ttl.block.broadcast(lg, dims=[-1], shape=(2, 2))))
                with yc_dfb.wait() as yc, gamma_dfb.wait() as gamma, ycs_dfb.reserve() as ycs:
                    ycs.store(yc * gamma)

                with db_dfb.reserve() as db:
                    db.store(ttl.block.broadcast(dskip, dims=[-1, -2], shape=(2, 2)))
                with (
                    yi_dfb.wait() as yi,
                    ycs_dfb.wait() as ycs,
                    db_dfb.wait() as db,
                    y_dfb.reserve() as y,
                ):
                    y.store(yi + ycs + db * x_blk)

                with delta_dfb.reserve() as d_out:
                    d_out.store(ttl.math.exp(ttl.block.broadcast(ld, dims=[-1], shape=(2, 2))))
                with delta_dfb.wait() as delta, xsc_dfb.reserve() as xsc:
                    xsc.store(xdt * delta)
                with xsc_dfb.wait() as xsc, xsc_T_dfb.reserve() as xsc_T:
                    xsc_T.store(ttl.block.transpose(xsc))
                with g_dfb.reserve() as g_out:
                    g_out.store(ttl.math.exp(ttl.block.broadcast(lgs, dims=[-1, -2], shape=(2, 4))))
                with (
                    g_dfb.wait() as g_bcast,
                    xsc_T_dfb.wait() as xsc_T,
                    h_state_dfb.reserve() as h_new,
                ):
                    h_new.store(g_bcast * h_prev + xsc_T @ b)

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

            with h_state_dfb.wait() as h_last, hout_dfb.reserve() as hout:
                hout.store(h_last)

        # ── Read ──────────────────────────────────────────────────────────────────
        @ttl.datamovement()
        def read() -> None:
            node_r, node_c = ttl.node(dims=2)
            h_idx = node_r * grid_c + node_c

            # SIM: read n_chunks from the kernel arg SimTensor via Python closure.
            # HW equivalent: copy to a local DFB → ttl.raw_element_read.
            n = int(n_chunks_t._tensor[0, 0].item())

            # Tile-row bases using runtime n.
            # HW note: `n` as a multiplier in address arithmetic is already
            # supported — tensor addresses are runtime via get_common_arg_val.
            h_chunk_r = h_idx * n * 2
            h_scalar_r = h_idx * n
            h_state_r = h_idx * 2
            h_dskip_r = h_idx
            g_idx = h_idx // n_heads_per_group
            g_chunk_r = g_idx * n * 2

            # One-time loads
            with hinit_dfb.reserve() as h_blk:
                ttl.copy(h_in[h_state_r : h_state_r + 2, 0:4], h_blk).wait()
            with dskip_dfb.reserve() as ds_blk:
                ttl.copy(D_skip_t[h_dskip_r : h_dskip_r + 1, 0:1], ds_blk).wait()

            for chunk in range(n):
                r = h_chunk_r + chunk * 2
                r_g = g_chunk_r + chunk * 2
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

        # ── Write ─────────────────────────────────────────────────────────────────
        @ttl.datamovement()
        def write() -> None:
            node_r, node_c = ttl.node(dims=2)
            h_idx = node_r * grid_c + node_c

            # SIM: read n_chunks from the kernel arg SimTensor via Python closure.
            n = int(n_chunks_t._tensor[0, 0].item())

            h_chunk_r = h_idx * n * 2
            h_state_r = h_idx * 2

            for chunk in range(n):
                r = h_chunk_r + chunk * 2
                with y_dfb.wait() as y:
                    ttl.copy(y, y_out[r : r + 2, 0:2]).wait()
            with hout_dfb.wait() as hout:
                ttl.copy(hout, h_out[h_state_r : h_state_r + 2, 0:4]).wait()

    _KERNEL_CACHE_DYN[key] = _kernel
    return _kernel
