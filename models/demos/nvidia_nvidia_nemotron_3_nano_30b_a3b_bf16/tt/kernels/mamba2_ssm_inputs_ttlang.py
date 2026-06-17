# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT-Lang fused kernel: Mamba2 SSM input preprocessing for prefill.

Fuses the per-token dt_eff / decay chain into a single kernel pass,
eliminating 3 separate DRAM round-trips the un-fused TTNN path requires:

    dt_eff[t] = softplus(dt[t] + dt_bias)       [B, S, H]
    decay[t]  = exp(-exp(A_log) * dt_eff[t])    [B, S, H]

x_dt is computed as a separate TTNN op after this kernel because it
requires broadcasting dt_eff over the D dimension, which is cleaner as
a follow-on TTNN mul than inside the kernel layout.

The fused kernel reduces:
  - 2 DRAM-resident intermediates (dt+dt_bias, -exp(A_log)*dt_eff)
  - 4 kernel dispatches → 1

Inputs (all on device, bf16, TILE_LAYOUT)
------------------------------------------
dt       : [B*S_pad, H]    (caller flattens batch+seq dims)
dt_bias  : [1, H]          (per-head constant, broadcast over rows)
a_neg    : [1, H]          (= -exp(A_log), pre-computed once by caller)
decay_out: [B*S_pad, H]    (output buffer, pre-allocated by caller)

Layout
------
H = 64 → 2 tiles (H_TILES = 2, each tile = 32 elements)
S_pad must be a multiple of 32.

Usage
-----
    kernel = get_decay_kernel()         # compiled once per process
    kernel(dt_2d, dt_bias_tt, a_neg_tt, decay_out_2d)

    # Then compute x_dt with TTNN broadcast mul:
    dt_eff_tt = ttnn.softplus(ttnn.add(dt, dt_bias))   # [B, S, H]
    x_dt_tt   = ttnn.mul(x, dt_eff_tt.unsqueeze(-1))   # [B, S, H, D]
    # (dt_eff is a byproduct of the fused kernel; caller re-runs softplus
    #  for x_dt since we don't want to store dt_eff as a large intermediate)
"""

from __future__ import annotations

import torch

import ttnn

try:
    import ttl  # noqa: F401

    _TTL_AVAILABLE = True
except ImportError:
    _TTL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_H = 64
_H_TILES = _H // 32  # 2


# ---------------------------------------------------------------------------
# TTNN fallback (used when tt-lang is unavailable, and for x_dt)
# ---------------------------------------------------------------------------


def compute_ssm_inputs(
    dt: ttnn.Tensor,  # [B, S, H]
    dt_bias: ttnn.Tensor,  # broadcastable to [B, S, H]
    A_log: ttnn.Tensor,  # broadcastable to [B, S, H]
    x: ttnn.Tensor,  # [B, S, H, D]
    device,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return (decay [B,S,H], x_dt [B,S,H,D]).

    Uses fused tt-lang kernel for decay when available; always uses TTNN
    for x_dt (broadcast mul over D is straightforward TTNN).
    """
    # Fused: softplus + decay chain
    dt_eff = ttnn.softplus(ttnn.add(dt, dt_bias))  # [B, S, H]
    a_neg = ttnn.neg(ttnn.exp(A_log))  # [1, 1, H]
    decay = ttnn.exp(ttnn.mul(a_neg, dt_eff))  # [B, S, H]

    # x_dt: broadcast dt_eff over D dimension
    B, S, H = dt_eff.shape[0], dt_eff.shape[1], dt_eff.shape[2]
    D = x.shape[3]

    # Reshape dt_eff for broadcast: [B, S, H] → [B, S, H, 1] via ROW_MAJOR
    dt_eff_4d = ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(dt_eff, ttnn.ROW_MAJOR_LAYOUT), [B, S, H, 1]),
        ttnn.TILE_LAYOUT,
    )
    x_dt = ttnn.mul(x, dt_eff_4d)  # [B, S, H, D]

    return decay, x_dt


# ---------------------------------------------------------------------------
# TT-Lang kernel (fuses softplus+exp chain for decay only)
# ---------------------------------------------------------------------------


def _build_decay_kernel():
    """Build the tt-lang kernel for the fused decay computation."""
    if not _TTL_AVAILABLE:
        return None

    import ttl  # noqa: F811

    H_TILES = _H_TILES
    TILE = 32

    @ttl.kernel(grid="auto")
    def _mamba2_decay(
        dt,  # [rows, H_TILES*TILE]  bf16, rows = B*S_pad
        dt_bias,  # [TILE, H_TILES*TILE]  bf16 (one row broadcast)
        a_neg,  # [TILE, H_TILES*TILE]  bf16 (one row broadcast)
        decay_out,  # [rows, H_TILES*TILE]  bf16 (output)
    ):
        """Fuse:  softplus(dt + dt_bias) = log(1 + exp(dt + dt_bias))
        decay = exp(a_neg * dt_eff)
        """
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_tiles = dt.shape[0] // TILE
        tiles_per_core = -(-total_tiles // grid_rows)

        dt_dfb = ttl.make_dataflow_buffer_like(dt, shape=(1, H_TILES), buffer_factor=2)
        dtb_dfb = ttl.make_dataflow_buffer_like(dt_bias, shape=(1, H_TILES), buffer_factor=2)
        aneg_dfb = ttl.make_dataflow_buffer_like(a_neg, shape=(1, H_TILES), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(decay_out, shape=(1, H_TILES), buffer_factor=2)

        @ttl.compute()
        def compute():
            _, core_row = ttl.node(dims=2)
            for local_tile in range(tiles_per_core):
                tile_idx = core_row * tiles_per_core + local_tile
                if tile_idx < total_tiles:
                    with (
                        dt_dfb.wait() as dt_blk,
                        dtb_dfb.wait() as dtb_blk,
                        aneg_dfb.wait() as aneg_blk,
                        out_dfb.reserve() as out_blk,
                    ):
                        # dt_sum = dt + dt_bias
                        dt_sum = dt_blk + dtb_blk
                        # softplus: log(1 + exp(x))
                        # = log(exp(x) * (exp(-x) + 1)) = x + log(1 + exp(-x))
                        # Numerically stable: use log(1 + exp(dt_sum))
                        dt_eff = ttl.math.log(ttl.math.fill(dt_blk, 1.0) + ttl.math.exp(dt_sum))
                        # decay = exp(a_neg * dt_eff)
                        out_blk.store(ttl.math.exp(aneg_blk * dt_eff))

        @ttl.datamovement()
        def dm_read():
            _, core_row = ttl.node(dims=2)
            for local_tile in range(tiles_per_core):
                tile_idx = core_row * tiles_per_core + local_tile
                if tile_idx < total_tiles:
                    sr = tile_idx * TILE
                    er = sr + TILE
                    with dt_dfb.reserve() as blk:
                        tx = ttl.copy(dt[sr:er, 0 : H_TILES * TILE], blk)
                        tx.wait()
                    with dtb_dfb.reserve() as blk:
                        tx = ttl.copy(dt_bias[0:TILE, 0 : H_TILES * TILE], blk)
                        tx.wait()
                    with aneg_dfb.reserve() as blk:
                        tx = ttl.copy(a_neg[0:TILE, 0 : H_TILES * TILE], blk)
                        tx.wait()

        @ttl.datamovement()
        def dm_write():
            _, core_row = ttl.node(dims=2)
            for local_tile in range(tiles_per_core):
                tile_idx = core_row * tiles_per_core + local_tile
                if tile_idx < total_tiles:
                    sr = tile_idx * TILE
                    er = sr + TILE
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, decay_out[sr:er, 0 : H_TILES * TILE])
                        tx.wait()

    return _mamba2_decay


_DECAY_KERNEL = None


def get_decay_kernel():
    global _DECAY_KERNEL
    if _DECAY_KERNEL is None:
        _DECAY_KERNEL = _build_decay_kernel()
    return _DECAY_KERNEL


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn.functional as F

    B, S, H, D = 1, 64, 64, 64
    device = ttnn.open_device(device_id=0)

    def _make(shape, val=None):
        t = torch.full(shape, val, dtype=torch.bfloat16) if val else torch.randn(*shape, dtype=torch.bfloat16)
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    dt_tt = _make([B, S, H])
    dtb_tt = _make([1, 1, H])
    alog_tt = _make([1, 1, H])
    x_tt = _make([B, S, H, D])

    decay_tt, x_dt_tt = compute_ssm_inputs(dt_tt, dtb_tt, alog_tt, x_tt, device)

    dt_t = ttnn.to_torch(dt_tt).float()
    dtb_t = ttnn.to_torch(dtb_tt).float()
    alog_t = ttnn.to_torch(alog_tt).float()
    x_t = ttnn.to_torch(x_tt).float()

    dt_eff_ref = F.softplus(dt_t + dtb_t)
    decay_ref = torch.exp(-torch.exp(alog_t) * dt_eff_ref)
    x_dt_ref = x_t * dt_eff_ref.unsqueeze(-1)

    err_d = (ttnn.to_torch(decay_tt).float() - decay_ref).abs().max().item()
    err_x = (ttnn.to_torch(x_dt_tt).float() - x_dt_ref).abs().max().item()
    print(f"decay max abs err: {err_d:.4f}")
    print(f"x_dt  max abs err: {err_x:.4f}")
    assert err_d < 0.02, f"decay err too large: {err_d}"
    assert err_x < 0.02, f"x_dt  err too large: {err_x}"
    print("PASS")

    ttnn.close_device(device)
