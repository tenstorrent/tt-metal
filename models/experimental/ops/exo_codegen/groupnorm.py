# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GroupNorm algorithm definitions using Exo.

Defines @proc procedures for each computational stage of the GroupNorm
3-pass algorithm:

    Pass 1 (Mean):
        gn_mask      - mask input tiles (input * input_mask)
        gn_reduce    - reduce masked tiles to partial sum

    Pass 2 (Variance):
        gn_sub_mean  - subtract mean from input (x - E[x])
        gn_mask      - mask residual tiles (reused)
        gn_square    - square masked residuals ((x - E[x])^2)
        gn_reduce    - reduce squared tiles to partial variance

    Pass 3 (Normalize):
        gn_sub_mean  - subtract mean (reused)
        gn_mask      - mask residual (reused)
        gn_mul_invstd - multiply by 1/sqrt(var+eps)
        gn_gamma     - multiply by gamma (broadcast rows)
        gn_beta      - add beta (broadcast rows)

Each @proc is a flat 1D loop for block_w=1 (single tile column).
Scheduling transforms (divide_loop) can create subblock-tiled variants.
"""

from __future__ import annotations

from exo import proc

from models.experimental.ops.exo_codegen.tt_target import TT_L1_CB
from models.experimental.ops.exo_codegen.groupnorm_target import (
    tt_gn_mask,
    tt_gn_sub_bcast,
    tt_gn_square,
    tt_gn_mul_bcast,
    tt_gn_gamma,
    tt_gn_beta,
    tt_gn_reduce,
)


# ---------------------------------------------------------------------------
# Per-tile operation loops (with subblock protocol per tile)
# ---------------------------------------------------------------------------


@proc
def gn_mask_tiles(
    N: size,
    inp: f32[N] @ TT_L1_CB,
    mask: f32[1] @ TT_L1_CB,
    out: f32[N] @ TT_L1_CB,
):
    for i in seq(0, N):
        tt_gn_mask(i, inp[i : i + 1], mask[0:1], out[i : i + 1])


@proc
def gn_sub_mean_tiles(
    N: size,
    inp: f32[N] @ TT_L1_CB,
    mean: f32[1] @ TT_L1_CB,
    out: f32[N] @ TT_L1_CB,
):
    for i in seq(0, N):
        tt_gn_sub_bcast(i, inp[i : i + 1], mean[0:1], out[i : i + 1])


@proc
def gn_square_tiles(
    N: size,
    inp: f32[N] @ TT_L1_CB,
    out: f32[N] @ TT_L1_CB,
):
    for i in seq(0, N):
        tt_gn_square(i, inp[i : i + 1], out[i : i + 1])


@proc
def gn_mul_invstd_tiles(
    N: size,
    inp: f32[N] @ TT_L1_CB,
    invstd: f32[1] @ TT_L1_CB,
    out: f32[N] @ TT_L1_CB,
):
    for i in seq(0, N):
        tt_gn_mul_bcast(i, inp[i : i + 1], invstd[0:1], out[i : i + 1])


@proc
def gn_gamma_tiles(
    N: size,
    inp: f32[N] @ TT_L1_CB,
    gamma: f32[1] @ TT_L1_CB,
    out: f32[N] @ TT_L1_CB,
):
    for i in seq(0, N):
        tt_gn_gamma(i, inp[i : i + 1], gamma[0:1], out[i : i + 1])


@proc
def gn_beta_tiles(
    N: size,
    inp: f32[N] @ TT_L1_CB,
    beta: f32[1] @ TT_L1_CB,
    out: f32[N] @ TT_L1_CB,
):
    for i in seq(0, N):
        tt_gn_beta(i, inp[i : i + 1], beta[0:1], out[i : i + 1])


# ---------------------------------------------------------------------------
# Reduce loop (no subblock protocol per tile — wraps entire loop)
# ---------------------------------------------------------------------------


@proc
def gn_reduce_tiles(
    N: size,
    inp: f32[N] @ TT_L1_CB,
    scaler: f32[1] @ TT_L1_CB,
    out: f32[N] @ TT_L1_CB,
):
    for i in seq(0, N):
        tt_gn_reduce(i, inp[i : i + 1], scaler[0:1], out[i : i + 1])


# ---------------------------------------------------------------------------
# Proc name to proc mapping for codegen
# ---------------------------------------------------------------------------


def get_procs():
    """Return dict of all GroupNorm @proc procedures."""
    return {
        "mask": gn_mask_tiles,
        "sub_mean": gn_sub_mean_tiles,
        "square": gn_square_tiles,
        "mul_invstd": gn_mul_invstd_tiles,
        "gamma": gn_gamma_tiles,
        "beta": gn_beta_tiles,
        "reduce": gn_reduce_tiles,
    }
