# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Eltwise unary operation definitions using Exo.

Defines reader, compute, and writer algorithms as @proc procedures,
plus scheduled variants that demonstrate Exo's loop tiling transforms.

Each @proc maps to one of the 3 kernels that run on a Tensix core:
    eltwise_reader   -> NCRISC reader kernel
    eltwise_compute  -> TRISC compute kernel (identity or relu)
    eltwise_writer   -> NCRISC writer kernel

Scheduled variants use divide_loop to create block-tiled loops that
match TT-Metal's per_core_block_cnt / per_core_block_dim pattern.
"""

from __future__ import annotations

from exo import proc
from exo.API_scheduling import divide_loop, rename

from models.experimental.ops.exo_codegen.tt_target import (
    TT_DRAM,
    TT_L1_CB,
    tt_read_tile,
    tt_identity_tile,
    tt_relu_tile,
    tt_write_tile,
)

# ---------------------------------------------------------------------------
# Base algorithms (flat loops)
# ---------------------------------------------------------------------------


@proc
def eltwise_reader(N: size, src: f32[N] @ TT_DRAM, cb_in: f32[N] @ TT_L1_CB):
    for i in seq(0, N):
        tt_read_tile(i, src[i : i + 1], cb_in[i : i + 1])


@proc
def eltwise_identity_compute(N: size, cb_in: f32[N] @ TT_L1_CB, cb_out: f32[N] @ TT_L1_CB):
    for i in seq(0, N):
        tt_identity_tile(cb_in[i : i + 1], cb_out[i : i + 1])


@proc
def eltwise_relu_compute(N: size, cb_in: f32[N] @ TT_L1_CB, cb_out: f32[N] @ TT_L1_CB):
    for i in seq(0, N):
        tt_relu_tile(cb_in[i : i + 1], cb_out[i : i + 1])


@proc
def eltwise_writer(N: size, cb_out: f32[N] @ TT_L1_CB, dst: f32[N] @ TT_DRAM):
    for i in seq(0, N):
        tt_write_tile(i, cb_out[i : i + 1], dst[i : i + 1])


# ---------------------------------------------------------------------------
# Scheduled variants (block-tiled loops)
# ---------------------------------------------------------------------------


def make_tiled_compute(base_proc, block_dim: int, name: str):
    """Apply divide_loop to create block_idx / tile_idx nested loops.

    This transforms:
        for i in seq(0, N): compute_tile(...)
    Into:
        for block_idx in seq(0, N / block_dim):
            for tile_idx in seq(0, block_dim):
                compute_tile(...)
        for tile_idx in seq(0, N % block_dim):    # remainder
            compute_tile(...)

    This matches TT-Metal's per_core_block_cnt / per_core_block_dim pattern.
    """
    scheduled = divide_loop(base_proc, "i", block_dim, ["block_idx", "tile_idx"], tail="cut")
    scheduled = rename(scheduled, name)
    return scheduled


# Pre-built tiled variants with block_dim=1 (simplest — matches existing kernels)
eltwise_identity_compute_tiled = make_tiled_compute(eltwise_identity_compute, 1, "eltwise_identity_compute_tiled")
eltwise_relu_compute_tiled = make_tiled_compute(eltwise_relu_compute, 1, "eltwise_relu_compute_tiled")


def get_procs(op: str = "identity", block_dim: int = 1):
    """Return (reader, compute, writer) procs for the given op.

    Args:
        op: "identity" or "relu"
        block_dim: Tile block dimension for compute loop tiling.
                   1 = flat loop, >1 = nested block/tile loops.

    Returns:
        Tuple of (reader_proc, compute_proc, writer_proc)
    """
    if op == "identity":
        base_compute = eltwise_identity_compute
    elif op == "relu":
        base_compute = eltwise_relu_compute
    else:
        raise ValueError(f"Unsupported op: {op}")

    if block_dim == 1:
        compute = base_compute
    else:
        compute = make_tiled_compute(base_compute, block_dim, f"eltwise_{op}_compute_bd{block_dim}")

    return eltwise_reader, compute, eltwise_writer
