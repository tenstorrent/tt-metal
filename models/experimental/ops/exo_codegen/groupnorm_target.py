# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT hardware instruction definitions for GroupNorm.

Defines @instr blocks for each tile-level operation used in the GroupNorm
compute kernel. Each @instr generates C++ code with placeholder CB names
(SRC_A, SRC_B, DST_CB) that the codegen stage replaces with actual CB
variable names for each step.

Operations:
    tt_gn_mask          - Element-wise multiply (masking, with subblock protocol)
    tt_gn_sub_bcast     - Subtract broadcast scalar (x - mean)
    tt_gn_square        - Self-multiply (squaring, with subblock protocol)
    tt_gn_mul_bcast     - Multiply broadcast scalar (x * inv_std)
    tt_gn_gamma         - Multiply broadcast rows (gamma)
    tt_gn_beta          - Add broadcast rows (beta)
    tt_gn_reduce        - Reduce tile (accumulate, no subblock protocol)
"""

from __future__ import annotations

from exo import instr

from models.experimental.ops.exo_codegen.tt_target import TT_L1_CB


# ---------------------------------------------------------------------------
# Per-tile operations WITH subblock protocol (acquire/commit/wait/pack/release)
# ---------------------------------------------------------------------------


@instr(
    "tile_regs_acquire();\n"
    "mul_tiles(SRC_A, SRC_B, {i}, 0, 0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, DST_CB);\n"
    "tile_regs_release();"
)
def tt_gn_mask(
    i: index,
    inp: [f32][1] @ TT_L1_CB,
    mask: [f32][1] @ TT_L1_CB,
    out: [f32][1] @ TT_L1_CB,
):
    out[0] = inp[0]


@instr(
    "tile_regs_acquire();\n"
    "sub_tiles_bcast_scalar(SRC_A, SRC_B, {i}, 0, 0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, DST_CB);\n"
    "tile_regs_release();"
)
def tt_gn_sub_bcast(
    i: index,
    inp: [f32][1] @ TT_L1_CB,
    scalar: [f32][1] @ TT_L1_CB,
    out: [f32][1] @ TT_L1_CB,
):
    out[0] = inp[0]


@instr(
    "tile_regs_acquire();\n"
    "mul_tiles(SRC_A, SRC_A, {i}, {i}, 0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, DST_CB);\n"
    "tile_regs_release();"
)
def tt_gn_square(
    i: index,
    inp: [f32][1] @ TT_L1_CB,
    out: [f32][1] @ TT_L1_CB,
):
    out[0] = inp[0]


@instr(
    "tile_regs_acquire();\n"
    "mul_tiles_bcast_scalar(SRC_A, SRC_B, {i}, 0, 0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, DST_CB);\n"
    "tile_regs_release();"
)
def tt_gn_mul_bcast(
    i: index,
    inp: [f32][1] @ TT_L1_CB,
    scalar: [f32][1] @ TT_L1_CB,
    out: [f32][1] @ TT_L1_CB,
):
    out[0] = inp[0]


@instr(
    "tile_regs_acquire();\n"
    "mul_tiles_bcast_rows(SRC_A, SRC_B, {i}, 0, 0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, DST_CB);\n"
    "tile_regs_release();"
)
def tt_gn_gamma(
    i: index,
    inp: [f32][1] @ TT_L1_CB,
    gamma: [f32][1] @ TT_L1_CB,
    out: [f32][1] @ TT_L1_CB,
):
    out[0] = inp[0]


@instr(
    "tile_regs_acquire();\n"
    "add_tiles_bcast_rows(SRC_A, SRC_B, {i}, 0, 0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, DST_CB);\n"
    "tile_regs_release();"
)
def tt_gn_beta(
    i: index,
    inp: [f32][1] @ TT_L1_CB,
    beta: [f32][1] @ TT_L1_CB,
    out: [f32][1] @ TT_L1_CB,
):
    out[0] = inp[0]


# ---------------------------------------------------------------------------
# Accumulating operations (NO subblock protocol — wraps entire loop)
# ---------------------------------------------------------------------------


@instr("reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>" "(SRC_A, SRC_B, {i}, 0, 0);")
def tt_gn_reduce(
    i: index,
    inp: [f32][1] @ TT_L1_CB,
    scaler: [f32][1] @ TT_L1_CB,
    out: [f32][1] @ TT_L1_CB,
):
    out[0] = inp[0]
