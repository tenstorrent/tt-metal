// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"

using namespace ckernel;

// CUSTOM_MM - Optimized matmul with negative indexing for direct face accumulation
// Supports 1x32 (4 MVMULs), 16x32 (8 MVMULs), and 32x32 (16 MVMULs) tile shapes
//
// For 1x32 output tile (M=1):
//   out[face0] = srcA[face0] × srcB[face0] + srcA[face2] × srcB[face1]
//   out[face1] = srcA[face1] × srcB[face0] + srcA[face3] × srcB[face1]
//
// For 16x32 output tile (M=16):
//   Processes 2 row chunks (rows 0-7, 8-15) × 2 column faces
//   8 MVMULs total with negative indexing for accumulation
//
// For 32x32 output tile (M=32):
//   Processes 4 row chunks (rows 0-7, 8-15, 16-23, 24-31) × 2 column faces
//   16 MVMULs total with negative indexing for accumulation

template <int MATH_FIDELITY_DESC>
inline void custom_mm_configure_addrmod(
    const bool transpose,
    [[maybe_unused]] const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face = false) {
    const bool is_in0_32x32 = (in0_tile_r_dim == TILE_R_DIM) && (in0_tile_c_dim == TILE_C_DIM);
    const bool is_in0_16x32 = (in0_tile_r_dim == 16) && (in0_tile_c_dim == TILE_C_DIM);

    // ADDR_MOD_0: srcA +16, srcB stay, dest +16
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // ADDR_MOD_1: srcA +16, srcB +16, dest -16 (negative increment: 16 - 16 = 0)
    // dest: 10-bit field, -16 = 1024-16 = 1008
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 1008, .clr = 0, .cr = 0},  // -16 wrapped
    }
        .set(ADDR_MOD_1);

    // ADDR_MOD_3: clear all
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);

    if (is_in0_32x32 || is_in0_16x32) {
        // ADDR_MOD_4: Transition between odd→even row chunks (chunks 0→1, 2→3)
        // srcB and dest go backwards by 8 to reach the next M-row section
        // srcA: 48 → 0, use clr=1
        // srcB: 16 → 8 (or 48 → 40), incr = -8 = 56 (6-bit)
        // dest: 16 → 8 (or 48 → 40), incr = -8 = 1016 (10-bit)
        addr_mod_t{
            .srca = {.incr = 0, .clr = 1, .cr = 0},     // clear srcA to 0
            .srcb = {.incr = 56, .clr = 0, .cr = 0},    // -8 wrapped (mod 64)
            .dest = {.incr = 1016, .clr = 0, .cr = 0},  // -8 wrapped (mod 1024)
        }
            .set(ADDR_MOD_4);
    }

    if (is_in0_32x32) {
        // ADDR_MOD_5: Transition from chunk 1→2 (rows 8-15 → rows 16-23)
        // srcB and dest go forward by 8 to reach the next face pair
        // srcA: 48 → 0, use clr=1
        // srcB: 24 → 32, incr = +8
        // dest: 24 → 32, incr = +8
        addr_mod_t{
            .srca = {.incr = 0, .clr = 1, .cr = 0},  // clear srcA to 0
            .srcb = {.incr = 8, .clr = 0, .cr = 0},  // 24 + 8 = 32
            .dest = {.incr = 8, .clr = 0, .cr = 0},  // 24 + 8 = 32
        }
            .set(ADDR_MOD_5);
    }
}

template <int NUM_FIDELITY_PHASES>
inline void custom_mm_configure_mop(
    [[maybe_unused]] bool transpose,
    [[maybe_unused]] const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face = false) {
    const bool is_in0_32x32 = (in0_tile_r_dim == TILE_R_DIM) && (in0_tile_c_dim == TILE_C_DIM);
    const bool is_in0_16x32 = (in0_tile_r_dim == 16) && (in0_tile_c_dim == TILE_C_DIM);

    if (is_in0_32x32) {
        // 32x32 tile: 16 MVMULs (4 row chunks × 2 columns × 2 K-partials)
        // srcB cycles: 0,0,16,16, 8,8,24,24, 32,32,48,48, 40,40,56,56
        // dest cycles: 0,16,0,16, 8,24,8,24, 32,48,32,48, 40,56,40,56
        load_replay_buf(ckernel::math::replay_buf_offset, 16, [] {
            // Row chunk 0 (rows 0-7): srcB 0,0,16,16, dest 0,16,0,16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // transition to chunk 1: srcB 16→8, dest 16→8

            // Row chunk 1 (rows 8-15): srcB 8,8,24,24, dest 8,24,8,24
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);  // transition to chunk 2: srcB 24→32, dest 24→32

            // Row chunk 2 (rows 16-23): srcB 32,32,48,48, dest 32,48,32,48
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // transition to chunk 3: srcB 48→40, dest 48→40

            // Row chunk 3 (rows 24-31): srcB 40,40,56,56, dest 40,56,40,56
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);  // final, clear all
        });
    } else if (is_in0_16x32) {
        // 16x32 tile: 8 MVMULs (2 row chunks × 2 columns × 2 K-partials)
        load_replay_buf(ckernel::math::replay_buf_offset, 8, [] {
            // Row chunk 0 (rows 0-7)
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // transition to chunk 1

            // Row chunk 1 (rows 8-15)
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);  // final, clear all
        });
    } else {
        // 1x32 tile: 8 MVMULs + MOVD2A/MOVD2B + ELWADD for K-reduction
        // First 4 MVMULs: partial K accumulation, then clear AB
        // Next 4 MVMULs: more partial K accumulation
        // MOVD2A/MOVD2B: move results to srcA/srcB
        // ELWADD: final reduction
        load_replay_buf(ckernel::math::replay_buf_offset, 14, [] {
            // First K-half: srcA 0,16,32,48 with srcB 0,16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // dest=0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // dest=0 (accumulate)
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // dest=16 (accumulate), clear AB

            // Second K-half: new srcA/srcB data
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // dest=0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // dest=0 (accumulate)
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);  // dest=16 (accumulate)

            // Move partial results from dest to srcA/srcB for reduction
            TTI_MOVD2A(0, 0, ADDR_MOD_3, p_movd2a::MOV_1_ROW, 0);
            TTI_MOVD2A(0, 16, ADDR_MOD_3, p_movd2a::MOV_1_ROW, 16);
            TTI_MOVD2B(0, 0, ADDR_MOD_3, p_movd2a::MOV_1_ROW, 32);
            TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2a::MOV_1_ROW, 48);

            // Element-wise add for final reduction
            TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
            TTI_ELWADD(p_setrwc::CLR_AB, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
        });
    }
}

template <int MATH_FIDELITY_DESC>
inline void _llk_math_custom_mm_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face = false,
    const std::uint32_t transpose = 0,
    const std::uint32_t kt_dim = 1) {
    custom_mm_configure_addrmod<MATH_FIDELITY_DESC>(
        transpose, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    custom_mm_configure_mop<MATH_FIDELITY_PHASES>(
        transpose > 0, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Optimized implementation with direct face accumulation via negative indexing.
// Supports 1x32 (14 instr), 16x32 (8 MVMULs), and 32x32 (16 MVMULs) tile shapes.
template <bool partial_acc = false>
inline void _llk_math_custom_mm_(
    uint dst_index,
    [[maybe_unused]] const bool transpose = false,
    [[maybe_unused]] const std::uint32_t kt_dim = 1,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    const bool is_in0_32x32 = (in0_tile_r_dim == TILE_R_DIM);
    const bool is_in0_16x32 = (in0_tile_r_dim == 16);

    if (is_in0_32x32) {
        // 32x32: 16 MVMULs per iteration
        for (uint32_t i = 0; i < kt_dim; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, 16);
        }
    } else if (is_in0_16x32) {
        // 16x32: 8 MVMULs per iteration
        for (uint32_t i = 0; i < kt_dim; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, 8);
        }
    } else {
        // 1x32: First (kt_dim-1) iterations use 4 MVMULs, last iteration uses remaining 10
        // Non-last iterations: 4 MVMULs with CLR_AB (allows new srcA/srcB to be loaded)
        for (uint32_t i = 0; i < kt_dim - 1; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, 4);
        }
        // Last iteration: skip first 4, replay remaining 10 (4 MVMULs + MOVD2A/MOVD2B + ELWADD)
        lltt::replay(ckernel::math::replay_buf_offset + 4, 10);
    }
}
