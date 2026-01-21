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

    // ADDR_MOD_3: clear all (common to all cases)
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);

    if (is_in0_16x32) {
        // 16x32: Standard matmul pattern for better cache locality
        // srcA reused for 2 MVMULs, srcB/dest sequential +8
        // Pattern: srcA 0,0,16,16,32,32,48,48; srcB 0,8,0,8,16,24,16,24; dest 0,8,16,24,0,8,16,24

        // ADDR_MOD_0: srcA stay, srcB +8, dest +8
        addr_mod_t{
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // ADDR_MOD_1: srcA +16, srcB -8 (56), dest +8
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 56, .clr = 0, .cr = 0},  // -8 wrapped
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);

        // ADDR_MOD_4: srcA +16, srcB +8, dest -24 (1000)
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 1000, .clr = 0, .cr = 0},  // -24 wrapped
        }
            .set(ADDR_MOD_4);
    } else if (is_in0_32x32) {
        // 32x32: Standard matmul pattern for better cache locality
        // srcA reused for 2 MVMULs, srcB/dest sequential +8

        // ADDR_MOD_0: srcA stay, srcB +8, dest +8
        addr_mod_t{
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // ADDR_MOD_1: srcA +16, srcB -8 (56), dest +8
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 56, .clr = 0, .cr = 0},  // -8 wrapped
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);

        // ADDR_MOD_4: srcA +16, srcB +8, dest -24 (1000) - K accumulation transition
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 1000, .clr = 0, .cr = 0},  // -24 wrapped
        }
            .set(ADDR_MOD_4);

        // ADDR_MOD_5: srcA +16 (wraps to 0), srcB +8, dest +8 - block transition
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},  // 48+16=64=0 mod 64
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_5);
    } else {
        // 1x32: Original pattern
        // ADDR_MOD_0: srcA +16, srcB stay, dest +16
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 0},
            .dest = {.incr = 16, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // ADDR_MOD_1: srcA +16, srcB +16, dest -16 (1008)
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 16, .clr = 0, .cr = 0},
            .dest = {.incr = 1008, .clr = 0, .cr = 0},  // -16 wrapped
        }
            .set(ADDR_MOD_1);
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
        // 32x32 tile: 16 MVMULs with standard matmul pattern for better cache locality
        // srcA: 0,0,16,16,32,32,48,48, 0,0,16,16,32,32,48,48 (reused for 2 MVMULs)
        // srcB: 0,8,0,8,16,24,16,24, 32,40,32,40,48,56,48,56 (sequential +8)
        // dest: 0,8,16,24,0,8,16,24, 32,40,48,56,32,40,48,56 (sequential +8, K accum)
        load_replay_buf(ckernel::math::replay_buf_offset, 16, [] {
            // Block 1: M-rows 0-15
            // K-half 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=0, srcB=0, dest=0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // srcA=0, srcB=8, dest=8
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=16, srcB=0, dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // srcA=16, srcB=8, dest=24 → K accum
            // K-half 1
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=32, srcB=16, dest=0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // srcA=32, srcB=24, dest=8
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=48, srcB=16, dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);  // srcA=48, srcB=24, dest=24 → block transition

            // Block 2: M-rows 16-31
            // K-half 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=0, srcB=32, dest=32
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // srcA=0, srcB=40, dest=40
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=16, srcB=32, dest=48
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // srcA=16, srcB=40, dest=56 → K accum
            // K-half 1
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=32, srcB=48, dest=32
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // srcA=32, srcB=56, dest=40
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=48, srcB=48, dest=48
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // srcA=48, srcB=56, dest=56, clear all
        });
    } else if (is_in0_16x32) {
        // 16x32 tile: 8 MVMULs with standard matmul pattern for better cache locality
        // srcA: 0,0,16,16,32,32,48,48 (reused for 2 MVMULs - better cache)
        // srcB: 0,8,0,8,16,24,16,24 (sequential +8)
        // dest: 0,8,16,24,0,8,16,24 (sequential +8)
        load_replay_buf(ckernel::math::replay_buf_offset, 8, [] {
            // K-half 0, srcA faces 0 and 16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=0, srcB=0, dest=0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // srcA=0, srcB=8, dest=8 → srcA=16, srcB=0, dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=16, srcB=0, dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // srcA=16, srcB=8, dest=24 → srcA=32, srcB=16, dest=0

            // K-half 1, srcA faces 32 and 48
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=32, srcB=16, dest=0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // srcA=32, srcB=24, dest=8 → srcA=48, srcB=16, dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // srcA=48, srcB=16, dest=16
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // srcA=48, srcB=24, dest=24, clear all
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
// in0_tile_r_dim is a template parameter for constexpr branching (no runtime overhead).
template <bool partial_acc = false, uint32_t in0_tile_r_dim = TILE_R_DIM>
inline void _llk_math_custom_mm_(
    uint dst_index, [[maybe_unused]] const bool transpose = false, [[maybe_unused]] const std::uint32_t kt_dim = 1) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    constexpr bool is_in0_32x32 = (in0_tile_r_dim == TILE_R_DIM);
    constexpr bool is_in0_16x32 = (in0_tile_r_dim == 16);

    if constexpr (is_in0_32x32) {
        // 32x32: 16 MVMULs per iteration
        for (uint32_t i = 0; i < kt_dim; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, 16);
        }
    } else if constexpr (is_in0_16x32) {
        // 16x32: 8 MVMULs per iteration
        for (uint32_t i = 0; i < kt_dim; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, 8);
        }
    } else {
        // 1x32 case
        if constexpr (partial_acc) {
            // Partial K accumulation: run all kt_dim iterations with CLR_AB (instructions 0-3)
            // MVMUL accumulates into dest, results persist for next K subblock
            // NO finalization - skip instructions 4-13
            for (uint32_t i = 0; i < kt_dim; i++) {
                lltt::replay(ckernel::math::replay_buf_offset, 4);
            }
        } else {
            // Full accumulation with finalization
            for (uint32_t i = 0; i < kt_dim - 1; i++) {
                lltt::replay(ckernel::math::replay_buf_offset, 4);
            }
            // Final K tile + finalization (MOVD2A/MOVD2B/ELWADD)
            lltt::replay(ckernel::math::replay_buf_offset + 4, 10);
        }
    }
}
