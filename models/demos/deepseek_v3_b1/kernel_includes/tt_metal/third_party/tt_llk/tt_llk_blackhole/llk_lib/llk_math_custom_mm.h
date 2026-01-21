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
// Supports 1x32 (4 MVMULs) and 16x32 (8 MVMULs) tile shapes
//
// For 1x32 output tile (M=1):
//   out[face0] = srcA[face0] × srcB[face0] + srcA[face2] × srcB[face1]
//   out[face1] = srcA[face1] × srcB[face0] + srcA[face3] × srcB[face1]
//
// For 16x32 output tile (M=16):
//   Processes 2 row chunks (rows 0-7, 8-15) × 2 column faces
//   8 MVMULs total with negative indexing for accumulation

template <int MATH_FIDELITY_DESC>
inline void custom_mm_configure_addrmod(
    const bool transpose,
    [[maybe_unused]] const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face = false) {
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

    if (is_in0_16x32) {
        // ADDR_MOD_4: After row chunk 0 (MVMULs 1-4), prepare for row chunk 1
        // srcB layout for 16x32 in0:
        //   addr 0:  M rows 0-7,  K cols 0-15
        //   addr 8:  M rows 8-15, K cols 0-15
        //   addr 16: M rows 0-7,  K cols 16-31
        //   addr 24: M rows 8-15, K cols 16-31
        // Row chunk 0 uses srcB 0, 0, 16, 16
        // Row chunk 1 uses srcB 8, 8, 24, 24
        // After MVMUL4: srcA=48, srcB=16, dest=16
        // For MVMUL5: srcA=0, srcB=8, dest=8
        // srcA: 48 → 0, use clr=1
        // srcB: 16 → 8, incr = -8 = 56 (6-bit)
        // dest: 16 → 8, incr = -8 = 1016 (10-bit)
        addr_mod_t{
            .srca = {.incr = 0, .clr = 1, .cr = 0},     // clear srcA to 0
            .srcb = {.incr = 56, .clr = 0, .cr = 0},    // 16 + 56 = 72 = 8 (mod 64)
            .dest = {.incr = 1016, .clr = 0, .cr = 0},  // 16 + 1016 = 1032 = 8 (mod 1024)
        }
            .set(ADDR_MOD_4);
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
    const bool is_in0_16x32 = (in0_tile_r_dim == 16) && (in0_tile_c_dim == TILE_C_DIM);

    if (is_in0_16x32) {
        // 16x32 tile: 8 MVMULs (2 row chunks × 2 columns × 2 K-partials)
        load_replay_buf(ckernel::math::replay_buf_offset, 8, [] {
            // Row chunk 0 (rows 0-7)
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // cols 0-15, K-part 1
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // cols 16-31, K-part 1
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // cols 0-15, K-part 2 (accumulates)
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // cols 16-31, K-part 2 (accumulates), prepare row chunk 1

            // Row chunk 1 (rows 8-15)
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // cols 0-15, K-part 1
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // cols 16-31, K-part 1 (reuse ADDR_MOD_1!)
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // cols 0-15, K-part 2 (accumulates)
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // cols 16-31, K-part 2 (accumulates), clear all
        });
    } else {
        // 1x32 tile: 4 MVMULs (original optimized version)
        // With negative dest incr in ADDR_MOD_1, dest cycles: 0->16->0->16
        // This allows MVMULs 3&4 to accumulate directly into same locations as MVMULs 1&2
        load_replay_buf(ckernel::math::replay_buf_offset, 4, [] {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // dest=0,  srcA=0,  srcB=0   -> after: dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // dest=16, srcA=16, srcB=0   -> after: dest=0 (16-16)
            TTI_MVMUL(
                p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);      // dest=0 (accumulates!), srcA=32, srcB=16 -> after: dest=16
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);  // dest=16 (accumulates!), srcA=48, srcB=16, clear all
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
// Supports both 1x32 (4 MVMULs) and 16x32 (8 MVMULs) tile shapes.
template <bool partial_acc = false>
inline void _llk_math_custom_mm_(
    uint dst_index,
    [[maybe_unused]] const bool transpose = false,
    [[maybe_unused]] const std::uint32_t kt_dim = 1,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    const bool is_in0_16x32 = (in0_tile_r_dim == 16);
    const uint32_t replay_len = is_in0_16x32 ? 8 : 4;

    // All iterations use the same MVMULs - face accumulation happens automatically
    // via dest address cycling (negative incr: 0->16->0->16)
    for (uint32_t i = 0; i < kt_dim; i++) {
        lltt::replay(ckernel::math::replay_buf_offset, replay_len);
    }
}
