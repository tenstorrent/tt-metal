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

// CUSTOM_MM
// Custom matmul that uses MOP to loop both srcA and srcB along inner dim. Output height
// and width should be single tile with tile shape [1, 32]. Further work will uplift the
// custom mm to support for tiles along the width.
template <int MATH_FIDELITY_DESC>
inline void custom_mm_configure_addrmod(
    const bool transpose,
    [[maybe_unused]] const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face = false) {
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = -16, .clr = 0, .cr = 0},  // Negative increment: 16 - 16 = 0
    }
        .set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);
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
    // With negative dest incr in ADDR_MOD_1, dest cycles: 0->16->0->16
    // This allows MVMULs 3&4 to accumulate directly into same locations as MVMULs 1&2
    // No need for MOVD2A/MOVD2B/ELWADD finalization
    load_replay_buf(
        ckernel::math::replay_buf_offset,
        4,
        // Lambda function to load reply buffer
        [] {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // dest=0,  srcA=0,  srcB=0   -> after: dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // dest=16, srcA=16, srcB=0   -> after: dest=0 (16-16)
            TTI_MVMUL(
                p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);      // dest=0 (accumulates!), srcA=32, srcB=16 -> after: dest=16
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);  // dest=16 (accumulates!), srcA=48, srcB=16, clear
        });
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

// Simplified implementation: NUM_FIDELITY_PHASES = 0 (no high fidelity mode)
// With dest cr=1 wrapping, face accumulation happens automatically during MVMULs
// Template parameter partial_acc:
//   false (default): Full custom_mm - MVMULs for all K tiles (faces auto-accumulated)
//   true: Partial K accumulation - MVMULs only, results persist for next K subblock
template <bool partial_acc = false>
inline void _llk_math_custom_mm_(
    uint dst_index, [[maybe_unused]] const bool transpose = false, [[maybe_unused]] const std::uint32_t kt_dim = 1) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // All iterations use the same 4 MVMULs - face accumulation happens automatically
    // via dest address cycling (negative incr in ADDR_MOD_1: 0->16->0->16)
    for (uint32_t i = 0; i < kt_dim; i++) {
        lltt::replay(ckernel::math::replay_buf_offset, 4);
    }
}
