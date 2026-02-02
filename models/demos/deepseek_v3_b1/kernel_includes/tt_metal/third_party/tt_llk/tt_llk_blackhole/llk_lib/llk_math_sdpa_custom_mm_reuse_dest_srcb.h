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

// sdpa_custom_mm_reuse_dest_srcb
// Custom matmul that uses MOP to loop both srcA and srcB along inner dim. Output height
// and width should be single tile with tile shape [1, 32]. Further work will uplift the
// custom mm to support for tiles along the width.
template <int MATH_FIDELITY_DESC>
inline void sdpa_custom_mm_reuse_dest_srcb_configure_addrmod(
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
        .dest = {.incr = 0, .clr = 0, .cr = 1},  // Return to tile base
    }
        .set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 48, .clr = 0, .cr = 0, .c_to_cr = 1},  // Update to next tile
    }
        .set(ADDR_MOD_3);
}

template <int NUM_FIDELITY_PHASES>
inline void sdpa_custom_mm_reuse_dest_srcb_configure_mop(
    [[maybe_unused]] bool transpose,
    [[maybe_unused]] const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face = false) {
    // Select MOV instruction count based on output tile height (in0_tile_r_dim)
    // m=1/4: 4 MOVs (MOV_4_ROWS), m=8: no tail reduction
    // Replay buffer size: m=8: 8 MVMULs, m=1/4: 8 MVMULs + 4 MOVs + 2 ELWADDs = 14
    const std::uint32_t replay_buf_len = (in0_tile_r_dim == 8) ? 8 : 14;
    // Finalization length (starting from offset 4):
    // m=8: 4 instructions (4 MVMULs), m=1/4: 10 instructions (4 MVMULs + 4 MOVs + 2 ELWADDs)
    // sdpa_custom_mm_reuse_dest_srcb::finalization_len = (in0_tile_r_dim == 8) ? 4 : 10;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        4,
        // Lambda function to load reply buffer
        [] {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0 (32)
            TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_3, 0);     // 16 (48)
        });
}

template <int MATH_FIDELITY_DESC>
inline void _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face = false,
    const std::uint32_t transpose = 0,
    const std::uint32_t kt_dim = 1) {
    sdpa_custom_mm_reuse_dest_srcb_configure_addrmod<MATH_FIDELITY_DESC>(
        transpose, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    sdpa_custom_mm_reuse_dest_srcb_configure_mop<MATH_FIDELITY_PHASES>(
        transpose > 0, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    // math::reset_counters(p_setrwc::SET_ABD_F);
}

// Simplified implementation: NUM_FIDELITY_PHASES = 0 (no high fidelity mode)
// Template parameter partial_acc:
//   false (default): Full sdpa_custom_mm_reuse_dest_srcb - MVMULs for all K tiles + finalization
//   true: Partial K accumulation - MVMULs only, NO finalization (for intermediate K subblocks)
template <bool partial_acc = false>
inline void _llk_math_sdpa_custom_mm_reuse_dest_srcb_(
    uint src_index,
    uint dst_index,
    [[maybe_unused]] const bool transpose = false,
    [[maybe_unused]] const std::uint32_t kt_dim = 1,
    const std::uint32_t nt_dim = 1) {
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);
    for (uint32_t i = 0; i < kt_dim; i++) {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(src_index + i);
        math::reset_counters(p_setrwc::SET_ABD_F);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 16, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 16);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 20, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 20);
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
        for (uint32_t j = 0; j < nt_dim; j++) {
            lltt::replay(ckernel::math::replay_buf_offset, 4);
        }
    }
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
