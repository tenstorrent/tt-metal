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

namespace custom_mm {
// Finalization instruction count: set during init, used during math
// m=1, m=4: 10 instructions (4 MVMUL + 4 MOV + 2 ELWADD)
// m=8: 4 instructions (4 MVMUL only, no tail reduction)
inline std::uint32_t finalization_len = 10;
}  // namespace custom_mm

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

    if (in0_tile_r_dim == 8) {
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 16, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 1, .cr = 0},
        }
            .set(ADDR_MOD_1);
    } else {
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 16, .clr = 0, .cr = 0},
            .dest = {.incr = 16, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    }

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
    // Select MOV instruction count based on output tile height (in0_tile_r_dim)
    // m=1/4: 4 MOVs (MOV_4_ROWS), m=8: no tail reduction
    // Replay buffer size: m=8: 8 MVMULs, m=1/4: 8 MVMULs + 4 MOVs + 2 ELWADDs = 14
    const std::uint32_t replay_buf_len = (in0_tile_r_dim == 8) ? 8 : 14;
    // Finalization length (starting from offset 4):
    // m=8: 4 instructions (4 MVMULs), m=1/4: 10 instructions (4 MVMULs + 4 MOVs + 2 ELWADDs)
    custom_mm::finalization_len = (in0_tile_r_dim == 8) ? 4 : 10;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [in0_tile_r_dim] {
            if (in0_tile_r_dim == 8) {
                // m=8: 8 MVMULs only, no tail reduction
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0 (32)
                TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // 16 (48)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0 (32)
                TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // 16 (48)
            } else {
                // m=1/4: 8 MVMULs + MOVs + ELWADDs
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0 (32)
                TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // 16 (48)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0 (32)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);  // 16 (48)
                // m=1/4: MOV_4_ROWS
                TTI_MOVD2A(0, 0, ADDR_MOD_3, p_movd2a::MOV_4_ROWS, 0);
                TTI_MOVD2A(0, 16, ADDR_MOD_3, p_movd2a::MOV_4_ROWS, 16);
                TTI_MOVD2B(0, 0, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 32);
                TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 48);
                TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
                TTI_ELWADD(3, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
            }
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
// Template parameter partial_acc:
//   false (default): Full custom_mm - MVMULs for all K tiles + finalization
//   true: Partial K accumulation - MVMULs only, NO finalization (for intermediate K subblocks)
template <bool partial_acc = false>
inline void _llk_math_custom_mm_(
    uint dst_index, [[maybe_unused]] const bool transpose = false, [[maybe_unused]] const std::uint32_t kt_dim = 1) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (partial_acc) {
        // Partial K accumulation: run all kt_dim iterations with first 4 MVMULs
        // MVMUL accumulates into dest, results persist for next K subblock
        // NO finalization - skip instructions 4+
        for (uint32_t i = 0; i < kt_dim; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, 4);
        }
    } else {
        // Full accumulation with finalization
        // First kt_dim-1 iterations: 4 MVMULs (instructions 0-3)
        for (uint32_t i = 0; i < kt_dim - 1; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, 4);
        }
        // Final K tile + finalization (instructions 4 to 4+finalization_len)
        // m=8: 4 MVMULs, m=1/4: 4 MVMULs + MOVs + ELWADDs
        lltt::replay(ckernel::math::replay_buf_offset + 4, custom_mm::finalization_len);
    }
}
