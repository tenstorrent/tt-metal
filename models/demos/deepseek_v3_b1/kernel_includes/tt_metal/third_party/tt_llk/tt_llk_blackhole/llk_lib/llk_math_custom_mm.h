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
// m=1, m=4: 10 instructions (4 MVMUL + 4 MOV + 2 ELWADD) for 32-wide
// m=1, m=4: 5 instructions (2 MVMUL + 2 MOV + 1 ELWADD) for 16-wide
// m=8: 4 instructions (4 MVMUL only, no tail reduction) for 32-wide
// m=8: 2 instructions (2 MVMUL only, no tail reduction) for 16-wide
inline std::uint32_t finalization_len = 10;

// Accumulation instruction count per K-tile: set during init
// 4 MVMULs for 32-wide (4 faces), 2 MVMULs for 16-wide (2 faces)
inline std::uint32_t accum_len = 4;

// Output tile column dimension: set during init, used during math for DstTileShape selection
// Supports 32 (Tile32x32) and 16 (Tile32x16)
inline std::uint32_t out_tile_c_dim = 32;
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
    const bool narrow_tile = (in1_tile_c_dim == 16);

    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // ADDR_MOD_1: Controls srcA, srcB, and dest increments between faces
    // For 16-wide tiles (narrow_tile): dest stays at 0 because both MVMULs
    // contribute to the same spatial output location (they accumulate directly)
    // For 32-wide tiles: dest increments to write to different spatial faces
    if (narrow_tile) {
        // 16-wide: srcA and srcB increment for next face, dest STAYS at 0
        // Both MVMULs accumulate their results to the same dest location
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 16, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    } else if (in0_tile_r_dim == 8) {
        // 32-wide m=8: dest clears after each MVMUL
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 16, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 1, .cr = 0},
        }
            .set(ADDR_MOD_1);
    } else {
        // 32-wide m=1/4: dest increments to different spatial faces
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
    // Determine number of faces based on tile column dimension
    // 32-wide: 4 faces, 16-wide: 2 faces
    const bool narrow_tile = (in1_tile_c_dim == 16);

    // Select MOV instruction count based on output tile height (in0_tile_r_dim)
    // and tile width (in1_tile_c_dim)
    // Replay buffer layout for 32-wide:
    //   m=8: 8 MVMULs
    //   m=1/4: 8 MVMULs + 4 MOVs + 2 ELWADDs = 14
    // Replay buffer layout for 16-wide:
    //   All m values: 4 MVMULs only (no tail reduction needed because both
    //   MVMULs accumulate directly to dest[0] via ADDR_MOD_1 with dest.incr=0)
    std::uint32_t replay_buf_len;
    if (narrow_tile) {
        // 16-wide: no tail reduction needed for any m value
        replay_buf_len = 4;
        custom_mm::accum_len = 2;
        custom_mm::finalization_len = 2;
    } else {
        replay_buf_len = (in0_tile_r_dim == 8) ? 8 : 14;
        // Accumulation: 4 MVMULs for 4 faces
        custom_mm::accum_len = 4;
        // Finalization (starting from offset accum_len):
        // m=8: 4 MVMULs, m=1/4: 10 instructions (4 MVMUL + 4 MOV + 2 ELWADD)
        custom_mm::finalization_len = (in0_tile_r_dim == 8) ? 4 : 10;
    }

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [in0_tile_r_dim, narrow_tile] {
            if (narrow_tile) {
                // 16-wide tile: 2 faces (B[0:15,:] and B[16:31,:])
                // Key insight: For A[M,32] x B[32,16], both MVMULs contribute to the
                // SAME spatial output location (row 0, cols 0-15). They should accumulate
                // directly to dest[0]. ADDR_MOD_1 is configured with dest.incr=0 so both
                // MVMULs write to dest[0] and accumulate their results.
                // No tail reduction needed for any m value.
                //
                // Layout:
                //   Face 0: A_left[0:15] x B[0:15,:] at srcB=0, srcA=0, dest=0
                //   Face 1: A_right[16:31] x B[16:31,:] at srcB=16, srcA=16, dest=0 (accumulates!)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // face 0: srcA=0, srcB=0, dest=0; incr srcA/srcB only
                TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // face 1: srcA=16, srcB=16, dest=0; then clear all
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // finalization face 0
                TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // finalization face 1
            } else {
                // 32-wide tile: 4 faces (original implementation)
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

    // Store output tile column dimension for use in _llk_math_custom_mm_
    // in1_tile_c_dim is the column dimension of B matrix, which becomes output column dimension
    custom_mm::out_tile_c_dim = in1_tile_c_dim;

    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Simplified implementation: NUM_FIDELITY_PHASES = 0 (no high fidelity mode)
// Template parameter partial_acc:
//   false (default): Full custom_mm - MVMULs for all K tiles + finalization
//   true: Partial K accumulation - MVMULs only, NO finalization (for intermediate K subblocks)
template <bool partial_acc = false>
inline void _llk_math_custom_mm_(
    uint dst_index, [[maybe_unused]] const bool transpose = false, [[maybe_unused]] const std::uint32_t kt_dim = 1) {
    // Select path based on output tile column dimension (set during init)
    // All lltt::replay arguments must be compile-time constants
    if (custom_mm::out_tile_c_dim == 16) {
        // 16-wide tile path (2 faces)
        // Both MVMULs accumulate to dest[0] directly (no tail reduction needed)
        // NOTE: Use Tile32x32 DST addressing even for 16-wide tiles because
        // pack_tile doesn't support narrow_tile DST addressing (it always assumes 64 datums per tile)
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        if constexpr (partial_acc) {
            // Partial K accumulation: 2 MVMULs per K tile
            for (uint32_t i = 0; i < kt_dim; i++) {
                lltt::replay(ckernel::math::replay_buf_offset, 2);
            }
        } else {
            // Full accumulation with finalization
            for (uint32_t i = 0; i < kt_dim - 1; i++) {
                lltt::replay(ckernel::math::replay_buf_offset, 2);
            }
            // Final K tile + finalization: always 2 MVMULs (no tail reduction for 16-wide)
            lltt::replay(ckernel::math::replay_buf_offset + 2, 2);
        }
    } else {
        // 32-wide tile path (4 faces, original implementation)
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        if constexpr (partial_acc) {
            // Partial K accumulation: 4 MVMULs per K tile
            for (uint32_t i = 0; i < kt_dim; i++) {
                lltt::replay(ckernel::math::replay_buf_offset, 4);
            }
        } else {
            // Full accumulation with finalization
            for (uint32_t i = 0; i < kt_dim - 1; i++) {
                lltt::replay(ckernel::math::replay_buf_offset, 4);
            }
            // Final K tile + finalization: use finalization_len to select path
            // m=8: finalization_len=4, m=1/4: finalization_len=10
            if (custom_mm::finalization_len == 4) {
                lltt::replay(ckernel::math::replay_buf_offset + 4, 4);
            } else {
                lltt::replay(ckernel::math::replay_buf_offset + 4, 10);
            }
        }
    }
}
