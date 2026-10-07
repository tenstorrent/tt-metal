// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// One SFPLOAD covers 4 face rows (even or odd columns), so four loads bring in a quad of 4 tile rows.
constexpr std::uint32_t RESHUFFLE_TILE_ROWS = TILE_R_DIM;
constexpr std::uint32_t RESHUFFLE_QUAD_ROWS = 4;
constexpr std::uint32_t RESHUFFLE_FACE_STRIDE = FACE_R_DIM;

// Quad-relative Dest slots: even/odd columns of the left face, then of the face beside it.
constexpr std::uint32_t RESHUFFLE_LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t RESHUFFLE_LEFT_ODD = p_sfpu::col_offset::ODD_COL;
constexpr std::uint32_t RESHUFFLE_RIGHT_EVEN = RESHUFFLE_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t RESHUFFLE_RIGHT_ODD = RESHUFFLE_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

constexpr std::uint32_t RESHUFFLE_OUTPUT_TILE_OFFSET =
    1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);

// Legacy tile-header bytes before the 32 mask bytes.
constexpr std::uint32_t RESHUFFLE_MASK_HEADER_BYTES = 16;

constexpr std::uint32_t RESHUFFLE_SFPMEM_DONE = 0;

// SFPTRANSP swizzles LREG0-3 and LREG4-7 independently.
constexpr std::uint32_t RESHUFFLE_IN_BANK = p_sfpu::LREG0;
constexpr std::uint32_t RESHUFFLE_OUT_BANK = p_sfpu::LREG4;

// Dest address of the quad holding `row`; rows 16-31 skip a face (map to 32, 36, 40, 44).
constexpr std::uint32_t _reshuffle_quad_addr_(const std::uint32_t row) {
    return (row & ~(RESHUFFLE_QUAD_ROWS - 1)) + (row & RESHUFFLE_FACE_STRIDE);
}

inline void _reshuffle_load_quad_(const std::uint32_t bank, const std::uint32_t quad_addr) {
    TT_SFPLOAD(
        bank + 0 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(
        bank + 1 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPLOAD(
        bank + 2 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(
        bank + 3 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_RIGHT_ODD /*dest_reg_addr*/);
}

// The bank must be back in load order (after the second SFPTRANSP).
inline void _reshuffle_store_quad_(const std::uint32_t bank, const std::uint32_t quad_addr) {
    TT_SFPSTORE(
        bank + 0 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(
        bank + 1 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPSTORE(
        bank + 2 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(
        bank + 3 /*lreg_ind*/,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        RESHUFFLE_SFPMEM_DONE /*done*/,
        quad_addr + RESHUFFLE_RIGHT_ODD /*dest_reg_addr*/);
}

// Keep the input loads ahead of the output loads: they separate the previous row's Dest stores from
// a same-quad reload, which the scoreboard does not cover (tenstorrent/tt-metal#51345).
inline void _calculate_reshuffle_rows_row_(const std::uint32_t in_row, const std::uint32_t out_row) {
    const std::uint32_t in_addr = _reshuffle_quad_addr_(in_row);
    const std::uint32_t out_addr = RESHUFFLE_OUTPUT_TILE_OFFSET + _reshuffle_quad_addr_(out_row);
    const std::uint32_t in_reg = RESHUFFLE_IN_BANK + (in_row & (RESHUFFLE_QUAD_ROWS - 1));
    const std::uint32_t out_reg = RESHUFFLE_OUT_BANK + (out_row & (RESHUFFLE_QUAD_ROWS - 1));

    _reshuffle_load_quad_(RESHUFFLE_IN_BANK, in_addr);
    _reshuffle_load_quad_(RESHUFFLE_OUT_BANK, out_addr);

    // Now LREG j holds input row quad+j and LREG 4+j output row quad+j.
    TTI_SFPTRANSP;

    // out = in * 1.0 + out; no SFPNOP needed, the dependent SFPTRANSP is interlocked.
    TT_SFPADD(in_reg /*lreg_a*/, p_sfpu::LCONST_1, out_reg /*lreg_c*/, out_reg /*lreg_dest*/, 0 /*instr_mod1*/);

    TTI_SFPTRANSP;

    _reshuffle_store_quad_(RESHUFFLE_OUT_BANK, out_addr);
}

/**
 * @brief Add each row i of tile idst into row mask[i] of tile idst+1; mask entries >= 32 are skipped.
 *
 * @param idx_addr: L1 address of the mask minus RESHUFFLE_MASK_HEADER_BYTES.
 * @note Once per tile (VectorMode::RC_custom); the caller must ensure idst+1 is a valid Dest tile.
 */
template <bool APPROXIMATION_MODE /*unused*/>
inline void calculate_reshuffle_rows(const std::uint32_t idx_addr) {
    volatile tt_l1_ptr std::uint8_t* mask =
        reinterpret_cast<volatile tt_l1_ptr std::uint8_t*>(idx_addr + RESHUFFLE_MASK_HEADER_BYTES);

    for (std::uint32_t in_row = 0; in_row < RESHUFFLE_TILE_ROWS; in_row++) {
        const std::uint32_t out_row = static_cast<std::uint32_t>(mask[in_row]);
        if (out_row >= RESHUFFLE_TILE_ROWS) {
            continue;
        }
        _calculate_reshuffle_rows_row_(in_row, out_row);
    }
}

}  // namespace sfpu
}  // namespace ckernel
