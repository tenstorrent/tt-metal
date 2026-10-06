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

// Dest geometry of a 32x32 tile as the SFPU addresses it: 64 addr units of 16 datums, one per face
// row, with face f holding units 16f to 16f+15. An SFPLOAD covers the four units of [addr & ~3, +3]
// and, by address bit 1, either the even or the odd 8 datums of each. So one bank's four loads bring
// in four whole tile rows - the row quad this kernel addresses Dest in.
constexpr std::uint32_t RESHUFFLE_TILE_ROWS = TILE_R_DIM;
constexpr std::uint32_t RESHUFFLE_QUAD_ROWS = 4;
constexpr std::uint32_t RESHUFFLE_FACE_STRIDE = FACE_R_DIM;

// The four Dest slots one LREG bank covers, relative to a quad base: both column parities of the
// left face of a face pair, then both of the face beside it.
constexpr std::uint32_t RESHUFFLE_LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t RESHUFFLE_LEFT_ODD = p_sfpu::col_offset::ODD_COL;
constexpr std::uint32_t RESHUFFLE_RIGHT_EVEN = RESHUFFLE_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t RESHUFFLE_RIGHT_ODD = RESHUFFLE_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

// The accumulator is the Dest tile right after the input tile.
constexpr std::uint32_t RESHUFFLE_OUTPUT_TILE_OFFSET =
    1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);

// Legacy tile-header bytes that precede the 32 destination-row mask bytes.
constexpr std::uint32_t RESHUFFLE_MASK_HEADER_BYTES = 16;

// Input quad lives in LREG0-3, output quad in LREG4-7: the two banks SFPTRANSP swizzles independently.
constexpr std::uint32_t RESHUFFLE_IN_BANK = p_sfpu::LREG0;
constexpr std::uint32_t RESHUFFLE_OUT_BANK = p_sfpu::LREG4;

/**
 * @brief Dest address of the row quad holding tile row `row`.
 *
 * Rows 0-15 map to 0, 4, 8, 12 and rows 16-31 to 32, 36, 40, 44 - the second face pair starts a
 * whole face further on than the row number alone suggests.
 *
 * @param row: Tile row, 0 to RESHUFFLE_TILE_ROWS-1.
 */
constexpr std::uint32_t _reshuffle_quad_addr_(const std::uint32_t row) {
    return (row & ~(RESHUFFLE_QUAD_ROWS - 1)) + (row & RESHUFFLE_FACE_STRIDE);
}

/**
 * @brief Load the row quad at Dest address `quad_addr` into LREG bank `bank`.
 *
 * @param bank: First LREG of the bank, values = <RESHUFFLE_IN_BANK/RESHUFFLE_OUT_BANK>
 * @param quad_addr: Dest address of the quad, from @ref _reshuffle_quad_addr_.
 */
inline void _reshuffle_load_quad_(const std::uint32_t bank, const std::uint32_t quad_addr) {
    TT_SFPLOAD(bank + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_LEFT_EVEN);
    TT_SFPLOAD(bank + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_LEFT_ODD);
    TT_SFPLOAD(bank + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_RIGHT_EVEN);
    TT_SFPLOAD(bank + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_RIGHT_ODD);
}

/**
 * @brief Write LREG bank `bank` back to the row quad at Dest address `quad_addr`.
 *
 * @param bank: First LREG of the bank, values = <RESHUFFLE_IN_BANK/RESHUFFLE_OUT_BANK>
 * @param quad_addr: Dest address of the quad, from @ref _reshuffle_quad_addr_.
 * @note The bank must be in load order, i.e. post-involution - see @ref _calculate_reshuffle_rows_row_.
 */
inline void _reshuffle_store_quad_(const std::uint32_t bank, const std::uint32_t quad_addr) {
    TT_SFPSTORE(bank + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_LEFT_EVEN);
    TT_SFPSTORE(bank + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_LEFT_ODD);
    TT_SFPSTORE(bank + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_RIGHT_EVEN);
    TT_SFPSTORE(bank + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, quad_addr + RESHUFFLE_RIGHT_ODD);
}

/**
 * @brief Reset the RWC counters so the tile-relative Dest immediates start at the tile base.
 *
 * @note Call this before @ref calculate_reshuffle_rows. On Quasar the generic
 *       @ref _llk_math_sfpu_init_ already resets the same counters, so this exists for parity with
 *       the Blackhole/Wormhole API name that metal wires through SFPU_UNARY_INIT.
 */
inline void reshuffle_rows_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

/**
 * @brief Accumulate input tile row `in_row` into accumulator tile row `out_row`.
 *
 * Loads the quad containing each row into its own LREG bank, then SFPTRANSP redistributes both banks
 * so one register per bank holds a whole tile row. A single SFPADD accumulates that row, the second
 * SFPTRANSP returns both banks to store order - restoring the input bank bit-exactly, since the
 * transpose is an involution - and only the output quad is written back.
 *
 * @param in_row: Row of the input tile to read, 0 to RESHUFFLE_TILE_ROWS-1.
 * @param out_row: Row of the accumulator tile to add it into, 0 to RESHUFFLE_TILE_ROWS-1.
 * @note Keep the input quad's loads ahead of the output quad's. A Dest SFPSTORE takes three cycles
 *       to land and the scoreboard does not cover Dest (tenstorrent/tt-metal#51345), so when
 *       consecutive rows target the same output quad this ordering is what separates the previous
 *       row's stores from this row's reload of the same addresses.
 */
inline void _calculate_reshuffle_rows_row_(const std::uint32_t in_row, const std::uint32_t out_row) {
    const std::uint32_t in_addr = _reshuffle_quad_addr_(in_row);
    const std::uint32_t out_addr = RESHUFFLE_OUTPUT_TILE_OFFSET + _reshuffle_quad_addr_(out_row);
    const std::uint32_t in_reg = RESHUFFLE_IN_BANK + (in_row & (RESHUFFLE_QUAD_ROWS - 1));
    const std::uint32_t out_reg = RESHUFFLE_OUT_BANK + (out_row & (RESHUFFLE_QUAD_ROWS - 1));

    _reshuffle_load_quad_(RESHUFFLE_IN_BANK, in_addr);
    _reshuffle_load_quad_(RESHUFFLE_OUT_BANK, out_addr);

    // Each LREG now holds one whole tile row: LREG j = input row quad+j, LREG 4+j = output row quad+j
    TTI_SFPTRANSP;

    // SFPADD is dest = a*b + c, so b = 1.0 makes it out = in + out. No SFPNOP before the dependent
    // SFPTRANSP - the hardware interlocks a dependent consumer of a 2-cycle MAD.
    TT_SFPADD(in_reg, p_sfpu::LCONST_1, out_reg, out_reg, 0 /* instr_mod1: no negation */);

    TTI_SFPTRANSP;

    _reshuffle_store_quad_(RESHUFFLE_OUT_BANK, out_addr);
}

/**
 * @brief Row-wise scatter-add of one Dest tile into the tile after it, driven by an L1 row mask.
 *
 * For every input row i of tile idst, accumulates it into row mask[i] of tile idst+1. Rows whose
 * mask entry falls outside the tile (the sentinel is 255) are skipped, and the input tile is left
 * unchanged. This is the SFPU core of embedding-backward gradient accumulation, so several input
 * rows may target the same output row; the rows are processed in order and each one accumulates.
 *
 * @tparam APPROXIMATION_MODE: Unused - a single exact FP add has no approximate variant.
 * @param idx_addr: L1 byte address the 32 uint8 mask entries follow, RESHUFFLE_MASK_HEADER_BYTES on.
 * @note Run this once per tile under VectorMode::RC_custom, not once per face - one call walks all
 *       RESHUFFLE_TILE_ROWS rows. It needs idst+1 to be a valid Dest tile, which the wrapper's
 *       _sfpu_check_ does not validate, and the caller packs idst+1 rather than idst.
 * @note Call @ref reshuffle_rows_init before this. The mask is read on the math RISC as plain
 *       volatile bytes, so ordering it against whoever wrote L1 is the caller's responsibility.
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
