// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void mask_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// The three bodies below take the data and the mask tile as absolute DEST tile indices, as the binary SFPU
// params frame delivers them: the frame starts at tile 0 and the DEST counter carries the face offset, so the
// mask tile can sit anywhere in the acquired DEST, before or after the data tile. The result is written in
// place, so dst_index_out is unused. Each row: load the mask vector, set the condition code where it is zero,
// store the replacement under that condition, clear the condition, step to the next row. This is the
// instruction stream the compiler emitted for the former sfpi bodies, written with runtime DEST addresses so
// that the mask index is honoured (the sfpi form addressed the mask at a fixed one tile after the data). As the
// compiler did, the five instructions of the first row are recorded into replay slots 0 to 4 while they execute
// and replayed for the remaining rows of the face, so the RISC issues one instruction per row instead of five.
namespace mask_detail {
// One DEST tile is 64 rows for the SFPU load and store address field, in both DEST widths.
constexpr std::uint32_t dst_tile_rows = 64;
}  // namespace mask_detail

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_mask, const std::uint32_t /* dst_index_out */) {
    const std::uint32_t data_addr = dst_index_data * mask_detail::dst_tile_rows;
    const std::uint32_t mask_addr = dst_index_mask * mask_detail::dst_tile_rows;
    // Record the row into replay slots 0 to 4 while executing it, then replay it for the remaining rows.
    TTI_REPLAY(0, 5, 1, 1);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, mask_addr);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, data_addr);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_REPLAY(0, 5, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_int_mask(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_mask, const std::uint32_t /* dst_index_out */) {
    const std::uint32_t data_addr = dst_index_data * mask_detail::dst_tile_rows;
    const std::uint32_t mask_addr = dst_index_mask * mask_detail::dst_tile_rows;
    // Record the row into replay slots 0 to 4 while executing it, then replay it for the remaining rows.
    TTI_REPLAY(0, 5, 1, 1);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, mask_addr);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, data_addr);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_REPLAY(0, 5, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask_posinf(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_mask, const std::uint32_t /* dst_index_out */) {
    const std::uint32_t data_addr = dst_index_data * mask_detail::dst_tile_rows;
    const std::uint32_t mask_addr = dst_index_mask * mask_detail::dst_tile_rows;
    // +infinity as a bf16 immediate in the high half of the register.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x7F80);
    // Record the row into replay slots 0 to 4 while executing it, then replay it for the remaining rows.
    TTI_REPLAY(0, 5, 1, 1);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, mask_addr);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, data_addr);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_REPLAY(0, 5, 0, 0);
    }
}

}  // namespace sfpu
}  // namespace ckernel
