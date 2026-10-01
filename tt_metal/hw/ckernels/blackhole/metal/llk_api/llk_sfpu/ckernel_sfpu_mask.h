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

// The bodies take the data and the mask tile as absolute DEST tile indices and write the result in place. The five
// instructions of the first row are recorded into replay slots 0 to 4 while they execute and replayed for the rest.
namespace mask_detail {
// One DEST tile is 64 rows for the SFPU load and store address field, in both DEST widths.
constexpr std::uint32_t dst_tile_rows = 64;
}  // namespace mask_detail

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_mask, const std::uint32_t /* dst_index_out */) {
    const std::uint32_t data_addr = dst_index_data * mask_detail::dst_tile_rows;
    const std::uint32_t mask_addr = dst_index_mask * mask_detail::dst_tile_rows;
    TTI_REPLAY(0, 5, 1, 1);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, mask_addr);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, data_addr);
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
    TTI_REPLAY(0, 5, 1, 1);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, mask_addr);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, data_addr);
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
    TTI_REPLAY(0, 5, 1, 1);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, mask_addr);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, data_addr);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_REPLAY(0, 5, 0, 0);
    }
}

}  // namespace sfpu
}  // namespace ckernel
