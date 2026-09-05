// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Configure the SFPU address mode used by the square op.
 *
 * Programs ADDR_MOD_6 with a dest increment of 2 (one SFPU pass writes 2 rows on Quasar).
 *
 * @note Call this before @ref calculate_square to set up the address mode it relies on.
 */
inline void init_square() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
}

/**
 * @brief Square one SFPU pass worth of rows (Quasar = 2 rows): dest = x * x.
 *
 * Loads x from dest, multiplies it by itself, and stores the result back to dest using
 * ADDR_MOD_6 to advance to the next pair of rows.
 *
 * @note ADDR_MOD_6 must already be programmed by @ref init_square.
 */
inline void _calculate_square_sfp_rows_() {
    sfpi::vFloat v = sfpi::dst_reg[0];                     // load x from dest (SFPLOAD)
    sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = v * v;  // x * x via SFPMUL, store back to dest (SFPSTORE)
}

/**
 * @brief Square a full Dest tile in place: dest = x * x.
 *
 * @tparam ITERATIONS: Number of SFPU passes (each covers 2 rows) needed to span the tile.
 * @note Call @ref init_square before this to program the address mode it depends on.
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_square() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_square_sfp_rows_();
    }
}

// Squares one pair of rows (Quasar SFPU ops cover 2 rows)
inline void _calculate_square_rows_(
    const int load_addr, const int store_addr, const std::uint32_t load_sfpmem, const std::uint32_t store_sfpmem) {
    TT_SFPLOAD(p_sfpu::LREG0, load_sfpmem, ADDR_MOD_7, 0, load_addr);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TT_SFPSTORE(p_sfpu::LREG0, store_sfpmem, ADDR_MOD_7, 0, store_addr);
}

// Addresses select Dest (bit 10 = 0) or SrcS (bit 10 = 1). Float16 needs an explicit FP16A.
inline void _calculate_square_(
    const int load_base_addr,
    const int store_base_addr,
    const int num_sfpu_iterations,
    const std::uint32_t load_sfpmem,
    const std::uint32_t store_sfpmem) {
#pragma GCC unroll 8
    for (int d = 0; d < num_sfpu_iterations; d++) {
        _calculate_square_rows_(load_base_addr + (d << 1), store_base_addr + (d << 1), load_sfpmem, store_sfpmem);
    }
}

}  // namespace sfpu
}  // namespace ckernel
