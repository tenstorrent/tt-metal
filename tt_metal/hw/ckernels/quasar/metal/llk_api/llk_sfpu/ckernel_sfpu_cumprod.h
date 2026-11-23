// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_cumsum.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Column-wise (top-to-bottom) cumulative product of one whole 32x32 Dest tile, in place.
 *
 * The cumsum tile walk (@ref calculate_cumsum) replaying the product chain @ref cumprod_init
 * recorded: each quad's four rows are chained with SFPMUL instead of SFPADD, in fp32, with one
 * rounding on the store.
 *
 * @param first: Whether this is the first tile of a top-to-bottom chain; seeds the carry bank with 1.0.
 * @note Run this once per tile under VectorMode::RC_custom, not once per face. As for cumsum,
 *       LREG4-7 carry this tile's 32 column products to the next call: feed tiles top-to-bottom and
 *       write nothing to LREG4-7 in between.
 * @note Call @ref cumprod_init before this, and again after any op that reprograms ADDR_MOD_6 or
 *       records into replay bank 1 (cumsum and EMA do).
 */
template <bool APPROXIMATION_MODE /*unused*/>
inline void calculate_cumprod(const bool first) {
    if (first) {
        // Seed the whole carry bank, not just its last register: SFPTRANSP swizzles lanes across the
        // entire bank, so any other value anywhere in it would surface in the carry's lanes.
        TTI_SFPMOV(p_sfpu::LCONST_1, CUMSUM_LREG_BANK_B + 0, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_1, CUMSUM_LREG_BANK_B + 1, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_1, CUMSUM_LREG_BANK_B + 2, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_1, CUMSUM_LREG_BANK_B + 3, 0 /* instr_mod1: plain copy */);
    }

    _scan_tile_walk_();
}

/**
 * @brief Configure the SFPU state the cumprod tile walk depends on: the RWC counters, ADDR_MOD_6 and
 *        the product body recorded into replay bank 1.
 *
 * @note Call this before @ref calculate_cumprod, and again before resuming cumprod after any op that
 *       reprograms ADDR_MOD_6 or records into replay bank 1 (cumsum and EMA do).
 */
template <bool APPROXIMATION_MODE /*unused*/>
inline void cumprod_init() {
    _scan_init_<true /*PRODUCT*/>();
}

}  // namespace sfpu
}  // namespace ckernel
