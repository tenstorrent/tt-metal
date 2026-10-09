// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

constexpr std::uint32_t SFPSETSGN_MOD1_ARG_IMM = 1;   // sign bit taken from imm12[0]
constexpr std::uint32_t SFPSETSGN_SIGN_POSITIVE = 0;  // imm12[0] = 0 -> sign bit cleared
// probability is compared signed (SFPIADD), so p = 1 is INT_MAX.
constexpr std::uint32_t DROPOUT_PROBABILITY_MAX = 0x7FFFFFFF;

// out = (rand31 <= probability) ? 0 : x * scale, for one row pair. Expects LREG1 = scale, LREG2 = probability.
// The PRNG read honours LaneEnable, so it must stay outside the CC-narrowed region.
inline void _calculate_dropout_sfp_rows_() {
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);
    TTI_SFPMOV(sfpi::SFPCONFIG_SRC_RAND, p_sfpu::LREG3, sfpi::SFPMOV_MOD1_CONFIG);  // per-lane random, steps PRNG
    TTI_SFPSETSGN(SFPSETSGN_SIGN_POSITIVE, p_sfpu::LREG3, p_sfpu::LREG3, SFPSETSGN_MOD1_ARG_IMM);  // rand &= 0x7FFFFFFF
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpiadd::MOD1_SUB_CC_GTE0);  // CC = (prob - rand >= 0)
    TTI_SFPMOV(p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1 */);
    TTI_SFPSTORE(
        p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /* done */, 0 /* dest_reg */);  // Dest += SFP_ROWS
}

// p = 0 body: out = x * scale, no PRNG compare.
inline void _calculate_dropout_scale_sfp_rows_() {
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);
    TTI_SFPSTORE(
        p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /* done */, 0 /* dest_reg */);  // Dest += SFP_ROWS
}

inline void _dropout_load_u32_(const std::uint32_t lreg, const std::uint32_t value) {
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF /* imm16 */);
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, value >> 16 /* imm16 */);
}

/**
 * @brief Dropout over one face of Dest; probability = p * INT_MAX, scale = fp32 bits (normally 1 / (1 - p)).
 * @note Requires @ref dropout_init. Inline body, no replay, so other ops' recorded replays stay valid.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_dropout(const std::uint32_t probability, const std::uint32_t scale) {
    LLK_ASSERT(
        probability <= DROPOUT_PROBABILITY_MAX,
        "dropout: probability is p * INT_MAX and is compared signed; bit 31 must be clear");
    _dropout_load_u32_(p_sfpu::LREG1, scale);
    if (probability == 0) {
        // The compare drops rand == 0 even at p = 0, so p = 0 skips it to keep every datum.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            _calculate_dropout_scale_sfp_rows_();
        }
        return;
    }
    _dropout_load_u32_(p_sfpu::LREG2, probability);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_dropout_sfp_rows_();
    }
}

/**
 * @brief Seed the PRNG and program ADDR_MOD_6 (Dest += SFP_ROWS on store).
 * @note Other SFPU inits that write ADDR_MOD_6 must run before this one.
 */
template <bool APPROXIMATION_MODE>
inline void dropout_init(const std::uint32_t seed) {
    init_prng_seed(seed);
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = ckernel::math::SFP_ROWS},
    }
        .set(ADDR_MOD_6);
}

}  // namespace sfpu
}  // namespace ckernel
