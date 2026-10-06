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
// probability is compared as a signed int32 (SFPIADD MOD1_SUB_CC_GTE0), so p = 1 is INT_MAX.
constexpr std::uint32_t DROPOUT_PROBABILITY_MAX = 0x7FFFFFFF;

/**
 * @brief Dropout body for one SFPU row pair (Quasar = 2 Dest rows):
 *        out = (rand31 <= probability) ? 0.0f : x * scale.
 *
 * @note Load LREG1 = scale and LREG2 = probability before this runs; @ref calculate_dropout does that.
 * @note The PRNG read advances the per-lane PRNG and honours LaneEnable, so it must stay outside any
 *       CC-narrowed region.
 */
inline void _calculate_dropout_sfp_rows_() {
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);  // x from dest
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);          // x * scale
    TTI_SFPMOV(sfpi::SFPCONFIG_SRC_RAND, p_sfpu::LREG3, sfpi::SFPMOV_MOD1_CONFIG);  // per-lane random, steps PRNG
    TTI_SFPSETSGN(SFPSETSGN_SIGN_POSITIVE, p_sfpu::LREG3, p_sfpu::LREG3, SFPSETSGN_MOD1_ARG_IMM);  // rand &= 0x7FFFFFFF
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpiadd::MOD1_SUB_CC_GTE0);  // CC = (prob - rand >= 0)
    TTI_SFPMOV(p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);                              // dropped lanes -> 0.0
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1 */);                                               // re-enable all lanes
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);  // store result
}

// Loads a 32-bit runtime value into an LREG as two 16-bit halves.
inline void _dropout_load_u32_(const std::uint32_t lreg, const std::uint32_t value) {
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF /* imm16 */);
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, value >> 16 /* imm16 */);
}

/**
 * @brief Apply dropout in place over one face of Dest.
 *
 * @tparam APPROXIMATION_MODE: Accepted for Compute API parity; dropout has no approximate variant.
 * @tparam ITERATIONS: SFPU row-pair iterations covering one face.
 * @param probability: Drop probability scaled to an integer, p * INT_MAX, range 0 .. DROPOUT_PROBABILITY_MAX.
 * @param scale: fp32 bit pattern of the scale applied to surviving data (normally 1 / (1 - p)).
 * @note Call @ref dropout_init once before this to seed the PRNG.
 * @note Overwrites LREG0-LREG3. Issues its body inline (no replay buffer), so replay-backed math
 *       ops recorded in their init stay valid across dropout calls.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_dropout(const std::uint32_t probability, const std::uint32_t scale) {
    LLK_ASSERT(
        probability <= DROPOUT_PROBABILITY_MAX,
        "dropout: probability is p * INT_MAX and is compared signed; bit 31 must be clear");
    _dropout_load_u32_(p_sfpu::LREG1, scale);
    _dropout_load_u32_(p_sfpu::LREG2, probability);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_dropout_sfp_rows_();
        // dest_reg++, by the two Dest rows just consumed
        ckernel::math::_incr_counters_<0x0 /* srca */, 0x0 /* srcb */, ckernel::math::SFP_ROWS, 0x0 /* cr */>();
    }
}

/**
 * @brief Seed the hardware PRNG that feeds the SFPU lanes.
 *
 * @tparam APPROXIMATION_MODE: Accepted for Compute API parity; unused.
 * @param seed: Seed value handed to the Tensix PRNG seeder.
 * @note Call once before @ref calculate_dropout.
 */
template <bool APPROXIMATION_MODE>
inline void dropout_init(const std::uint32_t seed) {
    init_prng_seed(seed);
}

}  // namespace sfpu
}  // namespace ckernel
