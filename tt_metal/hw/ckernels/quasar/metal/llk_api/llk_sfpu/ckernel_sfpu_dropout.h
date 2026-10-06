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

constexpr std::uint32_t SFPSETSGN_MOD1_ARG_IMM = 1;   // sign bit taken from imm12[0]
constexpr std::uint32_t SFPSETSGN_SIGN_POSITIVE = 0;  // imm12[0] = 0 -> sign bit cleared
constexpr std::uint32_t PRNG_SEED_WAIT_NOPS = 1024;   // SFPNOPs for the PRNG seeder to finish after the seed write
constexpr std::uint32_t DROPOUT_REPLAY_SLOT = 0;
constexpr std::uint32_t DROPOUT_REPLAY_LEN = 8;  // SFP instructions in _calculate_dropout_sfp_rows_

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

/**
 * @brief Apply dropout in place over one face of Dest.
 *
 * @tparam APPROXIMATION_MODE: Accepted for Compute API parity; dropout has no approximate variant.
 * @tparam ITERATIONS: SFPU row-pair iterations covering one face.
 * @param probability: Drop probability scaled to an integer, p * INT_MAX, range 0 .. 0x7FFFFFFF.
 * @param scale: fp32 bit pattern of the scale applied to surviving data (normally 1 / (1 - p)).
 * @note Call @ref dropout_init once before this to seed the PRNG.
 * @note Overwrites LREG0-LREG3 and replay slots [DROPOUT_REPLAY_SLOT, +DROPOUT_REPLAY_LEN).
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_dropout(const std::uint32_t probability, const std::uint32_t scale) {
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, probability >> 16);
    load_replay_buf<DROPOUT_REPLAY_SLOT, DROPOUT_REPLAY_LEN>([] { _calculate_dropout_sfp_rows_(); });
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_REPLAY(
            DROPOUT_REPLAY_SLOT,
            DROPOUT_REPLAY_LEN,
            0 /* last */,
            0 /* set_mutex */,
            0 /* execute_while_loading */,
            0 /* load_mode */);
        // dest_reg++, by the two Dest rows the replay just consumed
        ckernel::math::_incr_counters_<0x0 /* srca */, 0x0 /* srcb */, ckernel::math::SFP_ROWS, 0x0 /* cr */>();
    }
}

/**
 * @brief Seed the hardware PRNG that feeds the SFPU lanes.
 *
 * @tparam APPROXIMATION_MODE: Accepted for Compute API parity; unused.
 * @param seed: Seed value handed to the Tensix PRNG seeder.
 * @note Call once before @ref calculate_dropout. The seeder exposes no completion flag, so the
 *       SFPNOP wait is what guarantees the first PRNG read sees seeded lanes.
 * @note Do not place a TRISC_CFG STALLWAIT near this MMIO cfg write (errata TEN-4849).
 */
template <bool APPROXIMATION_MODE>
inline void dropout_init(const std::uint32_t seed) {
    ckernel::trisc::cfg[PRNG_SEED_Seed_Val_ADDR32] = seed;  // kicks the PRNG seeder
    for (std::uint32_t i = 0; i < PRNG_SEED_WAIT_NOPS; i++) {
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    }
}

}  // namespace sfpu
}  // namespace ckernel
