// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"
#include "tt-train/sources/ttml/metal/ops/gumbel_sample/gumbel_sample_constants.hpp"  // kGumbel* log constants, shared with the host-side invariant test

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

/**
 * Fused Gumbel scoring: one SFPI pass computing, per DST datum,
 *
 *     score = logits * inv_temperature + (-log(-log(U)))
 *
 * DST contract: the uniform tile U sits at `idst`, the logits tile LOGITS_DST_OFFSET slots above
 * it, and the score overwrites U in place. The caller's rand bounds keep U strictly inside (0, 1).
 *
 * Register discipline: the pass reads its log constants from the programmable const registers,
 * programmed by gumbel_score_tile_init(). rand_tile is the only clobberer of those registers, so
 * the init must run after the LAST rand_tile it is meant to survive -- which is what lets the
 * kernel draw a whole DST batch of noise first and cover it with one init.
 */

namespace ttml::metal::sfpu {

#ifdef TRISC_MATH

// Program the log constants once per DST batch (from gumbel_score_tile_init). sfpi exposes only
// three programmable float slots, so kGumbelPolyB -- fp16a-exact, a single-load immediate -- stays
// inline in gumbel_noise_neg_log.
inline void gumbel_score_constants_init() {
    sfpi::vConstFloatPrgm0 = kGumbelNegLn2;
    sfpi::vConstFloatPrgm1 = kGumbelPolyC;
    sfpi::vConstFloatPrgm2 = kGumbelPolyD;
}

// Approximate -ln(v): exponent split plus one quadratic over the mantissa octave, negation folded
// into the constants (both call sites want -log). Monotone, so argmax semantics hold; it does bias
// the sampled distribution slightly away from exact softmax (|q + ln| <= 5.3e-3 on the octave),
// which is the price of the speedup. The endpoint ties, monotonicity and error bound are pinned
// host-side by TestGumbelApproxLogInvariants; see gumbel_sample_constants.hpp for the re-fitting
// constraints. The shared +2^-20 shift keeps -log(U) strictly positive without a zero guard,
// capping the noise near 13.81 instead of 16.64.
sfpi_inline sfpi::vFloat gumbel_noise_neg_log(const sfpi::vFloat v) {
    const sfpi::vFloat m = sfpi::setexp(v, 127);
    const sfpi::vFloat poly = m * (m * kGumbelPolyB + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm2;
    const auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(v));
    const sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    return expf * sfpi::vConstFloatPrgm0 + poly;
}

template <std::uint32_t LOGITS_DST_OFFSET>
inline void calculate_gumbel_score(const std::uint32_t inv_temperature_bits) {
    // One DST tile is 32 sfpi rows. The 8 iterations cover ONE face: SFPU_UNARY_CALL re-invokes
    // this body per face with dst_reg re-based, so the tile-sized logits offset tracks the face.
    constexpr std::uint32_t dst_tile_size_sfpi = 32U;
    const sfpi::vFloat inv_temperature = ckernel::sfpu::Converter::as_float(inv_temperature_bits);
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat u = sfpi::dst_reg[0];
        const sfpi::vFloat logits = sfpi::dst_reg[LOGITS_DST_OFFSET * dst_tile_size_sfpi];
        const sfpi::vFloat neg_log_u = gumbel_noise_neg_log(u);
        const sfpi::vFloat gumbel = gumbel_noise_neg_log(neg_log_u);
        sfpi::vFloat score = logits * inv_temperature + gumbel;
        if constexpr (!DST_ACCUM_MODE) {
            score = sfpi::convert<sfpi::vFloat16b>(score, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = score;
        sfpi::dst_reg++;
    }
}

#endif  // TRISC_MATH

}  // namespace ttml::metal::sfpu

namespace ckernel {

#ifdef TRISC_MATH
namespace sfpu {

// SFPU_UNARY_CALL resolves its functor as ::ckernel::sfpu::FN, hence this forwarder.
template <std::uint32_t LOGITS_DST_OFFSET>
inline void _calculate_gumbel_score_(const std::uint32_t inv_temperature_bits) {
    ttml::metal::sfpu::calculate_gumbel_score<LOGITS_DST_OFFSET>(inv_temperature_bits);
}

}  // namespace sfpu
#endif  // TRISC_MATH

/**
 * @brief Initializes the fused Gumbel scoring SFPU operation.
 */
ALWI void gumbel_score_tile_init() {
    MATH(SFPU_UNARY_INIT(unused));
    // After the llk init (so nothing it resets clobbers them) and after the batch's last rand_tile
    // (which reprograms two of the three Prgm slots on Wormhole).
    MATH((ttml::metal::sfpu::gumbel_score_constants_init()));
}

/**
 * @brief Overwrites the uniform tile at `idst` with the fused Gumbel score.
 *
 * The logits tile must already sit `logits_dst_offset` DST slots above `idst`.
 * `inv_temperature_bits` is the FP32 bit pattern of 1/temperature.
 */
template <uint32_t logits_dst_offset = 1U>
ALWI void gumbel_score_tile(uint32_t idst, uint32_t inv_temperature_bits) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_gumbel_score_,
        (logits_dst_offset),
        idst,
        VectorMode::RC,
        inv_temperature_bits));
}

}  // namespace ckernel
