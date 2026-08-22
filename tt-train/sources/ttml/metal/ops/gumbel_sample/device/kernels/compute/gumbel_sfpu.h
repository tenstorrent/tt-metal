// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#ifdef GUMBEL_SAMPLE_PRECISE_LOG
// The precise branch is the only llk-internal dependency in this header, so the default build
// carries none.
#include "sfpu/ckernel_sfpu_log.h"
#endif
#endif

/**
 * Fused Gumbel scoring: one SFPI pass computing, per DST datum,
 *
 *     score = logits * inv_temperature + (-log(-log(U)))
 *
 * DST layout contract: the uniform noise tile U sits at `idst`, the logits
 * tile LOGITS_DST_OFFSET slots above it, and the score overwrites the noise
 * tile in place. U must be strictly inside (0, 1) so both logs stay finite;
 * the caller's rand from/scale bits enforce that.
 *
 * Both logs feed only the Gumbel noise magnitude, so they run through a
 * cheap approximation (see gumbel_noise_log) rather than tt-llk's precise
 * minimax body; define GUMBEL_SAMPLE_PRECISE_LOG to use it instead. Either
 * way the pass uses immediate constants only -- no programmable const LREGs
 * and no replay slots -- so it composes with rand_tile on both architectures
 * (Wormhole's rand programs LREG12/LREG13; nothing here reads them).
 */

namespace ttml::metal::sfpu {

#ifdef TRISC_MATH

#ifdef GUMBEL_SAMPLE_PRECISE_LOG
// Source-level debugging toggle, deliberately NOT wired through the factory's defines map: every
// define the factory emits is derived from an op attribute that compute_program_hash keys on, and
// this one has no attribute -- emitted from ambient host state (an env var, say) it would change
// the compiled binary without changing the program hash, silently serving stale cached programs.
// Flipping it means editing this header, which the JIT source hash does see.
sfpi_inline sfpi::vFloat gumbel_noise_log(const sfpi::vFloat v) {
    return ckernel::sfpu::_calculate_log_body_no_init_(v);
}
#else
/**
 * Approximate ln(v) for the noise chain: exponent split plus one quadratic
 * over the mantissa octave -- 2 MADs and 4 constants (3 of them single-load
 * fp16a) versus the precise body's 3 MADs, 4 two-load fp32 constants, and
 * predicated zero guard. kLn2, kB, and kC sit exactly on the fp16a grid
 * because an fp16a-exact literal loads with a single SFPLOADI where a full
 * fp32 constant takes two -- keep them on that grid when re-fitting.
 *
 * The noise needs distributional fidelity, not ULP accuracy. Invariants
 * (pinned host-side by TestGumbelApproxLogInvariants):
 *  - Monotone: p(m) = m*(m*B + C) + D rises on [1,2) (p' >= 0.45), and the
 *    stored constants satisfy p(1) = -2^-20 and p(2) = LN2 - 2^-20 exactly,
 *    so e*LN2 + p(m) is continuous and increasing across octave boundaries
 *    (up to 1-ulp fp32 jitter). A monotone transform of U cannot reorder
 *    samples, so argmax semantics survive the approximation.
 *  - Error: |p - ln| <= 5.4e-3 on [1,2), plus |e|*2.1e-4 from the fp16a
 *    ln(2) -- orders below the sampling test's binomial resolution.
 *  - No zero guard: the caller's rand bounds give U in [2^-32, 1-2^-24],
 *    and the uniform 2^-20 downward shift keeps -log(U) >= ~1e-6 under
 *    fp32 rounding, so the outer log's argument never reaches zero or
 *    flips sign. Cost: noise tops out near 13.75 instead of 16.64,
 *    compressing only the ~1e-6 upper quantile of the Gumbel tail.
 */
sfpi_inline sfpi::vFloat gumbel_noise_log(const sfpi::vFloat v) {
    constexpr float kLn2 = 0.693359375F;  // ln(2) to fp16a precision
    constexpr float kB = -0.240234375F;   // fp16a-exact minimax under the endpoint ties
    constexpr float kC = 1.4140625F;      // kLn2 - 3*kB, fp16a-exact
    constexpr float kD = -0x1.2c801p+0F;  // 2*kB - kLn2 - 2^-20, fp32-exact
    const sfpi::vFloat m = sfpi::setexp(v, 127);
    const sfpi::vFloat poly = m * (m * kB + kC) + kD;
    const auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(v));
    const sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    return expf * kLn2 + poly;
}
#endif

template <std::uint32_t LOGITS_DST_OFFSET>
inline void calculate_gumbel_score(const std::uint32_t inv_temperature_bits) {
    // One DST tile spans 64 rows / SFP_DESTREG_STRIDE = 32 sfpi rows -- same
    // convention as tt-llk's binary SFPU ops. The 8 iterations below cover
    // ONE face: SFPU_UNARY_CALL re-invokes this body per face with dst_reg
    // re-based to that face, so the tile-sized logits offset lands on the
    // matching face of the logits tile each time.
    constexpr std::uint32_t dst_tile_size_sfpi = 32U;
    const sfpi::vFloat inv_temperature = ckernel::sfpu::Converter::as_float(inv_temperature_bits);
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat u = sfpi::dst_reg[0];
        const sfpi::vFloat logits = sfpi::dst_reg[LOGITS_DST_OFFSET * dst_tile_size_sfpi];
        // Negations are LREG sign flips; neither intermediate touches DST.
        const sfpi::vFloat neg_log_u = -gumbel_noise_log(u);
        const sfpi::vFloat gumbel = -gumbel_noise_log(neg_log_u);
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

// SFPU_UNARY_CALL resolves its functor as ::ckernel::sfpu::FN (see
// llk_math_eltwise_unary_sfpu_macros.h), so this forwarder is the one piece
// that must live here; the implementation is ttml's, in ttml::metal::sfpu.
template <std::uint32_t LOGITS_DST_OFFSET>
inline void _calculate_gumbel_score_(const std::uint32_t inv_temperature_bits) {
    ttml::metal::sfpu::calculate_gumbel_score<LOGITS_DST_OFFSET>(inv_temperature_bits);
}

}  // namespace sfpu
#endif  // TRISC_MATH

/**
 * @brief Initializes the fused Gumbel scoring SFPU operation.
 */
ALWI void gumbel_score_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused>())); }

/**
 * @brief Overwrites the uniform tile at `idst` with the fused Gumbel score.
 *
 * The logits tile must already sit `logits_dst_offset` DST slots above
 * `idst`. `inv_temperature_bits` is the FP32 bit pattern of 1/temperature.
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
