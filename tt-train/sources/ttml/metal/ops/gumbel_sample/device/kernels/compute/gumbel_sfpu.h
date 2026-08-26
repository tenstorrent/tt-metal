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
 * cheap approximation (see gumbel_noise_neg_log) rather than tt-llk's precise
 * minimax body; define GUMBEL_SAMPLE_PRECISE_LOG to use it instead.
 *
 * Register discipline: the pass reads its log constants from the three
 * programmable const registers (vConstFloatPrgm0..2), programmed by
 * gumbel_score_tile_init(). Wormhole's rand_tile REPROGRAMS two of those
 * slots (LREG12/13) on every call -- its PRNG state lives elsewhere -- so
 * the init must run after the LAST rand_tile it is meant to survive. rand
 * is the ONLY clobberer, which is what lets the kernel batch: it draws a
 * whole DST batch's noise first, then one init covers every score pass in
 * the batch (rand_tile(); rand_tile(); gumbel_score_tile_init();
 * gumbel_score_tile(); gumbel_score_tile()). No replay slots are used, so
 * rand's replayed row (Blackhole) is untouched.
 */

namespace ttml::metal::sfpu {

#ifdef TRISC_MATH

#ifdef GUMBEL_SAMPLE_PRECISE_LOG
// Source-level debugging toggle, deliberately NOT wired through the factory's defines map: every
// define the factory emits is derived from an op attribute that compute_program_hash keys on, and
// this one has no attribute -- emitted from ambient host state (an env var, say) it would change
// the compiled binary without changing the program hash, silently serving stale cached programs.
// Flipping it means editing this header, which the JIT source hash does see.
// Returns -ln(v): both call sites want the negated log, so the default branch folds the
// negation into its constants; this debug branch pays one register sign flip per call instead.
sfpi_inline sfpi::vFloat gumbel_noise_neg_log(const sfpi::vFloat v) {
    return -ckernel::sfpu::_calculate_log_body_no_init_(v);
}

// The precise body manages its own constants; nothing to program.
inline void gumbel_score_constants_init() {
}
#else
/**
 * Approximate -ln(v) for the noise chain: exponent split plus one quadratic
 * over the mantissa octave -- 2 MADs, three FULL-FP32 constants read for
 * free from the programmable const registers (programmed once per DST batch
 * by gumbel_score_constants_init), and one fp16a immediate, versus the precise
 * body's 3 MADs, 4 two-load fp32 constants, and predicated zero guard. The
 * NEGATION is folded into the constants: both call sites want -log (the
 * inner produces -log U, the outer's result IS the Gumbel value), so
 * returning -ln directly deletes two per-element register sign flips.
 * kGumbelPolyB is the one value still constrained to the fp16a grid: sfpi
 * exposes only three programmable float slots, so it stays an inline
 * immediate, and an fp16a-exact literal loads with a single SFPLOADI where
 * a full fp32 immediate takes two. Keep B on that grid when re-fitting; the
 * other three are free fp32 (C and D must stay EXACT derivations from ln2
 * and B, or the endpoint ties below break).
 *
 * The noise needs distributional fidelity, not ULP accuracy. Invariants
 * (pinned host-side by TestGumbelApproxLogInvariants), stated for the
 * negated polynomial q(m) = m*(m*B + C) + D = -p(m):
 *  - Monotone: q falls on [1,2) (q' <= -0.45), and the stored constants
 *    satisfy q(1) = +2^-20 and q(2) = kNegLn2 + 2^-20 exactly, so
 *    e*kNegLn2 + q(m) is continuous and decreasing across octave
 *    boundaries (up to 1-ulp fp32 jitter). A monotone transform of U
 *    cannot reorder samples, so argmax semantics survive the approximation.
 *  - Error: |q + ln| <= 5.3e-3 on [1,2), plus |e|*1.9e-9 from fp32 ln(2)
 *    (the old fp16a ln2 cost |e|*2.1e-4 -- at U ~ 2^-32 that term alone
 *    exceeded the polynomial bound) -- orders below the sampling test's
 *    binomial resolution.
 *  - No zero guard: the caller's rand bounds give U in [2^-32, 1-2^-24],
 *    and the uniform +2^-20 shift on q keeps -log(U) >= ~1e-6 under fp32
 *    rounding, so the outer call's argument never reaches zero or flips
 *    sign. Cost: noise tops out near 13.81 instead of 16.64, compressing
 *    only the ~1e-6 upper quantile of the Gumbel tail.
 */
constexpr float kGumbelNegLn2 = -0x1.62e43p-1F;  // -ln(2), full fp32 -- lives in a Prgm reg
constexpr float kGumbelPolyB = 0.240234375F;     // fp16a-exact minimax under the ties (inline immediate)
constexpr float kGumbelPolyC = -0x1.69f218p+0F;  // kGumbelNegLn2 - 3*kGumbelPolyB, fp32-exact
constexpr float kGumbelPolyD = 0x1.2c7228p+0F;   // 2*kGumbelPolyB - kGumbelNegLn2 + 2^-20, fp32-exact

// Program the log constants into the SFPU's programmable const registers, once per DST batch from
// gumbel_score_tile_init(). Turning the per-use SFPLOADI immediates (2 call sites x 4 faces = 8
// materializations per constant per tile) into one programming per batch is what buys full-fp32
// budget for these constants; sfpi exposes only THREE float slots, so kGumbelPolyB -- the cheapest
// to materialize, a single-load fp16a immediate -- stays inline in gumbel_noise_neg_log.
inline void gumbel_score_constants_init() {
    sfpi::vConstFloatPrgm0 = kGumbelNegLn2;
    sfpi::vConstFloatPrgm1 = kGumbelPolyC;
    sfpi::vConstFloatPrgm2 = kGumbelPolyD;
}

sfpi_inline sfpi::vFloat gumbel_noise_neg_log(const sfpi::vFloat v) {
    const sfpi::vFloat m = sfpi::setexp(v, 127);
    const sfpi::vFloat poly = m * (m * kGumbelPolyB + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm2;
    const auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(v));
    const sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    return expf * sfpi::vConstFloatPrgm0 + poly;
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
        // Both consumers want -log, so the helper returns it directly (negation folded into
        // its constants); nothing here flips signs, and neither intermediate touches DST.
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
ALWI void gumbel_score_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused>()));
    // AFTER the llk init (so nothing it resets clobbers them), and after the LAST rand_tile of the
    // kernel's DST batch (Wormhole's rand reprograms LREG12/13 -- two of the three Prgm slots -- on
    // every call; nothing else clobbers them, so one init serves every score pass in the batch).
    MATH((ttml::metal::sfpu::gumbel_score_constants_init()));
}

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
