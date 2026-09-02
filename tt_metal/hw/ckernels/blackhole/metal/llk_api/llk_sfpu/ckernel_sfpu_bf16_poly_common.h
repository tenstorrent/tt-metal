// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"

// Shared evaluator helpers for the BF16-certified SFPU activation kernels
// (tenstorrent/tt-metal#49435).  Each activation header supplies only a
// Config struct of certified coefficient constants plus a thin wrapper; the
// arithmetic lives here, once per route family, so every sibling kernel is
// reviewed against the same evaluator.
//
// These paths serve the BF16 destination-register case: they assume fp32
// SFPU arithmetic on values that arrived from a BF16 tensor and they round
// results back to the BF16 grid (round-to-nearest-even) before the store.

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------
// log1p on an anchored binade split: ln(1 + v) for v in (-1, 0].
//
// Config must provide the certified fp32 correction coefficients
//     static constexpr float log1p_c0, log1p_c1, log1p_c2;
// for  log1p(r) ~= r + r^2 * ((c2*r + c1)*r + c0)  on r in [-0.25, 0.5).
// ---------------------------------------------------------------------------
template <typename Config>
sfpi_inline sfpi::vFloat log1p_anchored_bf16(sfpi::vFloat v) {
    // u = 1 + v, used only to pick the binade split k below.  The reduced
    // argument itself is rebuilt from v so it carries a single rounding.
    sfpi::vFloat u = v + 1.0f;

    // Choose k such that u * 2^-k lands in [0.75, 1.5): subtracting the fp32
    // bit pattern of 0.75 and flooring to a multiple of 2^23 leaves exactly
    // k << 23 (two's complement handles negative k).  The mantissa floor is
    // one SFPSETMAN clearing the low 23 bits.
    sfpi::vInt anchor_delta = sfpi::as<sfpi::vInt>(u) - sfpi::vInt(0x3F400000);  // bits(0.75f)
    sfpi::vInt k_shl23 = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(anchor_delta), 0));

    // 2^-k exactly, by subtracting k from the exponent field of 1.0f.
    sfpi::vFloat pow2_neg_k = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x3F800000) - k_shl23);
    // v * 2^-k exactly, the same way (exponent arithmetic only; the
    // reachable k on surviving lanes keeps this in normal range).
    sfpi::vFloat v_scaled = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(v) - k_shl23);

    // r = (1 + v) * 2^-k - 1 in [-0.25, 0.5).  (2^-k - 1) is exact for the
    // reachable k, so r carries exactly one rounding (the final add).
    sfpi::vFloat r = v_scaled + (pow2_neg_k - 1.0f);

    // log1p(r) ~= r + r^2 * ((c2*r + c1)*r + c0), minimax on [-0.25, 0.5).
    sfpi::vFloat correction = Config::log1p_c2 * r + Config::log1p_c1;
    correction = correction * r + Config::log1p_c0;
    sfpi::vFloat r_sq = r * r;
    sfpi::vFloat log1p_r = r_sq * correction + r;

    // log1p(v) = k * ln2 + log1p(r).  k << 23 converts exactly to k * 2^23
    // as fp32 (|k| <= 127 needs at most 7 mantissa bits), and the ln2
    // constant is pre-scaled by 2^-23 so a single FMA folds the shift out.
    constexpr float ln2_pow2_m23 = 0x1.62e430p-24f;  // 0x33B17218 = ln(2) * 2^-23
    sfpi::vFloat k_pow23 = sfpi::convert<sfpi::vFloat>(k_shl23, sfpi::RoundMode::Nearest);
    return k_pow23 * ln2_pow2_m23 + log1p_r;
}

// ---------------------------------------------------------------------------
// Exact halving of a doubled core value, reproducing BF16 round-to-nearest-
// even at the subnormal boundary.
//
// Kernels of the factorized-odd family compute (2x)*P(...) instead of
// x*P(...): the doubled product stays normal when x is the BF16 minimum
// normal (the direct product would flush to zero prematurely), and this
// helper folds the doubling back out.
// ---------------------------------------------------------------------------
sfpi_inline sfpi::vFloat halve_exact_bf16(sfpi::vFloat doubled) {
    sfpi::vFloat half = doubled * 0.5f;  // exact for normals; FTZ at biased exp 1

    // Blackhole flushes subnormal results AFTER rounding, so when `doubled`
    // has biased exponent 1 the halving above flushed to zero -- but if the
    // exact half is within half a BF16 ULP of the minimum normal (normalized
    // magnitude >= 2 - 2^-7 = 1.9921875), BF16 round-to-nearest-even would
    // have landed ON the minimum normal.  Reproduce that one boundary cell.
    v_if(sfpi::exexp_nodebias(doubled) == 1) {
        sfpi::vFloat norm_mag = sfpi::setexp(sfpi::abs(doubled), 127);
        v_if(norm_mag >= 1.9921875f) {
            half = sfpi::setman(doubled, 0);  // signed fp32 minimum normal
        }
        v_endif;
    }
    v_endif;
    return half;
}

// ---------------------------------------------------------------------------
// Route family "log_square_factorized_odd":
//
//     t = ln(1 - x^2);   f(x) ~= x * P(|t|)   on |x| < 1
//
// with the declared special-value shape of the inverse error function class:
//     x = +/-1 -> +/-Inf (the mathematical poles);
//     |x| > 1, +/-Inf, NaN -> +Inf;
//     x = +/-0 and BF16 subnormals -> +0.
//
// Why +Inf and +0 rather than NaN and -0: the kernel selects a quiet NaN /
// the signed core, but the FP32->BF16 rounding step (SFPSTOCHRND) maps
// exponent-255 values onto signed infinity and flushes zero/denormal inputs
// to +0 -- "+NaN becomes positive infinity", "Denormal or zero? Becomes
// zero" per the Blackhole ISA.  The input datacopy path has the same
// property, so NaN inputs already reach DST as +Inf.  This matches the
// silicon-certified receipts bit for bit (out-of-domain and NaN lanes all
// read back +Inf; -0 and negative subnormals read back +0).
//
// Config must provide, on top of the log1p coefficients above:
//     static constexpr int poly_degree;              // 3 or 4
//     static constexpr float poly[poly_degree + 1];  // P coefficients, low first
// ---------------------------------------------------------------------------
template <typename Config>
sfpi_inline sfpi::vFloat log_square_factorized_odd_body(sfpi::vFloat x) {
    static_assert(Config::poly_degree == 3 || Config::poly_degree == 4);

    // ---- Reduction: t = ln(1 - x^2) ---------------------------------------
    sfpi::vFloat x_sq = x * x;
    sfpi::vFloat neg_x_sq = -x_sq;
    sfpi::vFloat t = log1p_anchored_bf16<Config>(neg_x_sq);

    // ---- Core: q = P(|t|), Horner ------------------------------------------
    sfpi::vFloat t_abs = sfpi::abs(t);
    sfpi::vFloat q = Config::poly[Config::poly_degree];
#pragma GCC unroll 8
    for (int i = Config::poly_degree - 1; i >= 0; i--) {
        q = q * t_abs + Config::poly[i];
    }

    // ---- Reconstruction: y = x * q, via (2x)*q then an exact halving ------
    sfpi::vFloat two_x = x + x;
    sfpi::vFloat core = halve_exact_bf16(two_x * q);

    // ---- Declared specials --------------------------------------------------
    // Default quiet NaN covers |x| > 1, +/-Inf and NaN inputs (both
    // comparisons below are false there); the BF16 rounding in the row loop
    // then maps it onto +Inf, see the header comment.  x = +/-1 maps to the
    // signed infinity of the mathematical pole; setexp(+/-1.0f, 255) is
    // +/-Inf since the mantissa of 1.0 is zero.  Zeros ride the polynomial
    // core: q(0) is finite so core = 0.
    sfpi::vFloat abs_x = sfpi::abs(x);
    sfpi::vFloat result = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x7FC00000));  // quiet NaN
    v_if(abs_x < 1.0f) { result = core; }
    v_elseif(abs_x == 1.0f) { result = sfpi::setexp(x, 255); }
    v_endif;
    return result;
}

// Row loop shared by the family: evaluate, round to the BF16 grid, store,
// advance.  The rounding happens in the SFPU (round-to-nearest-even) and the
// 16-bit value is stored verbatim as vFloat16b: letting SFPSTORE convert an
// fp32 value instead would drop the NaN mantissa (NaN becomes Inf) and the
// sign of -0.
template <typename Config, int ITERATIONS>
inline void calculate_log_square_factorized_odd_bf16() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = log_square_factorized_odd_body<Config>(in);
        sfpi::vFloat16b rounded = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = rounded;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
