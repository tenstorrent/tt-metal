// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"

// ---------------------------------------------------------------------------
// SLOPE_MAX_ELIGIBLE — arch eligibility for the slope_max algebraic collapse.
//
// slope_max (9aaf85211) shipped with a Blackhole-only silicon A/B (its commit
// cites leaky_relu 1.80->1.70us and prelu 1.78->1.66us: Blackhole-class
// runtimes). On Wormhole every corpus row that declares it is wrong: 48 prelu
// s2 configs abort with 896 non-finite outputs, and 48 leaky_relu s2 configs
// complete with plausible runtimes but max_ulp_pure = inf -- measured, and
// garbage. Blackhole runs the identical corpus clean (prelu s2 0.0 pure,
// leaky_relu s2 0.48 pure, 100% ML).
//
// The MECHANISM IS NOT ESTABLISHED. This is a C-LOW-3 known fallback gated on
// the measurement, not on a story: WH falls through to the poly cascade, which
// is what these configs used before the corpus declared slope_max. The gate is
// structural (arch + eval kind), never an activation name (C-LOW-1).
//
// SLOPE_MAX_DISABLE is the C-LOW-2 escape arm, reachable on any arch via
// TT_ACT_KERNEL_EXTRA_DEFINES.
// ---------------------------------------------------------------------------
#if defined(ARCH_WORMHOLE) || defined(SLOPE_MAX_DISABLE)
#define SLOPE_MAX_ELIGIBLE 0
#else
#define SLOPE_MAX_ELIGIBLE 1
#endif

#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#if defined(TT_TARGET_BUILTIN_PERIODIC)
#include "api/compute/eltwise_unary/trigonometry.h"
#endif
#if defined(TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_EXACT_STORE) || \
    defined(TT_TARGET_BH_BF16_THRESHOLD_SOFTSHIFT_NATIVE)
#include "api/compute/eltwise_unary/activations.h"
#endif
#if defined(CLAMPED_AFFINE_BF16_NATIVE_RELU)
#include "api/compute/eltwise_unary/relu.h"
#endif
#if defined(NEWTON_ROOT_BACKEND_NATIVE_SFPU_TILE)
#if (NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL)
#include "api/compute/eltwise_unary/rsqrt.h"
#elif NEWTON_ROOT_N == 3
#include "api/compute/eltwise_unary/cbrt.h"
#endif
#endif
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

// Canonical eval_method taxonomy. Translates the single EVAL_METHOD_* selector
// (+ EXPONENT_ALU_* / REDUCE_* sub-tags) emitted by codegen into the legacy
// feature macros the kernel bodies below were written against. Must precede any
// RANGE_REDUCTION_* / EVAL_METHOD_* read.
#include "eval_method.h"

// Include reciprocal function for tan range reduction (tan_expand needs -1/poly),
// the exp2 sigmoid compose (1/(1+exp(-x))), sigmoid-product compose
// (x/(1+exp(-x))), hyperbolic composition, the REDUCE_EXP hyperbolic-even
// combine (cosh = 0.5*t + 0.5/t, t = exp(|x|)), and the pow path's final
// 1/result (rsqrt = 1/sqrt(x), tagged expalu_reciprocal -> POW_HW_RECIPROCAL).
#if (                                                                                                         \
    defined(RANGE_REDUCTION_TAN) || defined(EVAL_METHOD_TAN_STANDALONE) || defined(EXP_HW_COMPOSE_SIGMOID) || \
    defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) ||          \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) ||                                                               \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) ||                                                         \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) ||                                                    \
    defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD) ||             \
    defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT) || defined(REDUCE_EXP_COMPOSE_SIGMOID) ||                    \
    defined(POW_HW_RECIPROCAL) ||                                                                            \
    defined(TT_ABS_RESIDUAL_AFFINE_1_NEEDS_RECIPROCAL) ||                                                    \
    defined(TT_SQUARE_DECAY_NEEDS_RECIPROCAL) ||                                                             \
    defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1) ||                                                     \
    defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1) ||                                        \
    defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1) ||                                                \
    defined(TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL) ||                                                     \
    defined(TT_SELECTED_CORE_NEEDS_RECIPROCAL)) &&                                                          \
    defined(TRISC_MATH)
#include "ckernel_sfpu_recip.h"
#endif

#if (defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1) || \
     defined(TT_TARGET_SELECTED_BF16_SCRATCH_BOUNDARY)) && defined(TRISC_MATH)
#include "sfpu/ckernel_sfpu_log.h"
#endif

// Blackhole's exponent-aware RNE helper leaves |x| >= 2^23 unchanged, where
// every finite FP32 lane is already integral.  This gives Program range
// reduction exact full-lattice behavior without a bounded C231 assumption.
#if (defined(TT_INLINE_PROGRAM_NEEDS_ROUND_EVEN) || \
     defined(TT_SELECTED_CORE_TOTAL_NEEDS_ROUND_EVEN)) && defined(TRISC_MATH)
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#endif

namespace kutil = norm::kernel_util;

/**
 * Generic Piecewise Polynomial LUT Activation
 *
 * This kernel supports ANY polynomial degree (0-16) and ANY depth (4-64 segments).
 * Each segment is approximated as: y = c0 + c1*x + c2*x² + ... + cn*x^n
 * Evaluated using Horner's scheme for numerical stability.
 *
 * Template parameters (compile-time):
 *   POLY_DEGREE   - Polynomial degree (0=constant, 1=linear, 2=quadratic, ... 16=hexadecic)
 *   NUM_SEGMENTS  - Number of segments (4, 8, 16, 32, 64, etc.)
 *   LUT_SIZE      - Total LUT size = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)
 *
 * LUT Format: [b0, b1, ..., bN, c0_seg0, c1_seg0, ..., cn_seg0, c0_seg1, ...]
 * where bi are boundary points and cj are polynomial coefficients
 *
 * Benefits:
 *   - Single implementation for all degree/depth combinations
 *   - Zero runtime overhead (all branching at compile-time)
 *   - Automatic support for new configurations
 *   - Reduced code duplication (7 files → 1 file)
 *
 * Supports both modes:
 *   - EMBEDDED_LUT: Coefficients compiled directly into kernel (zero L1 overhead)
 *   - CB mode: Coefficients loaded from L1 circular buffer (runtime flexibility)
 */

#ifdef TRISC_MATH

#include "ckernel_sfpu_sqrt_custom.h"

// sqrt_factored basis (acos = sqrt(1-|x|)*P(|x|)) needs the single-tile eval
// path: the software sqrt has too many live registers for the dual-tile path
// (SFPU compiler ICE "maximum number of generated reload insns"). Force
// single-path here so the basis always compiles, independent of --no-dual-eval.
#if defined(BASIS_MUL_SQRT_1_MINUS_ABS) && defined(USE_DUAL_EVAL)
#undef USE_DUAL_EVAL
#endif

// The ASYMPTOTIC_FACTOR_QUADRATIC post-poly apply lives only in the single-tile
// piecewise_generic_lut path (not the dual-tile variant). Force single-path so
// the factor is actually applied (else the kernel silently emits the bare
// correction poly P(x) instead of (x-A)(x-B)*P(x)).
#if defined(ASYMPTOTIC_FACTOR_QUADRATIC) && defined(USE_DUAL_EVAL)
#undef USE_DUAL_EVAL
#endif

// The REDUCE_LOG sqrt2-centered mantissa fold (m>=sqrt2 -> setexp step), its
// residual-coordinate shift (x = m - 1.0f), and the linear-term-split poly eval
// (eval_polynomial_log_split) live ONLY in the single-tile evaluators
// (piecewise_generic_lut here and piecewise_generic_lut_specialized_N in
// piecewise_generic_specialized.cpp). The dual-tile variants emit a bare
// log_reduce on the raw mantissa m in [1,2) with none of the fold/shift/split,
// so pairing dual eval with this compose silently drops the whole reduction and
// evaluates the residual-fit polynomial on raw m (silicon MAE ~5.6, ~4.7M
// negative outputs, pure_ulp -> inf). Force single-path so the fold + residual
// shift + split eval are always applied, mirroring the two single-path-only
// guards above. (The specialized dispatch also pins REDUCED_POLY to single, but
// this makes the requirement explicit for every log route that reaches here.)
#if (defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED) || defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT)) && \
    defined(USE_DUAL_EVAL)
#undef USE_DUAL_EVAL
#endif

// Range reduction: Cody-Waite method for exp/trig, mantissa/exponent extraction for log/cbrt
// Only included when a reduction/standalone eval_method or ASYMPTOTIC_FACTOR_* is active.
#if EVAL_METHOD_NEEDS_REDUCTION_HELPERS || defined(TT_INLINE_PROGRAM_NEEDS_RANGE_INTEGER) || \
    defined(TT_SELECTED_CORE_TOTAL_NEEDS_ROUND_EVEN) || \
    defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || \
    defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace sfpi {

#include "sfpi_min_max.h"
#include "domain_actions.h"

// Complete-Program evaluation is a host/model API, not a production kernel
// selector.  Deployment Programs may only extend the already-selected mature
// CSV evaluator through TT_INLINE_PROGRAM_OVERLAY.
#if defined(TT_INLINE_PROGRAM)
#error "complete TT_INLINE_PROGRAM evaluation is not a production deployment path"
#endif

inline vFloat apply_output_postcompose(vFloat y) {
#if defined(POSTCOMPOSE_AFFINE_Y)
    y = POSTCOMPOSE_B * y + POSTCOMPOSE_A;
#endif
    return y;
}

#ifdef RANGE_REDUCTION_EXP
// Cody-Waite range reduction: x → (s, k_int) where exp(x) = 2^k * exp(s)
// s ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.347, 0.347]
inline void exp_reduce(vFloat x, vFloat& s, vInt& k_int) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    vFloat z = x * INV_LN2;

    // Round z to nearest integer (branch-free magic-number technique from ckernel_sfpu_exp.h)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    k_int = as<vInt>(tmp) - as<vInt>(c231);

    // Cody-Waite extended precision: s = x - k*ln2
    // Using negated constants so compiler can emit single SFPMAD instructions
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vFloat s_hi = k * NEG_LN2_HI + x;
    s = k * NEG_LN2_LO + s_hi;
}

// exp2 EXACT integer reduction: x -> (r, k_int) where 2^x = 2^k * 2^r,
// k = round(x), r = x - k in [-0.5, 0.5]. Unlike exp_reduce there is NO
// Cody-Waite ln2 split -- the base-2 reduction is bit-exact (subtracting a
// nearby integer), so a single subtract suffices and no low-word correction
// is needed. The reduced leaf P(r) ~= 2^r; reconstruction reuses exp_expand
// (result * 2^k) unchanged.
inline void exp2_reduce(vFloat x, vFloat& r, vInt& k_int) {
    // Branch-free round-to-nearest-even via the 2^23 magic number, identical
    // to exp_reduce's k derivation but applied directly to x (base 2), not to
    // x*INV_LN2 (base e).
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = x + c231;
    vFloat k = tmp - c231;
    k_int = as<vInt>(tmp) - as<vInt>(c231);
    r = x - k;
}

// Reconstruct: result = poly_result * 2^k via exponent bit manipulation.
//
// UNDERFLOW GUARD: same defect as asymptotic_exp below, dst_coeff_store.h::
// asym_exp_dst and the rational twin piecewise_rational.cpp::exp_expand --
// setexp writes the exponent FIELD directly, so a biased exponent at or below 0
// wraps into a large POSITIVE exponent and returns ~1e38 where the true value
// has underflowed to ~0.  The callers' EXP_UNDERFLOW floor (-88.5f) is looser
// than the arithmetic: with poly_result ~= exp(s), s in [-ln2/2, ln2/2], the
// exponent field is 126 or 127, so the wrap begins at x = -126*ln2 = -87.33654
// and x in [-88.5, -87.33654) reaches here with new_exp <= 0.
//
// Clamp AFTER setexp: setexp(0.0f, e) rebuilds a nonzero value from the zero
// mantissa, so zeroing poly_result beforehand does not help.  The incoming
// poly_result slot is reused for the result to keep the live set minimal.
inline vFloat exp_expand(vFloat poly_result, vInt k_int) {
    vInt new_exp = exexp(poly_result, ExponentMode::Biased) + k_int;
    poly_result = setexp(poly_result, new_exp);
    // Saturate on the RECONSTRUCTED exponent field, not a hardcoded x_orig
    // threshold: every exp-family function then overflows/underflows at its own
    // true FLT_MAX/denormal crossing (natural exp ~88.72, exp2 128, cosh
    // ~89.416) instead of a single fixed constant that was correct only for
    // natural exp.  new_exp <= 0 -> underflow to 0; new_exp >= 255 -> overflow
    // to +inf (255 is the fp32 inf/NaN exponent field).
    v_if(new_exp <= 0) { poly_result = 0.0f; }
    v_elseif(new_exp >= 255) { poly_result = std::numeric_limits<float>::infinity(); }
    v_endif;
    return poly_result;
}
#endif

#ifdef RANGE_REDUCTION_TRIG
// Trig range reduction: x → (s, q_int) where f(x) = (-1)^q * f(s)
// s ∈ [-π/2, π/2], works for both sin and cos
inline void trig_reduce(vFloat x, vFloat& s, vInt& q_int) {
    constexpr float FRAC_1_PI = 0.31830988618379067f;
    vFloat z = x * FRAC_1_PI;

    // Round z to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat q = tmp - c231;
    q_int = as<vInt>(tmp) - as<vInt>(c231);

    // Cody-Waite: s = x - q*π with extended precision
    constexpr float NEG_PI_HI = -3.140625f;
    constexpr float NEG_PI_LO = -0.00096765358979323846f;
    vFloat s_hi = q * NEG_PI_HI + x;
    s = q * NEG_PI_LO + s_hi;
}

// Reconstruct: sin(x + kπ) = (-1)^k * sin(x), cos(x + kπ) = (-1)^k * cos(x)
inline vFloat trig_expand(vFloat poly_result, vInt q_int) {
    v_if(q_int & 1) { poly_result = -poly_result; }
    v_endif;
    return poly_result;
}
#endif

#if defined(EVAL_METHOD_TRIG_RESIDUAL)
inline vFloat trig_residual_reduce(vFloat x, vFloat& s) {
#if defined(TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD)
    vFloat half = sFloat16b(0.5f);
    vFloat inv_pi = vConstFloatPrgm2;
    vFloat z = __builtin_rvtt_sfpmad(x.get(), inv_pi.get(), half.get(), SFPMAD_MOD1_OFFSET_NONE);

    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat q = tmp - c231;
    vInt q_int = as<vInt>(tmp) - as<vInt>(c231);

    vFloat two = sFloat16b(2.0f);
    vFloat neg_one = vFloat(-1.0f);
    vFloat j = __builtin_rvtt_sfpmad(q.get(), two.get(), neg_one.get(), SFPMAD_MOD1_OFFSET_NONE);
    constexpr float NEG_PI_2_P0 = -0x1.92p+0f;
    constexpr float NEG_PI_2_P1 = -0x1.fbp-12f;
    vFloat a = j * NEG_PI_2_P0 + x;
    a = j * NEG_PI_2_P1 + a;
#elif defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD)
    vFloat inv_pi = vConstFloatPrgm2;
    vFloat z = x * inv_pi;

    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat j = tmp - c231;
    vInt q_int = as<vInt>(tmp) - as<vInt>(c231);

    constexpr float NEG_PI_P0 = -0x1.92p+1f;
    constexpr float NEG_PI_P1 = -0x1.fbp-11f;
    vFloat a = j * NEG_PI_P0 + x;
    a = j * NEG_PI_P1 + a;
#else
#error "EVAL_METHOD_TRIG_RESIDUAL requires a supported TRIG_RESIDUAL_PHASE_*"
#endif
    a = j * vConstFloatPrgm0 + a;
    a = j * vConstFloatPrgm1 + a;

    q_int <<= 31;
    a = as<vFloat>(as<vInt>(a) ^ q_int);

    s = a * a;
    return a;
}

template <uint32_t DEGREE>
inline vFloat trig_residual_odd_eval(vFloat x) {
    vFloat s;
    vFloat a = trig_residual_reduce(x, s);
    if constexpr (DEGREE == 7) {
        vFloat r = TRIG_RESIDUAL_COEFFS[7] * s + TRIG_RESIDUAL_COEFFS[5];
        vFloat c = a * s;
        r = r * s + TRIG_RESIDUAL_COEFFS[3];
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + TRIG_RESIDUAL_COEFFS[1] * a;
#endif
    } else if constexpr (DEGREE == 5) {
        vFloat r = TRIG_RESIDUAL_COEFFS[5] * s + TRIG_RESIDUAL_COEFFS[3];
        vFloat c = a * s;
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + TRIG_RESIDUAL_COEFFS[1] * a;
#endif
    } else {
        constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
        constexpr int STEPS = (TOP - 1) / 2;
        vFloat r = TRIG_RESIDUAL_COEFFS[TOP];
#pragma GCC unroll 8
        for (int k = 1; k <= STEPS; k++) {
            r = r * s + TRIG_RESIDUAL_COEFFS[TOP - 2 * k];
        }
        return r * a;
    }
}

// ---------------------------------------------------------------------------
// EXPERIMENT (perf parity vs TTNN calculate_sine/calculate_cosine):
// hoisted-constant reduce + eval. Differences vs trig_residual_reduce above:
//   1. sign = as<vInt>(tmp) << 31 directly — the low bit of the
//      BIASED float IS the integer parity (bias low bits are zero), so the
//      explicit (int)tmp - (int)c231 SFPIADD is dropped and one live register
//      is freed (bit-exact; same trick as TTNN ckernel_sfpu_trigonometry.h:216
//      `q = as<vInt>(j)`). Freeing that register is what lets the allocator
//      keep the 3 polynomial coefficients pinned across the loop.
//   2. Cody-Waite P0 step emitted as an explicit SFPLOADI+SFPMAD builtin
//      (2 instrs) instead of the SFPMOV+SFPMULI+SFPADD (3) the optimizer
//      picks for the bf16-exact constant; bit-exact for |j| <= 1 (exact
//      product), which holds on the deployed [-pi, pi] domain.
//   3. sine phase only, under TRIG_FUSED_ROUND (default OFF): tmp =
//      mad(x, 1/pi, bias) in a single SFPMAD (TTNN trigonometry.h:211-212)
//      instead of mul-then-add (-1 instr/vector). Proven output-bit-identical
//      on the bf16 domain grid, but shifts fp32-precision outputs on
//      double-rounding boundaries: enabling it requires the
//      ttpoly/precision/rangered.py::_eval_trig_residual +
//      tests/craq_sim_golden/trig_residual_golden.cpp sync described at the
//      TRIG_FUSED_ROUND site below.
inline vFloat trig_residual_reduce_hoisted(vFloat x, vFloat& s) {
    // Round-magic bias 1.5*2^23 (0x4B400000). Written as a float literal so the
    // SFPI optimizer can fold `+ BIAS` / `- BIAS` into single SFPADDI immediates
    // where a register copy is not otherwise needed.
    constexpr float BIAS = 12582912.0f;
#if defined(TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD)
    vFloat half = sFloat16b(0.5f);
    vFloat inv_pi = vConstFloatPrgm2;
    vFloat z = __builtin_rvtt_sfpmad(x.get(), inv_pi.get(), half.get(), SFPMAD_MOD1_OFFSET_NONE);
    vFloat tmp = z + BIAS;
    // Quadrant parity read straight off the BIASED float's low bit (the bias
    // bit pattern has a zero LSB, so (int)tmp and (int)tmp-(int)bias share
    // parity). Shifting the biased bits kills both the SFPIADD subtract AND
    // the extra live register the old (int)tmp-(int)c231 needed — the same
    // trick as TTNN trigonometry.h:216/228 (as<vInt>(j); q <<= 31).
    vInt sign = as<vInt>(tmp) << 31;
    vFloat q = tmp - BIAS;

    vFloat two = sFloat16b(2.0f);
    vFloat neg_one = vFloat(-1.0f);
    vFloat j = __builtin_rvtt_sfpmad(q.get(), two.get(), neg_one.get(), SFPMAD_MOD1_OFFSET_NONE);
    // P0 step as an explicit SFPLOADI(bf16)+SFPMAD: written `x + j*P0` the SFPI
    // optimizer picks SFPMOV+SFPMULI+SFPADD (3 instrs) because P0 is
    // bf16-exact. Bit-exact swap: |j| == 1 on the deployed domain, so j*P0 is
    // an exact product and mul-then-add == fused mad (and the ttpoly
    // precision model rangered.py::_eval_trig_residual already models this
    // step as a fused FMA).
    vFloat p0 = sFloat16b(-0x1.92p+0f);
    vFloat a = __builtin_rvtt_sfpmad(j.get(), p0.get(), x.get(), SFPMAD_MOD1_OFFSET_NONE);
    constexpr float NEG_PI_2_P1 = -0x1.fbp-12f;
    a = a + j * NEG_PI_2_P1;
#elif defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD)
    vFloat inv_pi = vConstFloatPrgm2;
#if defined(TRIG_FUSED_ROUND)
    // OPTIONAL (-1 instr/vector): single-rounded biased round, TTNN sine
    // trigonometry.h:211-218. Output-bit-identical over the full bf16 domain
    // grid (proven exhaustively), but at fp32 precision it can move the
    // reduced argument on double-rounding boundary inputs — enabling this
    // requires syncing ttpoly/precision/rangered.py::_eval_trig_residual
    // (sine branch: z = fma(x, 1/pi, 0); tmp = z + c231  ->  a single
    // tmp = fma(x, 1/pi, c231)) and
    // tests/craq_sim_golden/trig_residual_golden.cpp (same change) +
    // regenerating trig_residual_golden.txt, then restamping fp32 metrics.
    vFloat bias_v = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = __builtin_rvtt_sfpmad(x.get(), inv_pi.get(), bias_v.get(), SFPMAD_MOD1_OFFSET_NONE);
    vInt sign = as<vInt>(tmp) << 31;  // see cosine comment above
    vFloat j = tmp - bias_v;
#else
    vFloat z = x * inv_pi;
    vFloat tmp = z + BIAS;
    vInt sign = as<vInt>(tmp) << 31;  // see cosine comment above
    vFloat j = tmp - BIAS;
#endif

    // See cosine comment: explicit LOADI+MAD beats the MOV+MULI+ADD the
    // optimizer picks for the bf16-exact P0; bit-exact for |j| <= 1.
    vFloat p0 = sFloat16b(-0x1.92p+1f);
    vFloat a = __builtin_rvtt_sfpmad(j.get(), p0.get(), x.get(), SFPMAD_MOD1_OFFSET_NONE);
    constexpr float NEG_PI_P1 = -0x1.fbp-11f;
    a = a + j * NEG_PI_P1;
#else
#error "trig_residual_reduce_hoisted requires a supported TRIG_RESIDUAL_PHASE_*"
#endif
    a = a + j * vConstFloatPrgm0;
    a = a + j * vConstFloatPrgm1;

    a = as<vFloat>(as<vInt>(a) ^ sign);

    s = a * a;
    return a;
}

// Odd-residual Horner with every coefficient in a caller-held LREG (loaded
// ONCE before the 32-iteration dst loop) — kills the per-iteration 2x SFPLOADI
// rematerialization per fp32 coefficient (-6 instr/vector at degree 7).
// ch/cm/cl = c[TOP], c[TOP-2], c[TOP-4] (cl unused at degree 5).
template <uint32_t DEGREE>
inline vFloat trig_residual_odd_eval_hoisted(vFloat x, vFloat ch, vFloat cm, vFloat cl) {
    vFloat s;
    vFloat a = trig_residual_reduce_hoisted(x, s);
    if constexpr (DEGREE == 7) {
        vFloat r = ch * s + cm;
        vFloat c = a * s;
        r = r * s + cl;
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + TRIG_RESIDUAL_COEFFS[1] * a;
#endif
    } else if constexpr (DEGREE == 5) {
        (void)cl;
        vFloat r = ch * s + cm;
        vFloat c = a * s;
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + TRIG_RESIDUAL_COEFFS[1] * a;
#endif
    } else {
        // Generic odd Horner (hoisted args unused beyond the top coefficient).
        (void)ch;
        (void)cm;
        (void)cl;
        constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
        constexpr int STEPS = (TOP - 1) / 2;
        vFloat r = TRIG_RESIDUAL_COEFFS[TOP];
#pragma GCC unroll 8
        for (int k = 1; k <= STEPS; k++) {
            r = r * s + TRIG_RESIDUAL_COEFFS[TOP - 2 * k];
        }
        return r * a;
    }
}

template <uint32_t DEGREE>
inline vFloat trig_residual_odd_eval_preloaded(vFloat x, const float* coeffs) {
    vFloat s;
    vFloat a = trig_residual_reduce(x, s);
    if constexpr (DEGREE == 7) {
        vFloat r = coeffs[0] * s + coeffs[1];
        vFloat c = a * s;
        r = r * s + coeffs[2];
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + coeffs[3] * a;
#endif
    } else if constexpr (DEGREE == 5) {
        vFloat r = coeffs[0] * s + coeffs[1];
        vFloat c = a * s;
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + coeffs[2] * a;
#endif
    } else {
        constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
        constexpr int STEPS = (TOP - 1) / 2;
        vFloat r = coeffs[0];
#pragma GCC unroll 8
        for (int k = 1; k <= STEPS; k++) {
            r = r * s + coeffs[k];
        }
        return r * a;
    }
}

// ----------------------------------------------------------------------------
// TTI REPLAY fast path for the standalone trig residual (sin/cos). Same
// mechanism as kExpHwTtiReplay below (see that comment block for the replay
// idiom, the SFPMAD stream-rate hazard, and the D-RWC/SETRWC hazard): the
// per-element body is recorded into replay-buffer slot 0 once per tile and
// then issued as ONE TTI_REPLAY per element, with the dst walk done by an
// ADDR_MOD_6 dest += 2 auto-increment on the SFPSTORE.
//
// Unlike exp (13 ops/element, 2-element interleave), the trig body is 24-28
// Tensix ops/element: a 2-element interleave (48-56 slots) cannot fit the
// 32-slot replay buffer, so the body is SINGLE-element with explicit gap
// slots. Gap discipline (hazard rule 1): the result of any MAD-family op
// (SFPMAD/SFPMUL/SFPADD/SFPADDI/SFPMULI) is not readable by the immediately
// following instruction at replay stream rate — every dependent consumer
// sits >= 2 instructions after its producer, with SFPLOADI/SFPSHFT/SFPNOP
// fillers in between. 1-cycle ops (EXEXP/EXMAN/SHFT/CAST/SETEXP/MOV/XOR/
// LOADI/IADD/logic) may be consumed back-to-back (TTNN exp_21f precedent).
//
// The arithmetic stream is a 1:1 transcription of the sfpi/GCC BH loop body
// (objdump of the selected p5/p7 configs, hash-bucket product tt-bh): same
// ops, same order, same operand roles — only the issue mechanism and the
// NOP/LOADI filler placement differ, so outputs are bit-identical. The raw
// dst_reg++ walk that corrupted wave-3 sin/cos is NOT used: the D-RWC
// counter advances via ADDR_MOD_6 (+2/element) and is re-zeroed by the
// trailing SETRWC SET_D (hazard rule 2).
//
// Gates: BH + bf16 + standalone trig residual + C1_IS_ONE + degree 5/7 +
// default (non-fused) rounding. TRIG_TTI_DISABLE=defined forces the sfpi
// path for A/B measurement.
#if defined(TRIG_RESIDUAL_C1_IS_ONE) &&                                                   \
    (defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD) || defined(TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD)) && \
    !defined(TRIG_FUSED_ROUND) && defined(USE_BF16) &&                                    \
    (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && !defined(TRIG_TTI_DISABLE) && \
    !defined(TT_DOMAIN_ACTION_PROGRAM)
#define TRIG_TTI_CANDIDATE 1
constexpr bool kTrigTtiReplay = (TRIG_RESIDUAL_DEGREE == 5) || (TRIG_RESIDUAL_DEGREE == 7);
#if defined(TT_INLINE_PROGRAM_OVERLAY)
static_assert(
    !kTrigTtiReplay,
    "typed core overlay requires route-aware TRIG_TTI replay; refusing silent fallback");
#endif
#else
#define TRIG_TTI_CANDIDATE 0
constexpr bool kTrigTtiReplay = false;
#endif
#endif
#if defined(TT_EXPERIMENT_PRELUDE_ISSUES)
// ============================================================================
// EXPERIMENT ONLY — DO NOT MERGE, DO NOT CERTIFY (trig prelude issue-cost
// microbenchmark, 2026-09-02).
//
// Emits TT_EXPERIMENT_PRELUDE_ISSUES synthetic-but-dependent SFPU vector
// instructions per dst row, issued directly (NOT recorded into the replay
// buffer) ahead of the deployed trig TTI-replay body, to measure the marginal
// silicon cost per issued instruction of a prospective periodic-window
// reduction prelude (analysis/bf16_periodic_window_prelude_lowering_refusal).
//
// Mix per 10 instructions: 4 MUL/MAD-class, 2 SFPSHFT, 2 SFPMOV, 2 SFPLOADI
// (~the refusal itemization's 40/20/20/20). The chain is genuinely dependent
// (each block feeds the next through L0-L3) and TTI_* expansions are volatile
// instruction-buffer stores, so nothing can be dead-code-eliminated.
//
// Output preservation is BY CONSTRUCTION, not by cleanup: the prelude touches
// ONLY L0-L3 — registers the sine/cosine replay bodies write before reading —
// and never dst, CC state, the per-tile coefficient pins L4-L6, or the prgm
// constants L12-L14. The exhaustive BF16 dump comparison at every N re-proves
// byte-identical outputs. The final two prelude slots are never MAD-class so
// no in-flight MAD writeback can race the body's first register writes.
//
// Emitted once in text (noinline) and called per row: call overhead is a
// constant per row for every N>0 and cancels out of the fitted slope.
#if !defined(EVAL_METHOD_TRIG_RESIDUAL)
#error "TT_EXPERIMENT_PRELUDE_ISSUES requires the trig_residual standalone route"
#endif
#if !TRIG_TTI_CANDIDATE
#error "TT_EXPERIMENT_PRELUDE_ISSUES requires the trig TTI replay candidate gate"
#endif
#if defined(ARCH_WORMHOLE)
#error "TT_EXPERIMENT_PRELUDE_ISSUES microbenchmark is Blackhole-only"
#endif
static_assert(kTrigTtiReplay,
              "TT_EXPERIMENT_PRELUDE_ISSUES requires the trig replay route to fire");
static_assert(TT_EXPERIMENT_PRELUDE_ISSUES >= 0 && TT_EXPERIMENT_PRELUDE_ISSUES <= 512,
              "prelude issue count out of experiment range");

#ifdef TRISC_MATH
template <int I, int N>
inline void tt_experiment_prelude_issue_one() {
    constexpr int k = I % 10;
    // Last two slots demoted from MAD-class (see comment block above).
    constexpr bool no_mad_tail = (I >= N - 2);
    if constexpr (k == 0) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x3F80);                  // L0 = const
    } else if constexpr (k == 1) {
        if constexpr (no_mad_tail) {
            TTI_SFPSHFT(0x001, 0, 1, 7);                                  // L1 = L0 << 1
        } else {
            TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);  // L1 = L0*L0
        }
    } else if constexpr (k == 2) {
        TTI_SFPSHFT(0x001, 3, 2, 7);                                      // L2 = L3 << 1
    } else if constexpr (k == 3) {
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0); // L3 = L1
    } else if constexpr (k == 4) {
        if constexpr (no_mad_tail) {
            TTI_SFPSHFT(0x002, 1, 2, 7);                                  // L2 = L1 << 2
        } else {
            TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0);  // L2 = L1*L0
        }
    } else if constexpr (k == 5) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 0, 0x0001);                  // L1 = const
    } else if constexpr (k == 6) {
        if constexpr (no_mad_tail) {
            TTI_SFPSHFT(0x003, 2, 3, 7);                                  // L3 = L2 << 3
        } else {
            TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 0);     // L3 = L2*L0+L3
        }
    } else if constexpr (k == 7) {
        TTI_SFPSHFT(0x002, 1, 2, 7);                                      // L2 = L1 << 2
    } else if constexpr (k == 8) {
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG0, 0); // L0 = L3
    } else {
        if constexpr (no_mad_tail) {
            TTI_SFPMOV(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, 0);  // L1 = L2
        } else {
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);     // L1 = L0*L2+L1
        }
    }
}

template <int N, int I = 0>
inline void tt_experiment_prelude_chain() {
    if constexpr (I < N) {
        tt_experiment_prelude_issue_one<I, N>();
        tt_experiment_prelude_chain<N, I + 1>();
    }
}

[[maybe_unused]] __attribute__((noinline)) static void tt_experiment_prelude_row() {
    tt_experiment_prelude_chain<(int)(TT_EXPERIMENT_PRELUDE_ISSUES)>();
}
#endif  // TRISC_MATH
#endif  // TT_EXPERIMENT_PRELUDE_ISSUES

#ifdef RANGE_REDUCTION_TAN
// Tan range reduction: x → (a, j_int) where a ∈ [-π/4, π/4]
// Uses Cody-Waite reduction matching TTNN's sfpu_tan implementation
// j = round(x / (π/2)), a = x - j * (π/2)
inline void tan_reduce(vFloat x, vFloat& a, vInt& j_int) {
    constexpr float FRAC_2_PI = 0.6366197723675814f;  // 2/π
    vFloat z = x * FRAC_2_PI;

    // Round z to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat j = tmp - c231;
    j_int = as<vInt>(tmp) - as<vInt>(c231);

    // Cody-Waite: a = x - j*(π/2) with extended precision
    // Split π/2 into high and low parts for accuracy
    constexpr float NEG_PI_2_HI = -1.5703125f;              // -π/2 high bits (fits in bf16)
    constexpr float NEG_PI_2_LO = -0.0004837512969970703f;  // -π/2 remainder
    vFloat a_hi = j * NEG_PI_2_HI + x;
    a = j * NEG_PI_2_LO + a_hi;
}

// Reconstruct: tan(x) where x = a + j*(π/2)
// j even: tan(x) = tan(a) = poly(a)
// j odd:  tan(x) = -cot(a) = -1/tan(a) = -1/poly(a)
inline vFloat tan_expand(vFloat poly_result, vInt j_int) {
    v_if(j_int & 1) {
        // j is odd: result = -1/poly(a)
        // Use 3 Newton-Raphson iterations for better precision (vs 2 in sfpu_reciprocal<false>)
        poly_result = -ckernel::sfpu::sfpu_reciprocal_iter<3>(poly_result);
    }
    v_endif;
    return poly_result;
}
#endif

#if defined(EVAL_METHOD_TAN_STANDALONE)
// tail_p2 = 3rd Cody-Waite term of -pi/2 (-0x1.51p-22f), caller-pinned into an
// LREG (hw_preload_pin at the dst-loop site). It MUST NOT live in
// vConstFloatPrgm0: the quadrant reciprocal below calls sfpu_reciprocal_iter,
// and sfpu_reciprocal owns Prgm0 on BH (2.0 Newton constant) and ALL of
// Prgm0/1/2 on WH (Sollya seed constants) — see the PRGM CONSTANT OWNERSHIP
// RULE in the preamble. This exact clobber shipped: sfpu_reciprocal_init ran
// after the preload and every tan_standalone corpus row evaluated
// a + j*2.0 instead of a + j*(-6.3e-7).
template <uint32_t DEGREE>
inline vFloat tan_standalone_eval(vFloat x, vFloat tail_p2) {
#if defined(ARCH_WORMHOLE)
    // WH: Prgm1/2 are reciprocal-owned too — 2/pi and the 4th Cody-Waite term
    // are plain locals (in-body materialization; correctness over issue count).
    vFloat inv_pio2 = 0.63661974668502808f;
#else
    // BH: the reciprocal owns only Prgm0; Prgm1/2 keep the preloaded constants.
    vFloat inv_pio2 = vConstFloatPrgm2;
#endif
    vFloat rounding_bias = sFloat16b(0x1.8p23f);
    vFloat j = __builtin_rvtt_sfpmad(x.get(), inv_pio2.get(), rounding_bias.get(), SFPMAD_MOD1_OFFSET_NONE);
    vInt i = as<vInt>(j);
    j += -rounding_bias;
    i <<= 31;

    constexpr float NEG_PI_2_P0 = -0x1.92p+0f;
    constexpr float NEG_PI_2_P1 = -0x1.fbp-12f;
    vFloat a = x + j * NEG_PI_2_P0;
    a = a + j * NEG_PI_2_P1;
    a = a + j * tail_p2;
#if defined(ARCH_WORMHOLE)
    a = a + j * -0x1.0b4612p-34f;
#else
    a = a + j * vConstFloatPrgm1;
#endif

    vFloat s = a * a;
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    vFloat r = TAN_STANDALONE_COEFFS[TOP];
#pragma GCC unroll 8
    for (int k = 1; k <= (TOP - 1) / 2; k++) {
        r = r * s + TAN_STANDALONE_COEFFS[TOP - 2 * k];
    }
    r = r * a;

    v_if(i < 0) { r = -ckernel::sfpu::sfpu_reciprocal_iter<3>(r); }
    v_endif;
    return r;
}
#endif

#if defined(RANGE_REDUCTION_LOG) || defined(BASIS_NORMALIZED_LOG_ODDS_SEPARABLE)
// Log range reduction: x → (m, e_int) where x = 2^e * m, m ∈ [1, 2)
// log(x) = e*ln(2) + log(m)
// Uses IEEE 754 bit manipulation: exponent extraction and mantissa normalization
inline void log_reduce(vFloat x, vFloat& m, vInt& e_int) {
    // Extract biased exponent: for x = 2^e * m, biased_exp = e + 127
    vInt biased_exp = exexp(x, ExponentMode::Biased);

    // Unbias: e = biased_exp - 127
    // But we want m in [1, 2), so we set exponent to 127 (bias)
    // This gives m = x / 2^e = x * 2^(-e) = x with exponent = 127
    e_int = biased_exp - 127;

    // Normalize mantissa to [1, 2) by setting exponent bits to 127
    m = setexp(x, 127);
}

// Reconstruct: log(x) = e*ln(2) + log(m)
// poly_result = log(m) where m ∈ [1, 2)
// Final result = e*ln(2) + poly_result
inline vFloat log_expand(vFloat poly_result, vInt e_int) {
    // LOG_EXPAND_CONSTANT is base-specific: ln(2) for log, 1.0 for log2, log10(2) for log10
#ifndef LOG_EXPAND_CONSTANT
#define LOG_EXPAND_CONSTANT 0.6931471805599453f
#endif
    constexpr float EXPAND_C = LOG_EXPAND_CONSTANT;
    // int32_to_float expects SIGN-MAGNITUDE format, not two's complement.
    // Negative exponents must be converted: twos complement → sign-magnitude.
    v_if(e_int < 0) { e_int = as<vInt>(setsgn(as<vSMag>(~e_int + 1), 1)); }
    v_endif;
    vFloat e_float = convert<vFloat>(as<vSMag>(e_int), RoundMode::Nearest);
    return e_float * EXPAND_C + poly_result;
}
#endif

#ifdef RANGE_REDUCTION_CBRT
// Cbrt range reduction: x → (m, q, r, sign) where cbrt(x) = sign * C[r] * cbrt(m) * 2^q
//
// For x = sign(x) * |x| where |x| = m * 2^e with m ∈ [1, 2):
//   e = 3*q + r  where r ∈ {0, 1, 2}
//   cbrt(|x|) = cbrt(m * 2^r) * 2^q = cbrt(m) * cbrt(2^r) * 2^q
//   cbrt(x) = sign(x) * C[r] * cbrt(m) * 2^q
//
// where C = [1.0, cbrt(2), cbrt(4)]
//
// Split into cbrt_reduce_m (trivial, 2 ops) + cbrt_reduce_qrs (heavier)
// to avoid SFPU register spills. cbrt_reduce's many inlined temporaries
// exceed the 8 LREG limit when combined with polynomial evaluation variables.

// Scale constants: C[r] = cbrt(2^r) for r = 0, 1, 2
constexpr float CBRT_SCALE_C0 = 1.0f;
constexpr float CBRT_SCALE_C1 = 1.2599210498948732f;  // cbrt(2)
constexpr float CBRT_SCALE_C2 = 1.5874010519681994f;  // cbrt(4)

// Phase 1: Extract mantissa m ∈ [1, 2) — trivial, no register pressure
inline vFloat cbrt_reduce_m(vFloat x) { return setexp(setsgn(x, 0), 127); }

// Reconstruct: cbrt(x) = sign * C[r] * poly(m) * 2^q
inline vFloat cbrt_expand(vFloat poly_result, vInt q, vInt r, vInt sign_bits) {
    // Look up scale factor C[r] = cbrt(2^r)
    vFloat scale = CBRT_SCALE_C0;  // default r=0
    v_if(r == 1) { scale = CBRT_SCALE_C1; }
    v_endif;
    v_if(r == 2) { scale = CBRT_SCALE_C2; }
    v_endif;

    // result = C[r] * poly(m)
    vFloat result = scale * poly_result;

    // Multiply by 2^q via exponent addition
    vInt result_exp = exexp(result, ExponentMode::Biased);
    result = setexp(result, result_exp + q);

    // Apply original sign (cbrt is an odd function)
    vInt result_bits = as<vInt>(result);
    result = as<vFloat>(result_bits | sign_bits);

    return result;
}
#endif

// ============================================================================
// Hardware-exponent-ALU range reduction (exp2 / log2 / pow)
//
// Instead of software Cody-Waite (float k-int + 2^k + Newton reciprocal), these
// paths decompose the input with the SFPU exponent ALU (exexp / exman / setexp /
// addexp) and evaluate a LOW-DEGREE polynomial on the reduced domain — the same
// algorithm TTNN's native exp/log/sqrt use. The fitter owns WHAT to approximate
// and the natural-basis coefficients; the kernel owns the intrinsic decompose +
// scale fold + recombine. The evaluator tag owns this standalone path; the
// coefficient CSV owns the reduced-domain fit.
//
// These are standalone evaluators: when one of RANGE_REDUCTION_*_HW is defined
// the per-tile loop calls the matching exp_hw_eval / log_hw_eval / pow_hw_eval
// directly and the piecewise segment cascade is bypassed entirely.
// ============================================================================

// Launder a compile-time-constant vector through a named LREG so the SFPI
// compiler CANNOT constant-fold it back into per-element immediates. Measured
// failure mode of the plain "hoisted local" (FIX A): GCC constant-propagates
// `vFloat c127 = 127.0f` and re-materializes it INSIDE the 32-element dst loop
// as SFPADDI/SFPLOADI (Blackhole exp p3: `sfpmul+sfpaddi` instead of one
// SFPMAD, plus 2 SFPLOADI per spilled Horner coeff, every element). The
// l_reg write/read pair is the same mechanism TTNN's gelu/dropout kernels use
// for cross-call LREG state: the value enters the named LREG once (pre-loop
// SFPLOADI), and every later use is an opaque register read the constant
// folder cannot see through. Numerically a no-op (same bits, same ops).
// (Unconditional — HW_PRELOAD paths and the tan_standalone tail pin use it.)
template <sfpi::LRegs LR>
sfpi_inline vFloat hw_preload_pin(vFloat v) {
    l_reg[LR] = v;
    return l_reg[LR];
}

// ============================================================================
// TTI REPLAY LOWERING FRAMEWORK — shared with piecewise_rational.cpp; the
// full contract (opt-in shape, HAZARD RULES, capacity, register conventions)
// lives in tti_replay.h. Every eval kind's gate and body references it.
}  // namespace sfpi (framework macros are file-scope; reopened below)

#include "tti_replay.h"

namespace sfpi {

#if defined(EXPONENT_ALU_EXP2)
// exp(x) = 2^(x * log2e). Decompose x*log2e into integer i + fraction f via the
// hardware exponent ALU; 2^f via the fitter's degree-N poly; recombine setexp.
//
// The fitter emits NATURAL [0,1)-basis coeffs (EXP_HW_COEFFS, degree N) for
// g(f)=2^f. exman() returns the fraction scaled by 2^23, so we NORMALIZE it back
// to a float f in [0,1) (multiply by 2^-23) and run a plain degree-N Horner over
// the natural coeffs — exactly like pow_hw_eval normalizes its mantissa. (The old
// per-coeff 2^-23k fold was hardcoded to degree 2 and underflowed past degree 3.)
//
// The fitter tags EXP_HW_MULT = log2-domain multiplier (1.0 for 2^x, log2e for
// e^x, -log2e for exp(-x)) and an optional compose post-transform (minus_one for
// expm1, sigmoid for 1/(1+exp(-x))). Honor whatever the codegen emits.
#ifndef EXP_HW_MULT
#define EXP_HW_MULT 1.4426950216293334961f  // default 1/ln2 == log2(e)
#endif
#ifndef EXP_HW_EXPONENT_BIAS
#define EXP_HW_EXPONENT_BIAS 127
#endif

#if defined(EXP_HW_FUSED)
// Per-coeff scale that folds the exman 2^23-normalize into the Horner
// coefficients: coeff at polynomial power k is multiplied by 2^-23k so the
// Horner can run over the UNNORMALIZED exman fraction (TTNN exp_21f trick).
// constexpr so the compiler bakes it in — zero runtime cost.
constexpr float exp_hw_fused_scale(int k) {
    float s = 1.0f;
    for (int i = 0; i < k; i++) {
        s *= 0x1p-23f;
    }
    return s;
}
#endif

// ----------------------------------------------------------------------------
// TTI REPLAY fast path (closes the issue-bandwidth gap vs TTNN's exp_21f).
//
// The sfpi loop issues every SFPU instruction individually from the RISC-V
// frontend PLUS per-element loop overhead (2 address adds + addi + bne + two
// MMIO instruction pushes for the dynamic-offset SFPLOAD/SFPSTORE). TTNN's
// _sfpu_exp_21f_bf16_tti_ instead RECORDS its per-element body into Tensix
// replay-buffer slot 0 once and then issues ONE TTI_REPLAY per element; the
// body streams back-to-back at Tensix rate with the dst walk done by an
// ADDR_MOD_6 dest+=2 auto-increment on SFPSTORE (no address computation, no
// loop-control issue slots).
//
// This path adopts exactly that mechanism for the clean exp kernel and also
// adopts TTNN's SETEXP trick: after SFPSHFT, LREG0 holds z = (int)(xlog2*2^23),
// whose bits 30:23 ARE the biased integer-part exponent (i+127). SFPSETEXP with
// mod1=ARG_EXPONENT reads the exponent field from lreg_dest in place — the
// second SFPEXEXP and the SFPMOV the sfpi path pays for setexp(p, ep) both
// vanish. The body processes TWO elements (A/B lanes interleaved) in 26 Tensix
// instructions = 13 per element (TTNN accurate exp: 16 per element); the
// interleave exists because SFPMAD results are NOT readable in the very next
// replay-stream cycle (see the hazard comment at the body). Per element:
//
//   SFPLOAD  Lx                       x               (ADDR_MOD_7, no incr)
//   SFPMAD   Lx = Lx*L12 + L5         xlog2 = x*mult + 127     (prgm0, pin)
//   SFPEXEXP Le <- Lx (debiased)      e
//   SFPEXMAN Lx <- Lx (implicit one)  m
//   SFPSHFT  Lx <<= Le                z bits (int part in exp field)
//   SFPEXMAN Le <- Lx (PAD9)          frac * 2^23
//   SFPCAST  Le <- Le                 f (fp32)
//   SFPMAD   Lp = Le*L13 + L14        c3f*f + c2f              (prgm1, prgm2)
//   SFPMAD   Lp = Lp*Le + L7          + c1f                    (pin)
//   SFPMAD   Le = Lp*Le + L10         + c0 == 1.0              (hw const 1.0)
//   SFPSETEXP Lx = mantissa(Le) exp(Lx)   bare recombine (ARG_EXPONENT)
//   SFP_STOCH_RND Lx                  fp32 -> bf16 RNE
//   SFPSTORE Lx                       (B store: ADDR_MOD_6, dest += 4)
//
// Numerics are bit-identical to exp_hw_eval_preloaded under the same gates
// (FUSED + BARE_SETEXP + SKIP_INPUT_CLAMP): same MAD order, same coefficients,
// and SETEXP/ARG_EXPONENT consumes exactly the exexp(z, NoDebias) bits.
// Silicon: pure ULP 0.5268874764442444 — byte-identical to the sfpi path —
// at 1.51-1.52 us vs 1.94-1.95 us sfpi and 1.90 us TTNN (chip 0, same session).
//
// Safety vs the wave-3 sin/cos dst-drift corruption: the walk is NOT sfpi
// dst_reg++ (software offset). It is the hardware D-RWC counter, advanced +4
// per body by ADDR_MOD_6 on the B-lane SFPSTORE, re-zeroed every tile by the
// SETRWC the per-tile scaffolding already issues after copy_tile, and re-zeroed
// AGAIN after the replays because the next tile's A2D datacopy MOP shares the
// counter (see the trailing TTI_SETRWC) — silicon-verified byte-identical.
//
// Gates (structural family, C-LOW-1): bf16 only (the body bakes in the
// FP32_TO_FP16B round), plain exp only (no compose — the reciprocal owns
// prgm0 there), FUSED + SKIP_CLAMP (the body shapes bake both cuts in), and
// degree 2 or 3. TWO body variants cover the bare/non-bare split the fitter
// produces (c0 >= 1 fits get EXP_HW_BARE_SETEXP; c0 < 1 fits do not):
//
//   BARE (EXP_HW_BARE_SETEXP defined) — the 2-element interleaved body above
//   with the 1-slot SETEXP(ARG_EXPONENT) recombine. Degree 3 requires
//   c0 == 1.0 (c0 folds into the hardware 1.0 constant register; L5/L7 are
//   taken by 127.0 and c1f). Degree 2 frees L7 (c1f lives in prgm2), so any
//   c0 pins there. KNOWN FALLBACK (C-LOW-3): degree-3 bare with c0 != 1.0 —
//   the interleave uses all 8 general LREGs (A: L0-L2, B: L3/L4/L6, pins
//   L5/L7), so a non-1.0 c0 has no register; no corpus fit has this shape
//   (deg-3 [0,1) 2^f minimax lands c0 <= 1.0: exactly 1.0 when pinned,
//   < 1.0 otherwise, which is the non-bare arm below).
//
//   PE-CORRECTED (no EXP_HW_BARE_SETEXP, c0 < 1.0) — single-element body
//   (13 + 2*DEG slots): the recombine transcribes the sfpi twin's NoDebias
//   2^i synthesis (`p * as<vFloat>(ep << 23)`, see exp_hw_eval_preloaded):
//   EXEXP(NoDebias) + SHFT-imm-23 + SFPMUL replace the bare SETEXP. The
//   pattern is lifted 1:1 from the silicon-proven kSigmoidTtiReplayD3 body
//   (same EXEXP/SHFT/MAD encodings), minus the reciprocal compose. A 2x
//   interleave cannot fit: deg 3 needs c1f AND c0 pinned (L5+L7+one more)
//   on top of 2x3 lane registers = 9 > 8 general LREGs.
//
// EXP_HW_TTI_DISABLE=defined lets A/B measurement force the sfpi path for
// the whole family.
#if defined(HW_PRELOAD) && defined(EXP_HW_FUSED) &&                                   \
    defined(EXP_HW_SKIP_INPUT_CLAMP) && !defined(EXP_HW_COMPOSE_SIGMOID) &&           \
    !defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) &&                                      \
    !defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) &&                            \
    !defined(EXP_HW_COMPOSE_MINUS_ONE) &&                                             \
    !defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) &&                           \
    !defined(EVAL_METHOD_ABS_RESIDUAL_AFFINE_1) &&                                    \
    defined(USE_BF16) && defined(ARCH_BLACKHOLE) && !defined(EXP_HW_TTI_DISABLE) && \
    !defined(TT_DOMAIN_ACTION_PROGRAM)
#define EXP_HW_TTI_CANDIDATE 1
// BARE: 2-element interleaved body: per element LOAD + (1 + DEG) MADs +
// EXEXP + EXMAN + SHFT + EXMAN + CAST + SETEXP + STOCH_RND + STORE = 10 + DEG.
constexpr uint32_t kExpHwTtiRouteSlots =
#if defined(TT_EXPONENT_ALU_EXP2_TERMINAL_CLAMP)
    4u;
#else
    0u;
#endif
constexpr uint32_t kExpHwTtiBodySlots =
    2u * (10u + EXP_HW_DEGREE) + kExpHwTtiRouteSlots;
// PE-CORRECTED: single-element body: LOAD + MAD+gap + EXEXP + EXMAN + SHFT +
// EXEXP(NoDebias) + EXMAN(PAD9) + CAST + DEG x (MAD+gap, last gap = SHFT-imm)
// + MUL + gap + STOCH_RND + STORE = 13 + 2*DEG.
constexpr uint32_t kExpHwTtiPeBodySlots = 13u + 2u * EXP_HW_DEGREE;
#if defined(EXP_HW_BARE_SETEXP)
constexpr bool kExpHwTtiReplayBare =
    ((EXP_HW_DEGREE == 3 && EXP_HW_COEFFS[0] == 1.0f) || EXP_HW_DEGREE == 2) &&
    tti_replay_fits(kExpHwTtiBodySlots);
constexpr bool kExpHwTtiReplayPe = false;
#else
constexpr bool kExpHwTtiReplayBare = false;
constexpr bool kExpHwTtiReplayPe =
    (EXP_HW_DEGREE == 2 || EXP_HW_DEGREE == 3) && tti_replay_fits(kExpHwTtiPeBodySlots);
#endif
constexpr bool kExpHwTtiReplay = kExpHwTtiReplayBare || kExpHwTtiReplayPe;
#if defined(TT_INLINE_PROGRAM_OVERLAY) && \
    !defined(TT_INLINE_PROGRAM_TARGET_EXACT_CORE_IDENTITY) && \
    !defined(TT_EXPONENT_ALU_EXP2_TERMINAL_CLAMP)
static_assert(
    !kExpHwTtiReplay,
    "typed core overlay requires route-aware EXP_TTI replay; refusing silent fallback");
#endif
#else
#define EXP_HW_TTI_CANDIDATE 0
constexpr bool kExpHwTtiReplay = false;
constexpr bool kExpHwTtiReplayBare = false;
constexpr bool kExpHwTtiReplayPe = false;
#endif

// ----------------------------------------------------------------------------
// TTI REPLAY fast path for the sigmoid / silu (sigmoid-product) compose
// kernels. Same replay mechanism as kExpHwTtiReplay above; single-element
// body (25-30 ops/element — a 2x interleave cannot fit 32 slots) with
// explicit gap slots after every MAD-family producer (see the trig replay
// comment for the gap discipline).
//
// TWO body shapes, each a 1:1 transcription of its sfpi/GCC BH loop:
//
// (a) DEG==2 + BARE_SETEXP (objdump of sigmoid_p2/silu_p2, ARECIP+one-Newton
//     reciprocal) with two proven, bit-identical substitutions carried over
//     from the exp replay path:
//   - SFPSETEXP(ARG_EXPONENT) reads the integer-part exponent field from
//     the z register in place, deleting the second SFPEXEXP + SFPMOV the
//     sfpi path pays (silicon-proven byte-identical on exp under the same
//     FUSED + BARE_SETEXP gates).
//   - The fraction is EXMAN'd into a scratch register instead of over the
//     z register (pure register allocation; values untouched).
//
// (b) DEG==3 + NON-bare (objdump of silu_p3/swish_p3, c0 < 1 so the bare
//     setexp is invalid). The sfpi path synthesizes the 2^i scale from the
//     NoDebias exponent (`p * as<vFloat>(ep << 23)`, see
//     exp_hw_eval_preloaded), and GCC fuses that multiply with the compose's
//     `1.0f + y` into ONE single-rounded MAD: u = p*2^i + 1.0. The fusion is
//     bit-identical to the two-op form because p*2^i is an exact
//     power-of-two scale (same mantissa, adjusted exponent). vs shape (a)
//     this adds one Horner MAD+gap (deg 3) and replaces the 1-slot bare
//     SETEXP recombine + SFPADD(u=y+1) + gap with EXEXP(NoDebias) +
//     SHFT-imm-23 + fused MAD (the two bit-unit ops double as the last
//     Horner rung's gap): 28 slots plain / 30 with the silu product.
//
// The reciprocal Newton chain keeps the exact GCC instruction encodings:
//   [u ready] -> SFPARECIP -> SFPMAD(t = u*r0 - prgm0, mod1=2) ->
//   SFPSETCC(t<0) -> SFPMAD(y = r0*-t - 0, mod1=3, predicated) -> SFPENCC,
// with gap slots between every producer/consumer pair (the auditor's
// stream-rate hazard: at inline issue rate the RISC-V frontend spaces these
// for free; in a replay stream they run back-to-back). SFPARECIP is given a
// 2-slot window conservatively. prgm0 (LREG12) holds 2.0 programmed by
// sfpu_reciprocal_init and is never written here.
//
// Gates: BH + bf16 + FUSED + SKIP_CLAMP + sigmoid/silu compose, and then
// shape (a) degree 2 + BARE_SETEXP (the selected sigmoid/sigmoid_accurate
// configs) or shape (b) degree 3 without BARE_SETEXP (the selected
// silu/swish p3 configs). EXP_HW_TTI_DISABLE=defined forces the sfpi path
// for A/B.
#if defined(HW_PRELOAD) && defined(EXP_HW_FUSED) &&                                 \
    defined(EXP_HW_SKIP_INPUT_CLAMP) &&                                             \
    (defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) ||  \
     defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)) &&                         \
    defined(USE_BF16) && defined(ARCH_BLACKHOLE) && !defined(EXP_HW_TTI_DISABLE) && \
    !defined(TT_DOMAIN_ACTION_PROGRAM)
#define EXP_HW_COMPOSE_TTI_CANDIDATE 1
#if defined(EXP_HW_BARE_SETEXP)
#if defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
constexpr bool kSigmoidTtiReplayD2Bare = false;
constexpr bool kSymmetricSigmoidProductTtiReplayD2Bare =
    (EXP_HW_DEGREE == 2);
#else
constexpr bool kSigmoidTtiReplayD2Bare = (EXP_HW_DEGREE == 2);
constexpr bool kSymmetricSigmoidProductTtiReplayD2Bare = false;
#endif
constexpr bool kSigmoidTtiReplayD3 = false;
#else
constexpr bool kSigmoidTtiReplayD2Bare = false;
constexpr bool kSymmetricSigmoidProductTtiReplayD2Bare = false;
#if defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
constexpr bool kSigmoidTtiReplayD3 = false;
#else
constexpr bool kSigmoidTtiReplayD3 = (EXP_HW_DEGREE == 3);
#endif
#endif
constexpr bool kSigmoidTtiReplay =
    kSigmoidTtiReplayD2Bare || kSigmoidTtiReplayD3 ||
    kSymmetricSigmoidProductTtiReplayD2Bare;
#if defined(TT_INLINE_PROGRAM_OVERLAY) && \
    !defined(TT_INLINE_PROGRAM_TARGET_EXACT_CORE_IDENTITY) && \
    !defined(TT_EXPONENT_ALU_EXP2_TERMINAL_CLAMP)
static_assert(
    !kSigmoidTtiReplay,
    "typed core overlay requires route-aware EXP_COMPOSE_TTI replay; refusing silent fallback");
#endif
#else
#define EXP_HW_COMPOSE_TTI_CANDIDATE 0
constexpr bool kSigmoidTtiReplayD2Bare = false;
constexpr bool kSigmoidTtiReplayD3 = false;
constexpr bool kSymmetricSigmoidProductTtiReplayD2Bare = false;
constexpr bool kSigmoidTtiReplay = false;
#endif

// Anonymous bounded-exp two-sided rational.  A 32-slot recorded body owns
// the ordinary P2 exponent leaf, one quadratic denominator/reciprocal, and
// both coefficient-declared reconstructions.  Its final sign predicate is
// retired by the same-row suffix (predicated move/ENCC/RNE/store), so
// no second element loop or DST shadow is introduced.
#if defined(HW_PRELOAD) && defined(EXP_HW_FUSED) &&                  \
    defined(EXP_HW_BARE_SETEXP) && defined(EXP_HW_SKIP_INPUT_CLAMP) && \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) &&           \
    defined(USE_BF16) && defined(ARCH_BLACKHOLE) &&                 \
    !defined(EXP_HW_TTI_DISABLE) && !defined(TT_DOMAIN_ACTION_PROGRAM)
#define EXP_HW_BOUNDED_TWO_SIDED_TTI_CANDIDATE 1
constexpr uint32_t kBoundedTwoSidedTtiBodySlots =
    TT_BOUNDED_EXP_TWO_SIDED_BODY_SLOTS;
constexpr uint32_t kBoundedTwoSidedTtiSuffixSlots =
    TT_BOUNDED_EXP_TWO_SIDED_SUFFIX_SLOTS;
constexpr bool kBoundedTwoSidedTtiReplay =
    EXP_HW_DEGREE == 2u && kBoundedTwoSidedTtiBodySlots == 32u &&
    kBoundedTwoSidedTtiSuffixSlots == 14u &&
    tti_replay_fits(kBoundedTwoSidedTtiBodySlots);
#else
#define EXP_HW_BOUNDED_TWO_SIDED_TTI_CANDIDATE 0
constexpr bool kBoundedTwoSidedTtiReplay = false;
#endif

// Whole-core even exponential composition.  The closed descriptor supplies
// abs(x), exponent bias 126 (so y=exp(abs(x))/2), and an fp32 compose scale
// for y+scale/y.  The scale is fitted on the ordinary coefficient lattice to
// absorb the selected reciprocal lane's rounding, without adding a replay op.
#if defined(HW_PRELOAD) && defined(EXP_HW_FUSED) && defined(EXP_HW_SKIP_INPUT_CLAMP) && \
    (defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR)) && defined(EXP_HW_INPUT_ABS) && \
    (EXP_HW_EXPONENT_BIAS == 126) && defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    !defined(EXP_HW_TTI_DISABLE) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    !defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER)
#define EXP_HW_HYPERBOLIC_TTI_CANDIDATE 1
constexpr bool kHyperbolicEvenTtiReplay =
#if defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN)
    (EXP_HW_DEGREE == 4);
#else
    false;
#endif
constexpr bool kHyperbolicOddFactorTtiReplay =
#if defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    (EXP_HW_DEGREE == 4) && (EXP_HW_ORIGIN_CUBIC == 0.166015625f);
#else
    false;
#endif
constexpr uint32_t kHyperbolicEvenTtiBodySlots = 32u;
// The odd core stops after the cancellation-safe exponential magnitude. Its
// ordinary sign/lower-bound/RNE/store terminal executes directly after every
// replay on the same row; it is deliberately not squeezed into dependency
// gaps whose coefficient sources are still live in the SFPU pipeline.
constexpr uint32_t kHyperbolicOddTtiCoreSlots = 29u;
static_assert(tti_replay_fits(kHyperbolicEvenTtiBodySlots));
static_assert(tti_replay_fits(kHyperbolicOddTtiCoreSlots));
#else
#define EXP_HW_HYPERBOLIC_TTI_CANDIDATE 0
constexpr bool kHyperbolicEvenTtiReplay = false;
constexpr bool kHyperbolicOddFactorTtiReplay = false;
constexpr uint32_t kHyperbolicEvenTtiBodySlots = 0u;
constexpr uint32_t kHyperbolicOddTtiCoreSlots = 0u;
#endif

template <uint32_t DEG>
inline vFloat exp_hw_eval(vFloat x) {
    constexpr float MULT = EXP_HW_MULT;
#if defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
    vFloat x_raw = x;
#endif

#if defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    vFloat x_original = x;
#endif
#if defined(EXP_HW_INPUT_ABS)
    x = setsgn(x, 0);
#endif
    vFloat xlog2 = x * MULT + (float)EXP_HW_EXPONENT_BIAS;

    // Full-range safety clamp: keep xlog2 in [0, 255] so the implicit float->int
    // conversion below cannot wrap (TTNN does this in its non-unsafe path).
    vFloat thr_lo = 0.0f;
    vFloat thr_hi = 255.0f;
#if !defined(EXP_HW_SKIP_INPUT_CLAMP)
    ordered_min_max(thr_lo, xlog2);  // xlog2 = max(0, xlog2)
    ordered_min_max(xlog2, thr_hi);  // xlog2 = min(xlog2, 255)
#endif

    // Branch-free float->int: shift mantissa left by (exp - bias) bits.
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);

    vInt ep = exexp(z, ExponentMode::Biased);  // 2^(integer part)
    vMag fm = exman(z);                          // fraction * 2^23
    // Normalize the exman 2^23-scaled fraction back to a float f in [0,1).
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest) * 0x1p-23f;

    // Plain degree-N Horner for 2^f over the natural [0,1) coeffs.
    const float* c = EXP_HW_COEFFS;
    vFloat p = c[DEG];
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        p = p * f + c[k];
    }

    // Recombine 2^i * 2^f. `ep` is the biased exponent of the integer part
    // (== i + 127). setexp only REPLACES p's exponent field, keeping its
    // mantissa — which is correct only when p in [1,2) (exponent field 127).
    // The fitter's natural [0,1) fit makes g(0)=c[0] dip just below 1.0 for some
    // degrees (e.g. odd-degree exp2: c0=0.99992), putting p in [0.5,1) at f~=0
    // (exponent field 126). Replacing that with ep then over-scales by 2x. Add
    // p's own exponent deviation from the bias so the integer part composes with
    // p's actual magnitude (mirrors pow_hw_eval's setexp(s, s_exp + q)).
    vInt pe = exexp(p, ExponentMode::Biased);
    vFloat y = setexp(p, ep + pe - 127);  // 2^i * 2^f == base^x

    // Optional compose post-transform (fitter folds the activation around exp2).
#if defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
    constexpr float S = 1.0f / EXP_HW_COMPOSE_SCALE;
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(S + y);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(S + y);
#endif
    vFloat magnitude = setsgn(x_raw, 0);
    vFloat result = magnitude * (S * inv);
    v_if(x_raw < 0.0f) { result = 0.0f - (magnitude * y) * inv; }
    v_endif;
    y = result;
#elif defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    // y == exp(-x); sigmoid(x) = 1 / (1 + exp(-x)).
#if defined(USE_BF16)
    y = ckernel::sfpu::sfpu_reciprocal_iter<1>(1.0f + y);
#else
    y = ckernel::sfpu::sfpu_reciprocal_iter<2>(1.0f + y);
#endif
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    y = x * y;
#endif
#elif defined(EXP_HW_COMPOSE_MINUS_ONE)
    // expm1(x) = exp(x) - 1.
    y = y - 1.0f;
#elif defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN)
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(y);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(y);
#endif
    y = EXP_HW_COMPOSE_SCALE * inv + y;
#elif defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR)
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(y);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(y);
#endif
    vFloat magnitude = y - EXP_HW_COMPOSE_SCALE * inv;
    vFloat ax = setsgn(x_original, 0);
    vFloat lower = ax * (1.0f + EXP_HW_ORIGIN_CUBIC * ax * ax);
    ordered_min_max(magnitude, lower);
    y = copysgn(magnitude, x_original);
#elif defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    vFloat square = y * y;
    vFloat denominator =
        (2.0f * EXP_HW_TWO_SIDED_SCALE) * y +
        square +
        2.0f * EXP_HW_TWO_SIDED_SCALE * EXP_HW_TWO_SIDED_SCALE;
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(denominator);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(denominator);
#endif
    vFloat negative_numerator =
        EXP_HW_TWO_SIDED_NEG_SLOPE * y + EXP_HW_TWO_SIDED_NEG_BIAS;
    vFloat negative =
        (x_original * y) * (negative_numerator * inv);
    vFloat positive =
        x_original * (1.0f - EXP_HW_TWO_SIDED_POS_SCALE * inv);
    y = positive;
    v_if(x_original < 0.0f) { y = negative; }
    v_endif;
#endif
    return y;
}

#if defined(EVAL_METHOD_ABS_RESIDUAL_AFFINE_1)
// Closed anonymous whole-core form:
//   t = exp(-abs(x)); residual = t * N(t)/D(t)
// followed by one enum-selected affine reconstruction.  Both coefficient
// arrays are ordinary fitter artifacts joined by activation JSON in codegen;
// this body contains no activation identity, expression interpreter, fallback,
// second element loop, or reader/writer specialization.
template <uint32_t NUM_DEG, uint32_t DEN_DEG>
inline vFloat abs_residual_affine_1_eval(vFloat x) {
#if defined(ABS_RESIDUAL_COORDINATE_TERMINAL)
    vFloat coordinate = setsgn(x, 0);
    vFloat coordinate_bound = ABS_RESIDUAL_COORDINATE_TERMINAL_BOUND;
    ordered_min_max(coordinate, coordinate_bound);
    vFloat t = exp_hw_eval<EXP_HW_DEGREE>(coordinate);
#else
    vFloat t = exp_hw_eval<EXP_HW_DEGREE>(x);
#endif
#if defined(ABS_RESIDUAL_CORRECTION_POLY)
    vFloat quotient = ABS_RESIDUAL_NUM_COEFFS[NUM_DEG];
#pragma GCC unroll 8
    for (int k = (int)NUM_DEG - 1; k >= 0; k--) {
        quotient = quotient * t + ABS_RESIDUAL_NUM_COEFFS[k];
    }
#else
    vFloat num = ABS_RESIDUAL_NUM_COEFFS[NUM_DEG];
#pragma GCC unroll 8
    for (int k = (int)NUM_DEG - 1; k >= 0; k--) {
        num = num * t + ABS_RESIDUAL_NUM_COEFFS[k];
    }
    vFloat den = ABS_RESIDUAL_DEN_COEFFS[DEN_DEG];
#pragma GCC unroll 8
    for (int k = (int)DEN_DEG - 1; k >= 0; k--) {
        den = den * t + ABS_RESIDUAL_DEN_COEFFS[k];
    }
#if defined(RATIONAL_RECIPROCAL_ZERO_ITER)
    vFloat quotient = num * ckernel::sfpu::sfpu_reciprocal_iter<0>(den);
#elif defined(RATIONAL_RECIPROCAL_ONE_ITER)
    vFloat quotient = num * ckernel::sfpu::sfpu_reciprocal_iter<1>(den);
#else
    vFloat quotient = num * ckernel::sfpu::sfpu_reciprocal_iter<2>(den);
#endif
#endif
    vFloat residual = t * quotient;
    vFloat zero = 0.0f;
#if defined(ABS_RESIDUAL_POSITIVE_PART_PLUS)
    vFloat affine = x;
    ordered_min_max(zero, affine);  // affine=max(x,0)
    return affine + residual;
#elif defined(ABS_RESIDUAL_NEGATIVE_PART_MINUS)
    vFloat affine = x;
    ordered_min_max(affine, zero);  // affine=min(x,0)
    return affine - residual;
#elif defined(ABS_RESIDUAL_FACTOR_BEFORE_SCALE_PRODUCT)
#error "factor-before-scale reconstruction requires retained exponent state"
#else
#error "abs_residual_affine_1 reconstruction is missing"
#endif
}
#endif

#endif  // EXPONENT_ALU_EXP2 (native whole-tile fusions are evaluator-neutral)

#if defined(EVAL_METHOD_ABS_SQUARE_DECAY_1) || \
    defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1) || \
    defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1) || \
    defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1)
// One Newton step for a coordinate whose typed structural form owns zero,
// infinity, and the SFPARECIP exponent boundary after reconstruction.  The
// stock helper predicates the correction to preserve its standalone special
// semantics.  Those predicate issues are redundant here: every ordinary
// coordinate is finite, nonzero, and below 2^126, while the fixed form's
// terminal actions overwrite all exceptional lanes before returning.
inline vFloat typed_finite_coordinate_reciprocal_iter1(vFloat x) {
    vFloat y = sfpi::approx_recip(x);
    vFloat residual = x * y - sfpi::vConstFloatPrgm0;
    return y * -residual - 0.0f;
}
#endif

#if defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1)
// Fixed anonymous reconstruction around an exact declared root.  The
// coefficient payload is an ordinary frontier P leaf in u=1/x; this common
// evaluator owns only the structural coordinate transform and native-log
// primitive.
template <uint32_t degree>
inline vFloat root_native_log_reciprocal_1_eval(vFloat x, uint32_t handoff_row) {
#if defined(TT_ROOT_NATIVE_LOG_TOTAL_DOMAIN)
    static_assert(degree == TT_ROOT_NATIVE_LOG_DEGREE);
    auto polynomial = [](vFloat coordinate, const float* coefficients,
                         uint32_t leaf_degree) {
        vFloat value = coefficients[leaf_degree];
#pragma GCC unroll 16
        for (int index = static_cast<int>(leaf_degree) - 1; index >= 0; --index) {
            value = value * coordinate + coefficients[index];
        }
        return value;
    };
    auto normalized_log = [&](vFloat coordinate) {
        vInt exponent = exexp(coordinate, ExponentMode::Biased) - 127;
        vFloat mantissa = setexp(coordinate, 127);
        v_if(mantissa >= 1.5f) {
            mantissa = mantissa * 0.5f;
            exponent = exponent + 1;
        }
        v_endif;
        vInt exponent_magnitude = exponent;
        v_if(exponent < 0) { exponent_magnitude = ~exponent + 1; }
        v_endif;
        vFloat exponent_value =
            convert<vFloat>(as<vSMag>(exponent_magnitude), RoundMode::Nearest);
        v_if(exponent < 0) { exponent_value = -exponent_value; }
        v_endif;
        return exponent_value * 0.6931471805599453f +
               polynomial(mantissa, TT_ROOT_TOTAL_LOG_COEFFS,
                          TT_ROOT_TOTAL_LOG_DEGREE);
    };

    // One structural coordinate owns all positive, recurrence, and reflected
    // consumers.  The BF16 lattice makes every negative |x|>=128 an integer
    // pole, so periodic reduction is needed only on the bounded remainder.
    vFloat z = setsgn(x, 0) + 1.0f;
    v_if(x >= 1.0f) { z = x; }
    v_endif;

    {
        // Finish normalization before opening the reciprocal/Horner chain.
        // Otherwise exponent, mantissa, reciprocal, and correction overlap
        // beyond the eight physical SFPU registers.
        vFloat log_z = normalized_log(z);
        vFloat inverse = typed_finite_coordinate_reciprocal_iter1(z);
        vFloat correction = TT_ROOT_NATIVE_LOG_COEFFS[degree];
#pragma GCC unroll 8
        for (int index = degree - 1; index >= 0; --index) {
            correction =
                correction * inverse + TT_ROOT_NATIVE_LOG_COEFFS[index];
        }
        vFloat core_result =
            (z - TT_ROOT_NATIVE_LOG_ROOT) * (log_z + correction);
        v_if(z < 2.0f) {
            vFloat unit_coordinate = z;
            vFloat unit_upper = 2.0f;
            ordered_min_max(unit_coordinate, unit_upper);
            vFloat unit_factor =
                (unit_coordinate - 1.0f) * (unit_coordinate - 2.0f);
            core_result =
                unit_factor *
                polynomial(unit_coordinate, TT_ROOT_TOTAL_UNIT_COEFFS,
                           TT_ROOT_TOTAL_UNIT_DEGREE);
        }
        v_endif;
        dst_reg[handoff_row].mode<DataLayout::F32>() = core_result;
    }

    // Recreate the bounded fraction only after the positive core is complete;
    // keeping it live across the two polynomial chains spills the eight-LREG
    // SFPU.  This remains one element pass and one common evaluator.
    vFloat bounded_negative = x;
    vFloat negative_lattice_bound = -128.0f;
    ordered_min_max(negative_lattice_bound, bounded_negative);
    vFloat zero_bound = 0.0f;
    ordered_min_max(bounded_negative, zero_bound);
    constexpr float kRound = 0x1.8p23f;
    vFloat nearest = bounded_negative + kRound;
    nearest = nearest - kRound;
    vFloat fraction = bounded_negative - nearest;
    vFloat fraction_magnitude = setsgn(fraction, 0);
    vFloat secondary_log_coordinate = 1.0f;
    v_if((x < 0.0f) && (fraction_magnitude != 0.0f)) {
        secondary_log_coordinate = fraction_magnitude;
    }
    v_endif;
    v_if((x > 0.0f) && (x < 1.0f)) { secondary_log_coordinate = x; }
    v_endif;
    vFloat secondary_log = normalized_log(secondary_log_coordinate);
    vFloat log_sinc =
        polynomial(fraction_magnitude, TT_ROOT_TOTAL_SINC_COEFFS,
                   TT_ROOT_TOTAL_SINC_DEGREE);
    vFloat result = dst_reg[handoff_row].mode<DataLayout::F32>();
    v_if((x > 0.0f) && (x < 1.0f)) { result = result - secondary_log; }
    v_endif;
    v_if(x < 0.0f) { result = -secondary_log - log_sinc - result; }
    v_endif;
    v_if((x < 0.0f) && (fraction == 0.0f)) {
        result = std::numeric_limits<float>::infinity();
    }
    v_endif;
    v_if(is_zero(x)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    return result;
#else
    (void)handoff_row;
    static_assert(degree == TT_ROOT_NATIVE_LOG_DEGREE);
    vFloat inverse = typed_finite_coordinate_reciprocal_iter1(x);
    vFloat correction = TT_ROOT_NATIVE_LOG_COEFFS[degree];
#pragma GCC unroll 8
    for (int index = degree - 1; index >= 0; --index) {
        correction = correction * inverse + TT_ROOT_NATIVE_LOG_COEFFS[index];
    }
    vFloat log_value = ckernel::sfpu::_calculate_log_body_no_init_(x);
    return (x - TT_ROOT_NATIVE_LOG_ROOT) * (log_value + correction);
#endif
}
#endif

#if defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1)
// Two-resource ordinary replay. The finite enum fixes the recurrence and
// reflection order; both coefficient arrays are materialized JSON resources.
// One predicate-free Newton correction is exact enough for every ordinary
// BF16 coordinate.
// Only x == 2^126 needs an explicit reconstruction below: its reciprocal is
// MIN_NORMAL, while SFPARECIP intentionally maps inputs >= 2^126 to zero.
inline vFloat shared_inverse_square_recurrence_p0_p2_1_eval(vFloat x) {
    vFloat zero = 0.0f;
    vFloat one = 1.0f;
    // psi1(x)=psi1(x+1)+1/x^2 for every positive x. Taking that one
    // recurrence step unconditionally makes the positive and reflection arms
    // share z=1+|x| and the same inverse-square pipeline.
    vFloat z = setsgn(x, 0) + one;
    constexpr float kRound = 0x1.8p23f;
    vFloat n = x + kRound;
    n = n - kRound;
    vFloat signed_one = copysgn(one, x);
    vFloat negative_mask = 0.5f - 0.5f * signed_one;
    vFloat q = x - negative_mask * n;
    vFloat u = typed_finite_coordinate_reciprocal_iter1(z);
    // Same Bernoulli core, in Horner order: one fewer issued arithmetic op
    // and one less live temporary than the expanded u/u^2 reconstruction.
    vFloat core = ((SHARED_INVERSE_SQUARE_P0 * u + 0.5f) * u + one) * u;
    vFloat iq = typed_finite_coordinate_reciprocal_iter1(q);
    // Mask before squaring: on the positive arm the regularizer is dead, but
    // evaluating it at q=x would overflow for large finite BF16 inputs and
    // turn the final 0*regularizer into NaN.
    vFloat regularizer_q = negative_mask * q;
    vFloat q2 = regularizer_q * regularizer_q;
    vFloat regularizer =
        (SHARED_INVERSE_SQUARE_P2[2] * q2 + SHARED_INVERSE_SQUARE_P2[1]) * q2 +
        SHARED_INVERSE_SQUARE_P2[0];
    vFloat result = iq * iq + copysgn(core, x);
    result = negative_mask * regularizer + result;
    v_if((x < zero) && (q == zero)) {
        result = std::numeric_limits<float>::infinity();
    }
    v_endif;
    vInt source_exponent = exexp(setsgn(x, 0), ExponentMode::Biased);
    v_if(source_exponent == 255) { result = zero; }
    v_endif;
    v_if((source_exponent == 255) && (x < zero)) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;
    v_if(is_zero(x)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    v_if(x == 0x1p126f) { result = 0x1p-126f; }
    v_endif;
    return result;
}

#if defined(ARCH_BLACKHOLE) && defined(USE_BF16)
// Physical single-row replay for the same anonymous closed form.  The body is
// exactly 32 issued words.  It deliberately interleaves the two one-Newton
// reciprocal chains and the P0/P2 Horner chains so every MAD/ARECIP producer
// has an independent issue before its consumer.  L7 is the sole replay-entry
// invariant (0.5); reciprocal init owns Prgm0=2, while P2 c1/c0 occupy
// Prgm1/2.  All other LREGs are dead at replay exit.
//
// The same-row suffix owns the finite/special cases not produced by the
// arithmetic itself: NaN/-Inf class, +Inf -> +0, and +2^126 -> MIN_NORMAL.  Raw BF16 in BH DST
// has sign15|mantissa14:8|exponent7:0, hence the exact physical words 0x00ff
// and 0x00fd.  NaNs are not selected and remain NaN.  This suffix runs after
// each replayed row, not as a second tile loop.
constexpr uint32_t kSharedInverseSquareBodySlots = 32u;
#if defined(TT_SHARED_INVERSE_SQUARE_TARGET_CLASS_REPAIR)
constexpr uint32_t kSharedInverseSquareSuffixSlots = 38u;
constexpr uint16_t kSharedInverseSquareFiniteThresholdBiasBf16 =
    __builtin_bit_cast(uint32_t, -TT_SHARED_INVERSE_SQUARE_FINITE_THRESHOLD) >> 16;
constexpr uint16_t kSharedInverseSquareZeroTransitionBiasBf16 =
    __builtin_bit_cast(
        uint32_t, TT_SHARED_INVERSE_SQUARE_ZERO_TRANSITION_PREVIOUS_MAGNITUDE) >> 16;
static_assert(
    TT_SHARED_INVERSE_SQUARE_POSITIVE_ZERO_FIRST_RAW == 0x7e80u,
    "shared inverse-square target positive-zero boundary changed");
#else
constexpr uint32_t kSharedInverseSquareSuffixSlots = 20u;
#endif
#if TT_SPECIAL_NEG_INF == 3
constexpr uint16_t kSharedInverseSquareNegInfResultBf16 = 0x0000u;
#elif TT_SPECIAL_NEG_INF == 1
constexpr uint16_t kSharedInverseSquareNegInfResultBf16 = 0x7f80u;
#elif TT_SPECIAL_NEG_INF == 2
constexpr uint16_t kSharedInverseSquareNegInfResultBf16 = 0xff80u;
#else
#error "shared inverse-square TTI suffix cannot encode declared -Inf terminal"
#endif
constexpr bool kSharedInverseSquareTtiReplay =
    TT_SHARED_INVERSE_SQUARE_BODY_SLOTS == kSharedInverseSquareBodySlots &&
    tti_replay_fits(kSharedInverseSquareBodySlots) &&
    (__builtin_bit_cast(uint32_t, SHARED_INVERSE_SQUARE_P0) & 0xffffu) == 0u &&
    (__builtin_bit_cast(uint32_t, SHARED_INVERSE_SQUARE_P2[0]) & 0xffffu) == 0u &&
    (__builtin_bit_cast(uint32_t, SHARED_INVERSE_SQUARE_P2[1]) & 0xffffu) == 0u &&
    (__builtin_bit_cast(uint32_t, SHARED_INVERSE_SQUARE_P2[2]) & 0xffffu) == 0u;

inline void shared_inverse_square_recurrence_p0_p2_1_tti_replay_tile() {
    static_assert(kSharedInverseSquareTtiReplay);
    constexpr uint16_t kRoundBiasBf16 = 0x4b40u;  // 1.5*2^23
    constexpr uint16_t kHalfBf16 = 0x3f00u;
    constexpr uint16_t kP0Bf16 =
        __builtin_bit_cast(uint32_t, SHARED_INVERSE_SQUARE_P0) >> 16;
    constexpr uint16_t kP2TopBf16 =
        __builtin_bit_cast(uint32_t, SHARED_INVERSE_SQUARE_P2[2]) >> 16;

    // Cross-replay pin.  Every body leaves L7 untouched.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_FLOATB,
        kHalfBf16);

    TT_REPLAY_RECORD(kSharedInverseSquareBodySlots);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG0,
        0,
        TT_ADDRMOD_DST_HOLD,
        0);  // 1: raw decoded x
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        1);  // 2: |x|
    TTI_SFPMOV(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG3,
        0);  // 3: preserve x sign carrier
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG3,
        0);  // 4: signed_one=copysign(1,x)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_FLOATB,
        kRoundBiasBf16);  // 5: round bias
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        0);  // 6: x+bias
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        0);  // 7: z=1+|x|; also round-MAD gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        2);  // 8: n=(x+bias)-bias
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        1);  // 9: negative_mask=.5-.5*signed_one
    TTI_SFPARECIP(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        0);  // 10: reciprocal(z) seed; also mask-MAD gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        1);  // 11: q=x-negative_mask*n
    TTI_SFPNOP;  // 12: q producer / reciprocal seed gap
    TTI_SFPARECIP(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        0);  // 13: reciprocal(q) seed
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        2);  // 14: z*ru-2; also q-seed gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG0,
        2);  // 15: q*riq-2; also u-residual gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        3);  // 16: refined u=ru*(-residual)
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG0,
        0,
        0);  // 17: q residual < 0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG6,
        3);  // 18: refined iq; predicate preserves q==0 +Inf seed
    TTI_SFPENCC(3, 0, 0, 10);  // 19
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG1,
        sfpi::SFPLOADI_MOD0_FLOATB,
        kP0Bf16);  // 20: a0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG1,
        0);  // 21: core1=a0*u+.5
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // 22: rq=negative_mask*q; core gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        0);  // 23: core2=core1*u+1
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // 24: q2=rq*rq; core gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // 25: core=core2*u
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG6,
        0);  // 26: iq2
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_FLOATB,
        kP2TopBf16);  // 27: b2
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG0,
        0);  // 28: reg1=b2*q2+b1
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG1,
        0);  // 29: signed core + iq2; regularizer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG0,
        0);  // 30: regularizer=reg1*q2+b0
    TTI_SFPNOP;  // 31: final dual-MAD gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        0);  // 32: result=mask*regularizer+signed_core+iq2

    auto same_row_suffix = []() {
        // Round the finite replay result first.  Blackhole's FP32-to-BF16
        // stochastic-round instruction canonicalizes every NaN payload to
        // infinity, so typed raw-class replacements must happen after it.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG2,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // 1
#if defined(TT_SHARED_INVERSE_SQUARE_TARGET_CLASS_REPAIR)
        // The selected form's typed terminal proves result=+Inf exactly on
        // finite nonpositive integers. Recompute that anonymous predicate
        // from the source, then use the target graph's abstractly derived
        // first-finite boundary. Finite-reference lanes are untouched.
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG0,
            0,
            TT_ADDRMOD_DST_HOLD,
            0);  // decoded source x
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            sfpi::SFPLOADI_MOD0_FLOATB,
            0x4b40);  // 1.5*2^23 BF16 rounding bias
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG4,
            0);  // x+bias
        TTI_SFPNOP;
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG4,
            2);  // rint(x)=(x+bias)-bias
        TTI_SFPNOP;
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG4,
            0);  // x-rint(x)
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG5,
            sfpi::SFPLOADI_MOD0_FLOATB,
            kSharedInverseSquareFiniteThresholdBiasBf16);
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG5,
            0);  // x - graph-derived midpoint
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG4,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, 0);  // below midpoint
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_SHARED_INVERSE_SQUARE_FINITE_REPRESENTATIVE);
        TTI_SFPENCC(0, 0, 0, 0);
        // At the graph-derived final transition every remaining negative BF16
        // finite value is integral. A strict comparison with the immediately
        // preceding representable magnitude therefore owns the complete
        // target +zero class without a numeric lookup table.
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG5,
            sfpi::SFPLOADI_MOD0_FLOATB,
            kSharedInverseSquareZeroTransitionBiasBf16);
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG5,
            0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, 0);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_SHARED_INVERSE_SQUARE_ZERO_REPRESENTATIVE);
        TTI_SFPENCC(0, 0, 0, 0);
#endif
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG0,
            sfpi::SFPLOAD_MOD0_FMT_UINT16,
            TT_ADDRMOD_DST_HOLD,
            0);  // 2: physical raw BF16
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            sfpi::SFPLOADI_MOD0_USHORT,
            0x00ff);  // 3: exponent-FF value and equality mask
        TTI_SFPXOR(
            0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG0,
            0);  // 4: raw ^ 0x00ff, written to L0
        TTI_SFPAND(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            1);  // 5: exponent==0xff iff result is zero
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG1,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 6
        // The broad exponent-FF quotient canonicalizes first. The exact raw
        // -Inf terminal is restored below from the compiler's typed special
        // policy, rather than baking one activation/profile result here.
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_FLOATB,
            0x0000);  // 7: every NaN and -Inf -> +0
        TTI_SFPENCC(0, 0, 0, 0);  // 8
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG0,
            sfpi::SFPLOAD_MOD0_FMT_UINT16,
            TT_ADDRMOD_DST_HOLD,
            0);  // 9: physical raw BF16 for exact overrides
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            sfpi::SFPLOADI_MOD0_USHORT,
            0x80ff);  // 10: exact -Inf physical word
        TTI_SFPXOR(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            0);  // 11: +Inf delta; XOR destination is its third operand
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG1,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 12
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_FLOATB,
            kSharedInverseSquareNegInfResultBf16);  // 13: typed exact -Inf result
        TTI_SFPENCC(0, 0, 0, 0);  // 14
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            sfpi::SFPLOADI_MOD0_USHORT,
            0x00fd);  // 15: +2^126 physical word
        TTI_SFPXOR(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            0);  // 16: boundary delta
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG1,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 17
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_FLOATB,
#if defined(TT_SHARED_INVERSE_SQUARE_TARGET_CLASS_REPAIR)
#if defined(TT_SHARED_INVERSE_SQUARE_FINITE_REFERENCE_PRECEDENCE)
            0x0080);  // finite-reference lane retains selected MIN_NORMAL
#else
            TT_SHARED_INVERSE_SQUARE_ZERO_REPRESENTATIVE);  // target +zero
#endif
#else
            0x0080);  // 18: MIN_NORMAL
#endif
        TTI_SFPENCC(0, 0, 0, 0);  // 19
        TTI_SFPSTORE(
            ckernel::p_sfpu::LREG2,
            0,
            TT_ADDRMOD_DST_ADV,
            0);  // 20
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (int d = 1; d < 32; d++) {
        TTI_REPLAY(0, kSharedInverseSquareBodySlots, 0, 0);
        same_row_suffix();
    }
    TT_REPLAY_TILE_EPILOGUE();
}

#endif
#endif

#if defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1)
#if defined(TT_TARGET_SELECTED_GUARDED_CLASS_TERMINAL)
static_assert(
    TT_TARGET_NEGATIVE_INTEGER_PARTITION_COUNT >= 1u &&
        TT_TARGET_NEGATIVE_INTEGER_PARTITION_COUNT <= 4u,
    "guarded target integer repair exceeds its compiler-owned partition budget");
inline void apply_target_selected_guarded_class_terminal(
    vFloat input, vFloat& result) {
    vUInt bits = as<vUInt>(input);
    vUInt sign = bits >> 31;
    vUInt magnitude = bits & 0x7fffffffu;
    vUInt result_bits = as<vUInt>(result);
    // The selected exponent-bucket certificate proves that a +Inf egress on a
    // negative finite input is its exact-negative-integer terminal.  Interpret
    // only the target graph's class transitions on that typed subset.
    v_if((sign != 0u) && (result_bits == 0x7f800000u) &&
         (bits < (TT_TARGET_NEGATIVE_INTEGER_NEG_INF_FIRST << 16))) {
        result = __builtin_bit_cast(
            float, TT_TARGET_NEGATIVE_INTEGER_OUTPUT_BF16[0] << 16);
    }
    v_endif;
    // Above the second transition every BF16 step is already an exact integer,
    // so the graph-derived raw interval is itself the typed pole predicate.
    v_if((bits >= (TT_TARGET_NEGATIVE_INTEGER_NEG_INF_FIRST << 16)) &&
         (bits < (TT_TARGET_NEGATIVE_INTEGER_POS_INF_FIRST << 16))) {
        result = __builtin_bit_cast(
            float, TT_TARGET_NEGATIVE_INTEGER_OUTPUT_BF16[1] << 16);
    }
    v_endif;
    // Later, broader type-class repairs dominate the negative-terminal arm.
    // This ordering removes redundant exponent/mantissa predicates while
    // preserving the exact class quotient.
    v_if(magnitude < 0x00800000u) {
        result = __builtin_bit_cast(
            float, TT_TARGET_ZERO_SUBNORMAL_REPRESENTATIVE << 16);
    }
    v_endif;
    v_if((sign == 0u) && (magnitude > 0x7f800000u)) {
        result = __builtin_bit_cast(
            float, TT_TARGET_NAN_REPRESENTATIVE << 16);
    }
    v_endif;
}
#endif
#if defined(TT_TARGET_SELECTED_TTI_CLASS_REPAIR)
static_assert(
    TT_TARGET_CLASS_TTI_RAW_SHADOW_ROWS == 32u &&
        TT_TARGET_CLASS_TTI_RAW_SHADOW_ROW_BASE +
                TT_TARGET_CLASS_TTI_RAW_SHADOW_ROWS <=
            TT_TARGET_CLASS_TTI_DST_CAPACITY,
    "typed target-class raw shadow exceeds the exponent-bucket DST plan");
static_assert(
    TT_TARGET_CLASS_TTI_BODY_SLOTS == 31u &&
        TT_TARGET_CLASS_TTI_BODY_SLOTS <= 32u &&
        TT_TARGET_CLASS_TTI_PEAK_LIVE <= 8u,
    "typed target-class replay exceeds SFPU issue or register capacity");

inline void exponent_bucket_target_class_repair_tti_tile() {
    // L6/L7 are graph-derived BF16 transition boundaries.  They are pinned
    // once outside the replay and are disjoint from the selected evaluator,
    // whose temporaries have all retired before this phase starts.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG6,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_CLASS_TTI_NEG_INF_FIRST);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_CLASS_TTI_POS_INF_FIRST);
    TT_REPLAY_RECORD(TT_TARGET_CLASS_TTI_BODY_SLOTS);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 64);  // selected BF16
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG1,
        0,
        TT_ADDRMOD_DST_HOLD,
        2 * TT_TARGET_CLASS_TTI_RAW_SHADOW_ROW_BASE);  // decoded raw input
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_UPPER,
        TT_TARGET_CLASS_TTI_POS_INF_RAW >> 16);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_LOWER,
        TT_TARGET_CLASS_TTI_POS_INF_RAW & 0xffffu);
    TTI_SFPMOV(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG3,
        0);
    TTI_SFPXOR(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        0);  // selected +Inf delta

    // A selected +Inf on a negative input above the first target transition
    // is an exact negative-integer pole whose target class is finite.
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG3,
        0,
        sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);  // raw input < 0
    TTI_SFPGT(
        0,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG1,
        1);  // raw input > first transition
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_CLASS_TTI_FINITE_INTEGER);
    TTI_SFPENCC(0, 0, 0, 0);

    // The closed interval [first -Inf transition, next +Inf transition) is
    // target -Inf.  BF16 spacing proves every value there is an integer, so
    // no observed-value membership table is needed.
    TTI_SFPLE(
        0,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG1,
        1);  // raw input <= first transition
    TTI_SFPGT(
        0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG1,
        1);  // raw input > next transition
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_CLASS_TTI_NEG_INF);
    TTI_SFPENCC(0, 0, 0, 0);

    // Raw exponent zero (both signs) has a finite target class.
    TTI_SFPAND(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        1);
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG3,
        0,
        sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_CLASS_TTI_ZERO_SUBNORMAL);
    TTI_SFPENCC(0, 0, 0, 0);

    // Positive NaN is sign-zero, exponent-FF, and nonzero mantissa in the
    // physical U16 view.  Negative NaNs already have the target class at the
    // selected handoff, so the typed class-disagreement relation preserves
    // them instead of applying the positive-NaN representative.
    // The finite representative comes from abstract evaluation of the target
    // graph; payload/sign equality is intentionally outside C-SEL-1.
    TTI_SFPMOV(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        0);
    TTI_SFPXOR(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        0);
    TTI_SFPAND(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        1);  // exponent delta
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOADI_MOD0_UPPER,
        TT_TARGET_CLASS_TTI_MAN_MASK >> 16);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOADI_MOD0_LOWER,
        TT_TARGET_CLASS_TTI_MAN_MASK & 0xffffu);
    TTI_SFPAND(
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        1);  // mantissa bits
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG3,
        0,
        sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG5,
        0,
        sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG1,
        0,
        sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_CLASS_TTI_NAN);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);
    TT_REPLAY_REST(TT_TARGET_CLASS_TTI_BODY_SLOTS, 31);
    TT_REPLAY_TILE_EPILOGUE();
}
#endif
// One shared P3 over the normalized mantissa plus an unbiased-exponent term and
// an inverse-z P2 correction.  This selector-free layout is the target-lattice
// factorization of the former coefficient buckets.  Recurrence and reflection
// then share one explicitly modeled reciprocal lane.  The form is anonymous: all
// numerical payload comes from its ordinary frontier leaves.
inline vFloat exponent_bucket_log_derivative_1_eval(
    vFloat x, uint32_t handoff_row) {
    static_assert(TT_EXPONENT_BUCKET_COUNT == 0u);
    static_assert(TT_EXPONENT_BUCKET_NATIVE_LOG == 0u);
    static_assert(TT_EXPONENT_BUCKET_RECIPROCAL_DEGREE == 2u);
    static_assert(TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS == 1u ||
                  TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS == 2u);
    static_assert(TT_EXPONENT_BUCKET_REFLECTION_DEGREE == 8u);

    // The selector-free compact form has room to keep inverse_z live through
    // its P3 core.  Keeping this phase in LREGs avoids the first destination
    // handoff and the duplicate abs(x)+1 coordinate reconstruction.  The core
    // still closes once before the independent reflection suffix.
    {
        vFloat z = setsgn(x, 0) + 1.0f;
        vFloat inverse_z = typed_finite_coordinate_reciprocal_iter1(z);
        vInt exponent = exexp(z, ExponentMode::Biased) - 127;
        vFloat mantissa = setexp(z, 127);
        vFloat core = TT_EXPONENT_BUCKET_COEFFS[3] * mantissa +
                      TT_EXPONENT_BUCKET_COEFFS[2];
        core = core * mantissa + TT_EXPONENT_BUCKET_COEFFS[1];
        core = core * mantissa + TT_EXPONENT_BUCKET_COEFFS[0];
        vFloat exponent_value = convert<vFloat>(as<vSMag>(exponent), RoundMode::Nearest);
        core = exponent_value * TT_EXPONENT_BUCKET_COEFFS[4] + core;
        vFloat inverse_correction =
            TT_EXPONENT_BUCKET_COEFFS[6] * inverse_z +
            TT_EXPONENT_BUCKET_COEFFS[5];
        core = inverse_correction * inverse_z + core;
        dst_reg[handoff_row].mode<DataLayout::F32>() = core;
    }

    // Every finite BF16 magnitude >=128 is integral, so the add/sub identity
    // already returns a zero fraction there.  Reflection consumes the fraction
    // only for negative nonintegers; clamping the inactive lanes is redundant.
    constexpr float kRound = 0x1.8p23f;
    vFloat nearest = x + kRound;
    nearest = nearest - kRound;
    vFloat fraction = x - nearest;
    // Positive recurrence uses 1/x; negative reflection uses 1/fraction.
    // Selecting the divisor first lets both mutually-exclusive consumers own
    // one physical reciprocal instead of issuing two full Newton chains.  The
    // declared zero and negative-integer terminals dominate divisor==0 after
    // this body, so they do not need a second arithmetic-only safety branch.
    vFloat divisor = fraction;
    v_if(x > 0.0f) { divisor = x; }
    v_endif;
    vFloat inverse = ckernel::sfpu::sfpu_reciprocal_iter<
        TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS>(divisor);

    vFloat fraction_square = fraction * fraction;
    vFloat scaled_cot = TT_EXPONENT_BUCKET_REFLECTION_COEFFS[8];
    scaled_cot = scaled_cot * fraction_square +
                 TT_EXPONENT_BUCKET_REFLECTION_COEFFS[6];
    scaled_cot = scaled_cot * fraction_square +
                 TT_EXPONENT_BUCKET_REFLECTION_COEFFS[4];
    scaled_cot = scaled_cot * fraction_square +
                 TT_EXPONENT_BUCKET_REFLECTION_COEFFS[2];
    scaled_cot = scaled_cot * fraction_square +
                 TT_EXPONENT_BUCKET_REFLECTION_COEFFS[0];

    vFloat core = dst_reg[handoff_row].mode<DataLayout::F32>();
    // Select the mutually-exclusive correction scale, then issue one fused
    // correction for every lane.  This avoids issuing separate positive and
    // negative arithmetic under predicates; scale=1 preserves recurrence.
    vFloat correction_scale = 1.0f;
    v_if(x < 0.0f) { correction_scale = scaled_cot; }
    v_endif;
    vFloat result = core - correction_scale * inverse;
    // The common finalizer maps this generic pole marker to the form-declared
    // terminal class (+Inf).  A raw reciprocal-by-zero result is -Inf instead
    // and therefore cannot carry the declared negative-integer distinction.
    v_if((x < 0.0f) && (fraction == 0.0f)) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;
    return result;
}

#if defined(TT_TARGET_SELECTED_BF16_SCRATCH_BOUNDARY)
inline vFloat target_declared_log(vFloat base) {
    vFloat x = setexp(base, 127);
    vFloat series = x * TT_TARGET_LOG_COEFFS[0] + TT_TARGET_LOG_COEFFS[1];
    series = x * series + TT_TARGET_LOG_COEFFS[2];
    series = x * series + TT_TARGET_LOG_COEFFS[3];
    auto exponent = convert<vSMag>(exexp(base));
    vFloat exponent_value = convert<vFloat>(exponent, RoundMode::Nearest);
    vFloat result = exponent_value * TT_TARGET_LOG_LN2 + series;
    v_if(base == 0.0f) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
    return result;
}

inline vFloat target_native_rational_asymptotic_eval(vFloat x) {
    vFloat numerator = TT_TARGET_CORE_NUM[6];
    vFloat denominator = TT_TARGET_CORE_DEN[5];
    numerator = numerator * x + TT_TARGET_CORE_NUM[5];
    numerator = numerator * x + TT_TARGET_CORE_NUM[4];
    denominator = denominator * x + TT_TARGET_CORE_DEN[4];
    numerator = numerator * x + TT_TARGET_CORE_NUM[3];
    denominator = denominator * x + TT_TARGET_CORE_DEN[3];
    numerator = numerator * x + TT_TARGET_CORE_NUM[2];
    denominator = denominator * x + TT_TARGET_CORE_DEN[2];
    numerator = numerator * x + TT_TARGET_CORE_NUM[1];
    denominator = denominator * x + TT_TARGET_CORE_DEN[1];
    numerator = numerator * x + TT_TARGET_CORE_NUM[0];
    denominator = denominator * x + TT_TARGET_CORE_DEN[0];
    vFloat result = numerator * ckernel::sfpu::sfpu_reciprocal_iter<
        TT_TARGET_CORE_RECIP_ITERS>(denominator);
    v_if(x > TT_TARGET_TAIL_CROSSOVER) {
        vFloat inverse = ckernel::sfpu::sfpu_reciprocal_iter<
            TT_TARGET_TAIL_RECIP_ITERS>(x);
        vFloat inverse_square = inverse * inverse;
        vFloat bernoulli = TT_TARGET_TAIL_B2 - inverse_square * TT_TARGET_TAIL_B4;
        result = target_declared_log(x) -
                 inverse * TT_TARGET_TAIL_HALF - inverse_square * bernoulli;
        v_if((exexp(x, ExponentMode::Biased) == 255) && (exman(x) == 0)) {
            result = std::numeric_limits<float>::infinity();
        }
        v_endif;
    }
    v_endif;
    return result;
}

inline vFloat target_result_class(vFloat value) {
    vFloat code = 0.0f;
    vFloat magnitude = setsgn(value, 0);
    vInt exponent = exexp(magnitude, ExponentMode::Biased);
    vUInt sign = as<vUInt>(value) >> 31;
    v_if(magnitude == 0.0f) {
        code = 1.0f;
        v_if(sign != 0) { code = 2.0f; }
        v_endif;
    }
    v_endif;
    v_if(exponent == 255) {
        code = 5.0f;
        v_if(exman(magnitude) == 0) {
            code = 3.0f;
            v_if(sign != 0) { code = 4.0f; }
            v_endif;
        }
        v_endif;
    }
    v_endif;
    return code;
}
#endif

#endif
#if defined(TT_FACTORED_CW_REPLAY)
template <uint32_t DEGREE, uint32_t SEGMENTS, uint32_t SIZE,
          uint32_t C2_BITS, uint32_t C3_BITS, uint32_t C4_BITS,
          uint32_t BODY_SLOTS, uint32_t SUFFIX_SLOTS, uint32_t ROWS>
inline void integer_exponent_residual_replay_tile() {
    static_assert(DEGREE == TT_FACTORED_CW_REPLAY_DEGREE &&
                      SEGMENTS == TT_FACTORED_CW_REPLAY_SEGMENTS &&
                      SIZE == TT_FACTORED_CW_REPLAY_LUT_SIZE,
                  "verified replay descriptor/LUT resource mismatch");
    static_assert(BODY_SLOTS == TT_FACTORED_CW_REPLAY_BODY_SLOTS &&
                      SUFFIX_SLOTS == TT_FACTORED_CW_REPLAY_SUFFIX_SLOTS &&
                      ROWS == TT_FACTORED_CW_REPLAY_ROWS &&
                      tti_replay_fits(BODY_SLOTS),
                  "verified replay schedule exceeds certified resources");
    constexpr uint32_t kRoundingBias =
        TT_FACTORED_CW_REPLAY_ROUNDING_BIAS_BITS;
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, C4_BITS);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG5, C3_BITS);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG6, C2_BITS);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, kRoundingBias);

    TT_REPLAY_RECORD(BODY_SLOTS);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB,
                 TT_FACTORED_CW_REPLAY_LOWER_BF16);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 9);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB,
                 TT_FACTORED_CW_REPLAY_UPPER_BF16);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 1);
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12,
               ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP;
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPADD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG7,
               ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    // Keep L7's rounding bias live across all 31 replays.  The selected leaf's
    // exact c1=0.5 is already preloaded in L14, so it is also the scale/bias
    // bit addend.  Overwriting L7 here made only recorded row zero correct;
    // every replay then rounded with 0.5 instead of 12582912.
    // The shift reads the preserved integer copy in L3, not the MAD-family
    // result in L1.  It is therefore the required independent issue between
    // the L1 producer and its later residual-FMA consumer; no NOP belongs here.
    TTI_SFPSHFT(0x017, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 7);
    TTI_SFPIADD(
        0,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG3,
        sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG13,
               ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0,
               ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0,
               ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
               ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
               ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1,
               ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    // The reduced result is already the k==0 default output in L0.  The base
    // calculation below is independent and supplies its MAD hazard gap; the
    // reconstruction consumes L0 directly, so no L2->L0 copy is required.
    TTI_SFPADD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG14,
               ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG0,
               ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0,
                 sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2,
               ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        TT_ADDRMOD_DST_HOLD,
        0);  // raw BH DST word; fills the final MAD hazard slot
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_USHORT,
        TT_FACTORED_CW_REPLAY_RAW_EQUAL_VALUE);
    TTI_SFPXOR(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        0);
    TTI_SFPAND(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        1);

    auto same_row_suffix = []() {
        static_assert(
            TT_FACTORED_CW_REPLAY_RAW_EQUAL_MASK ==
                    TT_FACTORED_CW_REPLAY_RAW_EQUAL_VALUE &&
                (TT_FACTORED_CW_REPLAY_RAW_EQUAL_MASK &
                 TT_FACTORED_CW_REPLAY_RAW_NONZERO_MASK) == 0u &&
                (TT_FACTORED_CW_REPLAY_RAW_EQUAL_MASK |
                 TT_FACTORED_CW_REPLAY_RAW_NONZERO_MASK) == 0xffffu &&
                TT_FACTORED_CW_REPLAY_RAW_PATTERN_COUNT > 0u,
            "verified replay raw predicate descriptor is inconsistent");
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG2,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG3,
            0,
            sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPSETEXP(
            TT_FACTORED_CW_REPLAY_POSITIVE_INFINITY_EXPONENT,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            1);  // exact +Inf override value
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(
            ckernel::p_sfpu::LREG0,
            0,
            TT_ADDRMOD_DST_ADV,
            0);
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (uint32_t row = 1; row < ROWS; ++row) {
        TTI_REPLAY(0, BODY_SLOTS, 0, 0);
        same_row_suffix();
    }
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(EXPONENT_ALU_EXP2)

#if defined(EVAL_METHOD_ABS_SQUARE_DECAY_1)
inline vFloat typed_finite_coordinate_reciprocal_iter1(vFloat x);
// Closed anonymous whole-core form:
//   a = min(abs(x), bound); y = exp(-a*a) * N(a)/D(a)
// followed by the reflection f(-a)=2-f(a).  The two coefficient arrays are
// ordinary fitter artifacts joined by activation JSON; the common element
// loop, input transport, target finalizer, and store path remain unchanged.
template <uint32_t NUM_DEG, uint32_t DEN_DEG>
inline vFloat abs_square_decay_1_eval(vFloat x) {
    vFloat coordinate = setsgn(x, 0);
#if defined(SQUARE_DECAY_COORDINATE_TERMINAL)
    vFloat coordinate_bound = SQUARE_DECAY_COORDINATE_TERMINAL_BOUND;
    ordered_min_max(coordinate, coordinate_bound);
#endif
    vFloat negative_square = -(coordinate * coordinate);
    vFloat decay = exp_hw_eval<EXP_HW_DEGREE>(negative_square);
#if defined(SQUARE_DECAY_CORRECTION_POLY)
    vFloat quotient = SQUARE_DECAY_NUM_COEFFS[NUM_DEG];
#pragma GCC unroll 8
    for (int k = (int)NUM_DEG - 1; k >= 0; k--) {
        quotient = quotient * coordinate + SQUARE_DECAY_NUM_COEFFS[k];
    }
#else
    vFloat num = SQUARE_DECAY_NUM_COEFFS[NUM_DEG];
    vFloat den = SQUARE_DECAY_DEN_COEFFS[DEN_DEG];
#pragma GCC unroll 8
    for (int k = (int)NUM_DEG - 1; k >= 0; k--) {
        num = num * coordinate + SQUARE_DECAY_NUM_COEFFS[k];
    }
#pragma GCC unroll 8
    for (int k = (int)DEN_DEG - 1; k >= 0; k--) {
        den = den * coordinate + SQUARE_DECAY_DEN_COEFFS[k];
    }
#if defined(RATIONAL_RECIPROCAL_ZERO_ITER)
    vFloat quotient = num * ckernel::sfpu::sfpu_reciprocal_iter<0>(den);
#elif defined(RATIONAL_RECIPROCAL_ONE_ITER)
#if defined(TT_SQUARE_DECAY_TYPED_FINITE_RECIPROCAL)
    vFloat quotient = num * typed_finite_coordinate_reciprocal_iter1(den);
#else
    vFloat quotient = num * ckernel::sfpu::sfpu_reciprocal_iter<1>(den);
#endif
#else
    vFloat quotient = num * ckernel::sfpu::sfpu_reciprocal_iter<2>(den);
#endif
#endif
    vFloat result = decay * quotient;
#if defined(SQUARE_DECAY_TWO_MINUS_ON_NEGATIVE)
    v_if(x < 0.0f) { result = 2.0f - result; }
    v_endif;
#else
#error "abs_square_decay_1 reconstruction is missing"
#endif
    return result;
}
#endif

#endif  // defined(EXPONENT_ALU_EXP2)

#if defined(EVAL_METHOD_ZONE_FACTORED_AFFINE_EVEN_DECAY_2)
// Closed anonymous three-leaf form.  The body evaluates one bounded P2 exp2
// leaf, one P5 affine-even core, and one P3 negative decay correction.  The
// decay correction is multiplied before exponent scaling so a normal final
// result is never lost through a standalone subnormal exp intermediate.
constexpr float zone_exp_fraction_scale(int power) {
    float scale = 1.0f;
    for (int index = 0; index < power; ++index) {
        scale *= 0x1p-23f;
    }
    return scale;
}

// Zone constant-pool preload (programmed ONCE in kernel_main; see the HW_PRELOAD
// precedent there): Prgm0 carries the biased-exponent multiplier and Prgm1/2 the
// two upper pre-scaled exponent-leaf coefficients, so the per-row body issues no
// SFPLOADI pair for any of the three.  The programmable constant registers sit
// outside the eight-LREG budget, so the body's peak live set is unchanged.
// TT_ZONE_PRGM_PRELOAD_DISABLE (via TT_ACT_KERNEL_EXTRA_DEFINES) is the A/B
// escape arm: the identical arithmetic with in-body literal materialization.
constexpr float kZoneNegativeHalfLog2E = -0.7213475204444817f;

inline vFloat zone_factored_affine_even_decay_2_eval(vFloat x) {
    static_assert(
        TT_ZONE_EXP_DEGREE == 2u && TT_ZONE_CORE_DEGREE == 5u &&
            TT_ZONE_DECAY_DEGREE == 3u && TT_ZONE_BODY_SLOTS == 25u &&
            TT_ZONE_BODY_SLOTS <= TT_ZONE_RECORD_CAPACITY &&
            TT_ZONE_PEAK_LIVE <= 8u,
        "zone-factored descriptor exceeds compact structural resources");

    // The terminal and identity predicates below own every lane outside the
    // fitted core/decay coordinates.  Evaluate the raw coordinate speculatively
    // and overwrite those lanes, matching the established SFPU tail topology;
    // clamping here only duplicated the already-total route ownership.
    vFloat coordinate = x;
    vFloat square = coordinate * coordinate;

    // Moroz-style biased exponent decomposition matching the fitter model.
#if !defined(TT_ZONE_PRGM_PRELOAD_DISABLE)
    // Same SFPMAD; the multiplier operand reads the preloaded constant
    // register instead of a per-row two-issue SFPLOADI materialization.
    vFloat xlog2 = square * vConstFloatPrgm0 + 127.0f;
#else
    vFloat xlog2 = square * kZoneNegativeHalfLog2E + 127.0f;
#endif
    vInt exponent = exexp(xlog2);
    vInt mantissa = exman(xlog2, MantissaMode::ImplicitOne);
    mantissa = shft(mantissa, exponent, ShiftMode::Logical);
    vFloat encoded = as<vFloat>(mantissa);
    vInt integer_part = exexp(encoded, ExponentMode::Biased);
    vMag fractional_magnitude = exman(encoded);
    vFloat fraction = convert<vFloat>(fractional_magnitude, RoundMode::Nearest);

    // Fold the exact exman 2^-23 normalization into each semantic leaf
    // coefficient.  The CSV remains in its ordinary [0,1) basis while the
    // physical Horner consumes the raw integer magnitude and saves one SFPMUL.
#if !defined(TT_ZONE_PRGM_PRELOAD_DISABLE)
    // The P2 Horner over the preloaded upper coefficients: the head fuses the
    // init and first step into the one SFPMAD they always were
    // (c2'*fraction + c1'), with both constant operands read from Prgm2/Prgm1.
    // The FMA sequence, operand values, and rounding are identical to the
    // escape arm below; only the operand sourcing changes.
    vFloat exp_polynomial = vConstFloatPrgm2 * fraction + vConstFloatPrgm1;
    exp_polynomial = exp_polynomial * fraction +
                     TT_ZONE_EXP_COEFFS[0] * zone_exp_fraction_scale(0);
#else
    vFloat exp_polynomial = TT_ZONE_EXP_COEFFS[TT_ZONE_EXP_DEGREE] *
                            zone_exp_fraction_scale(TT_ZONE_EXP_DEGREE);
#pragma GCC unroll 4
    for (int k = (int)TT_ZONE_EXP_DEGREE - 1; k >= 0; --k) {
        exp_polynomial = exp_polynomial * fraction +
                         TT_ZONE_EXP_COEFFS[k] * zone_exp_fraction_scale(k);
    }
#endif
    vFloat decay_polynomial = TT_ZONE_DECAY_COEFFS[TT_ZONE_DECAY_DEGREE];
#pragma GCC unroll 4
    for (int k = (int)TT_ZONE_DECAY_DEGREE - 1; k >= 0; --k) {
        decay_polynomial =
            decay_polynomial * coordinate + TT_ZONE_DECAY_COEFFS[k];
    }
    vFloat unscaled = exp_polynomial * decay_polynomial;
    // Synthesize the exact 2^(integer_part-127) factor from its biased exponent
    // bits.  Multiplication preserves unscaled's own exponent automatically and
    // lets BH FTZ the final product, replacing the five-op corrected-setexp
    // sequence with the established SHFT+SFPMUL pair.
    vFloat result = unscaled * as<vFloat>(integer_part << 23);

    v_if(x > TT_ZONE_DECAY_CORE_BOUNDARY) {
        vFloat core_polynomial = TT_ZONE_CORE_COEFFS[TT_ZONE_CORE_DEGREE];
#pragma GCC unroll 8
        for (int k = (int)TT_ZONE_CORE_DEGREE - 1; k >= 0; --k) {
            core_polynomial =
                core_polynomial * square + TT_ZONE_CORE_COEFFS[k];
        }
        vFloat cdf = coordinate * core_polynomial + TT_ZONE_ORIGIN_SLOPE;
        result = coordinate * cdf;
    }
    v_endif;

    v_if(x > TT_ZONE_POSITIVE_IDENTITY) { result = x; }
    v_endif;
    v_if(x <= TT_ZONE_NEGATIVE_TERMINAL) { result = 0.0f; }
    v_endif;
    return result;
}

#if defined(TT_ZONE_ORIGIN_HALF_TARGET_REPAIR)
inline void apply_zone_factored_origin_half_target_repair(
    vUInt raw_u16, vFloat x, vFloat& result) {
    static_assert(
        TT_ZONE_ORIGIN_SLOPE == 0.5f &&
            TT_ZONE_ORIGIN_HALF_EQUAL_MASK_U16 == 0x7fffu &&
            TT_ZONE_ORIGIN_HALF_EQUAL_VALUE_U16 == 0x7f01u &&
            TT_ZONE_ORIGIN_HALF_OUTPUT_MANTISSA == 0u &&
            TT_ZONE_ORIGIN_HALF_CORRECTED_PATTERNS == 2u,
        "zone-factored origin-half target repair lost its typed proof");
    // Blackhole U16 DST layout is sign15|mantissa14:8|exponent7:0.  Masking
    // only sign therefore recognizes exactly IEEE BF16 0x00ff and 0x80ff.
    // SETMAN(0) keeps the decoded input sign/exponent and constructs signed
    // FP32 MIN_NORMAL before the ordinary BF16 RNE/store.  This is the target
    // rounding of the declared half-origin reconstruction, not a witness table.
#if defined(TT_ZONE_ORIGIN_HALF_FOLDED_PREDICATE)
    vUInt delta =
        (raw_u16 ^ vUInt(TT_ZONE_ORIGIN_HALF_EQUAL_VALUE_U16)) &
        vUInt(TT_ZONE_ORIGIN_HALF_EQUAL_MASK_U16);
#else
    // C-VER escape arm: two literal physical-word deltas implementing the
    // same typed target result.  It is intentionally less compact but remains
    // output-byte-identical to the folded sign-agnostic predicate.
    constexpr uint16_t positive_word = TT_ZONE_ORIGIN_HALF_EQUAL_VALUE_U16;
    constexpr uint16_t negative_word =
        TT_ZONE_ORIGIN_HALF_EQUAL_VALUE_U16 |
        (uint16_t)(~TT_ZONE_ORIGIN_HALF_EQUAL_MASK_U16);
    vUInt positive_delta = raw_u16 ^ vUInt(positive_word);
    vUInt negative_delta = raw_u16 ^ vUInt(negative_word);
    vUInt delta = positive_delta & negative_delta;
#endif
    v_if(delta == 0u) {
        result = setman(x, TT_ZONE_ORIGIN_HALF_OUTPUT_MANTISSA);
    }
    v_endif;
}
#endif
#endif

#if defined(EXPONENT_ALU_EXP2)

#if defined(EVAL_METHOD_ABS_RESIDUAL_AFFINE_1) &&                 \
    defined(ABS_RESIDUAL_CORRECTION_POLY) &&                     \
    (defined(ABS_RESIDUAL_POSITIVE_PART_PLUS) ||                 \
     defined(ABS_RESIDUAL_NEGATIVE_PART_MINUS)) &&               \
    defined(EXP_HW_FUSED) && defined(EXP_HW_BARE_SETEXP) &&      \
    defined(EXP_HW_SKIP_INPUT_CLAMP) && defined(USE_BF16) &&     \
    defined(ARCH_BLACKHOLE) && !defined(ABS_RESIDUAL_TTI_DISABLE) && \
    defined(TT_ABS_RESIDUAL_TYPED_TTI_TERMINAL)
#define ABS_RESIDUAL_TTI_CANDIDATE 1
// A unit correction origin needs one hazard-filling NOP.  A non-unit origin
// is materialized by the two half-immediate loads below, so its recorded body
// owns one additional issue slot.  Keep the replay resource count derived
// from the typed coefficient shape rather than assuming a unit-origin shape.
constexpr uint32_t kAbsResidualTtiBodySlots =
    27u + ((ABS_RESIDUAL_NUM_COEFFS[0] == 1.0f) ? 0u : 1u);
constexpr bool kAbsResidualTtiReplay =
    EXP_HW_DEGREE == 2u && ABS_RESIDUAL_NUM_DEGREE == 3u &&
    ABS_RESIDUAL_DEN_DEGREE == 0u && tti_replay_fits(kAbsResidualTtiBodySlots);
#else
#define ABS_RESIDUAL_TTI_CANDIDATE 0
constexpr bool kAbsResidualTtiReplay = false;
#endif

#if ABS_RESIDUAL_TTI_CANDIDATE
inline void abs_residual_tti_replay_tile() {
    static_assert(kAbsResidualTtiReplay);
    // One anonymous single-element replay body.  Register ownership:
    // L0 coordinate/z/t/result, L1 exponent/fraction then reloaded raw affine,
    // L2 Horner/short-lived exact constants, L3 exponent-scale scratch,
    // L4 correction c3, L5 correction c1, L6 correction c2, L7 exponent c0
    // then correction c0 scratch.  Prgm0/1/2 retain the
    // ordinary exponent leaf's multiplier and fused c2/c1 coefficients.
    constexpr uint32_t kCorrectionC3 =
        __builtin_bit_cast(uint32_t, ABS_RESIDUAL_NUM_COEFFS[3]);
    constexpr uint32_t kCorrectionC2 =
        __builtin_bit_cast(uint32_t, ABS_RESIDUAL_NUM_COEFFS[2]);
    constexpr uint32_t kCorrectionC1 =
        __builtin_bit_cast(uint32_t, ABS_RESIDUAL_NUM_COEFFS[1]);
    constexpr uint32_t kCorrectionC0 =
        __builtin_bit_cast(uint32_t, ABS_RESIDUAL_NUM_COEFFS[0]);
    constexpr uint32_t kExpC0 =
        __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, kCorrectionC3);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG5, kCorrectionC1);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG6, kCorrectionC2);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, kExpC0);

    TT_REPLAY_RECORD(kAbsResidualTtiBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPSETSGN(
        0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);  // |x|
#if defined(ABS_RESIDUAL_COORDINATE_TERMINAL)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_FLOATB,
        (ABS_RESIDUAL_COORDINATE_TERMINAL_BITS >> 16));
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        1);  // coordinate=min(|x|, bound)
#else
    // The same typed form may own the complete exponent range without a
    // declared coordinate terminal.  Preserve the two producer/consumer
    // issue slots; substituting a clamp here would change its finite tail.
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
    // Match the compiler body exactly: its immediate exponent bias lowers to
    // separate MUL then ADDI, not a fused MAD.  Fusing these two source
    // operations changes one BF16 output lane after final rounding.
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // xlog2=coordinate*(-log2e)
    TTI_SFPNOP;  // replay bypasses the MUL -> ADDI scoreboard stall
    TTI_SFPADDI(
        0x42fe,
        ckernel::p_sfpu::LREG0,
        0);  // xlog2 += exact 127.0 exponent bias
    TTI_SFPNOP;
    TTI_SFPEXEXP(
        0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPEXMAN(
        0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPSHFT(0, 1, 0, 0);
    TTI_SFPEXMAN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(
        ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        TT_ADDRMOD_DST_HOLD,
        0);  // raw BH DST word, fills the exponent Horner hazard slot
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPNOP;  // replay bypasses the MAD -> SETEXP scoreboard stall
    // The fitted exponent P2 is certified in [1,2), so the ordinary bare
    // SETEXP reconstruction is exact and leaves three replay slots for the
    // shared target-class finalizer below.
    TTI_SFPSETEXP(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        2);  // t=p*2^i, exponent source is the retained z bits in L0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG2,
        0);
    if constexpr (ABS_RESIDUAL_NUM_COEFFS[0] == 1.0f) {
        TTI_SFPNOP;
    } else {
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_UPPER,
            (kCorrectionC0 >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_LOWER,
            (kCorrectionC0 & 0xffffu));
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        (ABS_RESIDUAL_NUM_COEFFS[0] == 1.0f)
            ? ckernel::p_sfpu::LCONST_1
            : ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG1,
        0,
        TT_ADDRMOD_DST_HOLD,
        0);  // raw fp lane, fills the final correction hazard slot
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // residual=t*Q(t)
#if defined(ABS_RESIDUAL_POSITIVE_PART_PLUS)
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        9);  // affine=max(x,0)
#else
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        1);  // affine=min(x,0)
#endif
#if defined(ABS_RESIDUAL_POSITIVE_PART_PLUS)
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        0);
#else
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        0);
#endif
    auto same_row_suffix = []() {
#if TT_SPECIAL_NEG_INF == 3
        TTI_SFPMUL(
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG1,
            0);  // typed -Inf quotient terminal is +0
#elif TT_SPECIAL_NEG_INF == 2
        TTI_SFPSETEXP(
            255,
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LREG1,
            1);  // exact -Inf override value
#elif TT_SPECIAL_NEG_INF == 1
        TTI_SFPSETEXP(
            255,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG1,
            1);  // exact +Inf override value
#else
#error "abs-residual replay cannot encode declared -Inf quotient terminal"
#endif
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_USHORT,
            0x80ff);
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPAND(
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG2,
            1);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG2,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG3,
            0,
            sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);
        if constexpr (ABS_RESIDUAL_NUM_COEFFS[0] != 1.0f) {
            // The non-unit correction origin uses LREG7 after SETEXP, while
            // the next recorded exponent body expects its pinned exp c0
            // there.  Restore that schedule-owned cross-row resource.
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG7,
                sfpi::SFPLOADI_MOD0_UPPER,
                (kExpC0 >> 16));
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG7,
                sfpi::SFPLOADI_MOD0_LOWER,
                (kExpC0 & 0xffffu));
        }
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (int row = 1; row < 32; ++row) {
        TTI_REPLAY(0, kAbsResidualTtiBodySlots, 0, 0);
        same_row_suffix();
    }
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if EXP_HW_BOUNDED_TWO_SIDED_TTI_CANDIDATE
inline void bounded_exp_two_sided_tti_replay_tile() {
    static_assert(kBoundedTwoSidedTtiReplay);
    constexpr uint32_t kMultBits =
        __builtin_bit_cast(uint32_t, (float)EXP_HW_MULT);
    constexpr uint32_t kBiasBits =
        __builtin_bit_cast(uint32_t, (float)EXP_HW_EXPONENT_BIAS);
    constexpr uint32_t kC0Bits =
        __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
    constexpr uint32_t kNegSlopeBits =
        __builtin_bit_cast(uint32_t, (float)EXP_HW_TWO_SIDED_NEG_SLOPE);
    constexpr uint32_t kNegativePositiveScaleBits =
        __builtin_bit_cast(uint32_t, -(float)EXP_HW_TWO_SIDED_POS_SCALE);

    auto restore_cross_row_pins = [&]() {
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            sfpi::SFPLOADI_MOD0_UPPER,
            (kMultBits >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            sfpi::SFPLOADI_MOD0_LOWER,
            (kMultBits & 0xffffu));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG6,
            sfpi::SFPLOADI_MOD0_FLOATB,
            (kBiasBits >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_UPPER,
            (kC0Bits >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_LOWER,
            (kC0Bits & 0xffffu));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_BOUNDED_EXP_TWO_SIDED_COORDINATE_UPPER_BF16);
    };
    restore_cross_row_pins();

    TT_REPLAY_RECORD(kBoundedTwoSidedTtiBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 0);  // 1: x
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        0);  // 2: x*log2(e)+scaled bias
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG6,
        sfpi::SFPLOADI_MOD0_UPPER,
        (kNegSlopeBits >> 16));  // 3: producer gap
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        9);  // 4: max(xlog2, 0)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG6,
        sfpi::SFPLOADI_MOD0_LOWER,
        (kNegSlopeBits & 0xffffu));  // 5: lower-clamp retirement gap
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG0,
        1);  // 6: min(xlog2, typed right-bound coordinate)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_UPPER,
        (kNegativePositiveScaleBits >> 16));  // 7: clamp retirement gap
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 0);  // 8
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // 9
    TTI_SFPSHFT(0, 4, 0, 0);  // 10
    TTI_SFPEXMAN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        sfpi::SFPEXMAN_MOD1_PAD9);  // 11
    TTI_SFPCAST(5, 5, 0);  // 12
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG4,
        0);  // 13: P2 head
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_LOWER,
        (kNegativePositiveScaleBits & 0xffffu));  // 14: Horner gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG4,
        0);  // 15: P2 tail
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_FLOATB,
        0x4380);  // 16: 2*scale == 256, also Horner gap
    TTI_SFPSETEXP(0, 4, 0, 2);  // 17: scaled exp(x) in L0
    TTI_SFPNOP;  // 18: SETEXP result retirement
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);  // 19: s^2
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG3,
        0);  // 20: x*s, also square retirement gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG5,
        0);  // 21: s^2 + 256*s
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        0);  // 22: negative numerator, cross-term retirement gap
    TTI_SFPADDI(0x4700, ckernel::p_sfpu::LREG5, 0);  // 23: +32768
    TTI_SFPNOP;  // 24: denominator retirement
    TTI_SFPARECIP(
        0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, 0);  // 25
    TTI_SFPNOP;  // 26: reciprocal seed retirement
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG5,
        2);  // 27: reciprocal Newton residual
    TTI_SFPNOP;  // 28
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, 0);  // 29
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        3);  // 30: refined reciprocal
    TTI_SFPENCC(3, 0, 0, 10);  // 31
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG6,
        0);  // 32: negative numerator / denominator

    auto same_row_suffix = [&]() {
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG5,
            0);  // positive factor, also negative-ratio retirement gap
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            0);  // negative result, also positive-factor retirement gap
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG5,
            0);  // positive result
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);  // x < 0
        TTI_SFPMOV(
            0,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG5,
            0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG5,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(
            ckernel::p_sfpu::LREG5, 0, TT_ADDRMOD_DST_ADV, 0);
        restore_cross_row_pins();
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (int row = 1; row < 32; ++row) {
        TTI_REPLAY(0, kBoundedTwoSidedTtiBodySlots, 0, 0);
        same_row_suffix();
    }
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(HW_PRELOAD)
// ----------------------------------------------------------------------------
// GENERIC constant-pool preload variant of exp_hw_eval.
//
// Generalizes the proven exp deg-2 prototype: the codegen turns on HW_PRELOAD
// for EVERY exponent-ALU kind (exp2/log2/pow) at ANY degree (no per-activation
// hand-wiring). The mechanism is purely about WHERE the loop-invariant
// constexpr constants live so the recorded replay body re-loads none of them:
//
//   - vConstFloatPrgm0/1/2 : the 3 hottest constants, programmed ONCE before the
//     tile loop (persist across every replayed body + every tile).
//   - hoisted vFloat cv[]  : the remaining coefficients, declared as pre-loop
//     locals by the caller (hw_reduce) so GCC keeps them in iteration-invariant
//     LREGs — no SFPLOADI inside the recorded body.
//
// Per-kind ranking (deterministic, owned by the kernel since the kind dictates
// which constant is touched every element):
//   exp2 : prgm0 = MULT, prgm1 = c[DEG], prgm2 = c[DEG-1]; cv[] = c[DEG-2..0]
//          (255.0 clamp also hoisted; 0.0/127.0 are const-lane/SFPMAD imm).
// Anything past the prgm + LREG budget the compiler simply emits as in-body
// literals; the codegen logs that count (HW_PRELOAD_SPILL) — never silent.
//
// COMPOSE_SIGMOID and COMPOSE_SIGMOID_PRODUCT reserve vConstFloatPrgm0 for
// sfpu_reciprocal (it expects 2.0 there on BH, set by sfpu_reciprocal_init).
// In that case MULT is demoted from prgm0 to a hoisted LREG param (mult_hoist)
// so the two uses of prgm0 never collide. On WORMHOLE the reciprocal owns ALL
// of Prgm0/1/2 (its Sollya quadratic-seed constants are read at every
// sfpu_reciprocal_iter call), so under compose on WH the top-2 Horner coeffs
// are ALSO demoted — from prgm1/2 to foldable constexpr locals (see the
// Horner below and the PRGM CONSTANT OWNERSHIP RULE in the preamble).
//
// The remaining coefficients (c[DEG-2..0]) are read from the constexpr global
// EXP_HW_COEFFS: the compiler hoists what fits in the LREG budget and emits the
// rest as in-body literals (the natural "spill" the codegen reports) — this
// avoids forcing a fixed-size vFloat array that would overflow the register file
// under heavy paths like sigmoid (reciprocal + clamp + mult all live at once).
//
// Math is byte-identical to exp_hw_eval (same SFPMAD Horner, same order).
// cvspill points at a caller-hoisted, loop-invariant vFloat array holding the
// coefficients BELOW the two prgm-resident ones: cvspill[0]=c[DEG-2],
// cvspill[1]=c[DEG-3], ... cvspill[DEG-2]=c[0]. Materializing them as pre-loop
// LREGs (not reading EXP_HW_COEFFS[k] inside the body) is FIX A: it drives the
// in-body SFPLOADI count for the Horner chain to ZERO (top-2 in prgm, rest in
// LREGs) instead of 2-per-spilled-coeff. nullptr -> fall back to the global.
// cva/cvb/cvc: the first three below-prgm coefficients (c[DEG-2], c[DEG-3],
// c[DEG-4]) pre-laundered through hw_preload_pin by the caller so they arrive
// as LREG-resident values (NOT foldable constants). Deeper coefficients (DEG>4)
// still come from cvspill and re-materialize in-body (the logged spill).
template <uint32_t DEG>
inline vFloat exp_hw_eval_preloaded(
    vFloat x,
    vFloat thr_hi_hoist,
    vFloat c127_hoist,
    vFloat cva,
    vFloat cvb,
    vFloat cvc,
    const float* cvspill
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    ,
    vFloat mult_hoist
#endif
) {
#if defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
    vFloat x_raw = x;
#endif
    // MULT lives in prgm0 normally; under sigmoid compose prgm0 is owned by the
    // reciprocal, so MULT is read from the hoisted LREG instead. The bias 127.0
    // is supplied via a hoisted, loop-invariant register (c127_hoist) so the
    // compiler fuses x*MULT + 127 into ONE SFPMAD (matching TTNN exp_21f) rather
    // than emitting a separate SFPMUL + SFPADDI for the literal addend.
#if defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    vFloat x_original = x;
#endif
#if defined(EXP_HW_INPUT_ABS)
    x = setsgn(x, 0);
#endif
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    vFloat xlog2 = x * mult_hoist + c127_hoist;
#else
    vFloat xlog2 = x * vConstFloatPrgm0 + c127_hoist;
#endif

    // ------------------------------------------------------------------
    // INSTRUCTION-COUNT FAST PATH (matches TTNN's exp_21f schedule). Two
    // independent cuts, each gated by a generator-emitted macro so the generic
    // path stays correct for fits where the cut is unsafe:
    //   EXP_HW_FUSED       : fold the 2^-23 normalize into the coeffs (pre-scaled
    //                        in prgm1/prgm2/cvspill) -> removes one SFPMUL.
    //   EXP_HW_BARE_SETEXP : drop the pe correction -> removes one SFPEXEXP +
    //                        two SFPIADD. Valid when g(f) in [1,2) (c0>=1).
    // ------------------------------------------------------------------

    // Full-range safety clamp to [0, 255]. Two SFPSWAP (vec_min_max): the lower
    // max(0,xlog2) flushes inputs below ~-88 to 0, the upper min(xlog2,255)
    // prevents the implicit float->int conversion from wrapping. 0.0 uses the
    // const-0 lane (no SFPLOADI); 255.0 comes from the hoisted thr_hi_hoist.
    //
    // We measured the TTNN-style single-swap + masked-lower variants (v_if flush
    // and branch-free integer sign-mask); both cost MORE here than the two clean
    // swaps because the compiler can't hide the mask ops in SFPMAD windows the
    // way TTNN's hand-recorded TTI replay buffer does. The two swaps stay.
    vFloat thr_lo = 0.0f;
#if !defined(EXP_HW_SKIP_INPUT_CLAMP)
    ordered_min_max(thr_lo, xlog2);        // xlog2 = max(0, xlog2)
    ordered_min_max(xlog2, thr_hi_hoist);  // xlog2 = min(xlog2, 255)
#endif

    // Branch-free float->int: shift mantissa left by (exp - bias) bits.
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);

    vInt ep = exexp(z, ExponentMode::Biased);  // 2^(integer part)
    vMag fm = exman(z);                          // fraction * 2^23

    // Fraction for the 2^f refinement polynomial.
#if defined(EXP_HW_FUSED)
    // FUSED: skip the `* 0x1p-23f` normalize. The coeffs in prgm1/prgm2/cvspill
    // are pre-scaled by 2^-23k (generator), so Horner runs over the raw exman
    // fraction. Removes one SFPMUL per element (TTNN exp_21f trick).
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest);
#else
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest) * 0x1p-23f;
#endif

    // Degree-N Horner. Top-2 coeffs come from prgm regs; the rest from the
    // caller-hoisted LREG array cvspill (FIX A: no in-body SFPLOADI for them).
    // p = (((c[DEG])*f + c[DEG-1])*f + c[DEG-2])*f + ... + c[0]
#if (defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
     defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)) && defined(ARCH_WORMHOLE)
    // WH compose: Prgm1/2 belong to the reciprocal (Sollya seed constants read
    // at every sfpu_reciprocal_iter call below) — the top-2 Horner coeffs are
    // foldable constexpr locals instead (the baseline compose schedule: GCC
    // materializes them in-body; LREG pins ICE under compose, see the
    // call-site comment). BH keeps them in Prgm1/2 — its reciprocal owns only
    // Prgm0. See the PRGM CONSTANT OWNERSHIP RULE in the preamble.
#if defined(EXP_HW_FUSED)
    [[maybe_unused]] constexpr float exp_wh_ctop =
        EXP_HW_COEFFS[DEG] * exp_hw_fused_scale((int)DEG);
    [[maybe_unused]] constexpr float exp_wh_csec =
        (DEG >= 1) ? EXP_HW_COEFFS[DEG - 1] * exp_hw_fused_scale((int)DEG - 1) : 0.0f;
#else
    [[maybe_unused]] constexpr float exp_wh_ctop = EXP_HW_COEFFS[DEG];
    [[maybe_unused]] constexpr float exp_wh_csec = (DEG >= 1) ? EXP_HW_COEFFS[DEG - 1] : 0.0f;
#endif
#endif
    vFloat p;
    if constexpr (DEG >= 1) {
#if (defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
     defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)) && defined(ARCH_WORMHOLE)
        p = exp_wh_ctop * f + exp_wh_csec;  // c[DEG]*f + c[DEG-1] (no prgm reads)
#else
        p = vConstFloatPrgm1 * f + vConstFloatPrgm2;  // c[DEG]*f + c[DEG-1]
#endif
        // First three below-prgm coefficients live in caller-pinned LREGs
        // (cva/cvb/cvc): one SFPMAD each, zero in-body materialization.
        if constexpr (DEG >= 2) {
            p = p * f + cva;  // + c[DEG-2]
        }
        if constexpr (DEG >= 3) {
            p = p * f + cvb;  // + c[DEG-3]
        }
        if constexpr (DEG >= 4) {
            p = p * f + cvc;  // + c[DEG-4]
        }
#pragma GCC unroll 16
        for (int k = (int)DEG - 5; k >= 0; k--) {
            // Spill tail (DEG>4): cvspill index walks c[DEG-5] ... c[0].
            p = p * f + cvspill[(int)DEG - 2 - k];
        }
    } else {
#if (defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
     defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)) && defined(ARCH_WORMHOLE)
        p = exp_wh_ctop;  // degree-0: c[0] (no prgm reads)
#else
        p = vConstFloatPrgm1;  // degree-0: c[0]
#endif
    }

    // Recombine 2^i * 2^f.
#if defined(EXP_HW_BARE_SETEXP)
    // BARE: p is in [1,2) by construction (rising-to-2 exp2 fit, c0>=1), so a
    // plain setexp(p, ep) is correct. The lower bound (negative-input flush) is
    // already enforced by the max(0,xlog2) swap above. Removes one SFPEXEXP +
    // two SFPIADD vs the pe-corrected form below (TTNN exp_21f recombine).
    vFloat y = setexp(p, ep);
#else
    // setexp replaces p's exponent but keeps its mantissa, correct only when p
    // in [1,2). The fitter's g(0)=c[0] dips below 1.0 for some degrees
    // (odd-degree exp2 c0=0.99992), so p lands in [0.5,1) at f~=0 and a bare
    // setexp(p, ep) over-scales by 2x. Instead of the pe-corrected setexp
    // (SFPEXEXP + SFPMOV + 2x SFPIADD + SFPSETEXP = 5 ops), synthesize the
    // scale 2^i directly from the biased integer exponent ep (== i+127):
    // reinterpret(ep << 23) is EXACTLY the float 2^i, and p * 2^i is an exact
    // power-of-two scale — bit-identical to setexp(p, ep + pe - 127) for every
    // in-range input (both compute p * 2^(ep-127) exactly; they differ only
    // outside the fit domain, where ep+pe-127 would over/underflow the
    // exponent field). 2 ops instead of 5.
    vFloat y = p * as<vFloat>(ep << 23);  // 2^i * 2^f == base^x
#endif

#if defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
    constexpr float S = 1.0f / EXP_HW_COMPOSE_SCALE;
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(S + y);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(S + y);
#endif
    vFloat magnitude = setsgn(x_raw, 0);
    vFloat result = magnitude * (S * inv);
    v_if(x_raw < 0.0f) { result = 0.0f - (magnitude * y) * inv; }
    v_endif;
    y = result;
#elif defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
#if defined(USE_BF16)
    y = ckernel::sfpu::sfpu_reciprocal_iter<1>(1.0f + y);
#else
    y = ckernel::sfpu::sfpu_reciprocal_iter<2>(1.0f + y);
#endif
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    y = x * y;
#endif
#elif defined(EXP_HW_COMPOSE_MINUS_ONE)
    y = y - 1.0f;
#elif defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN)
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(y);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(y);
#endif
    y = EXP_HW_COMPOSE_SCALE * inv + y;
#elif defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR)
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(y);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(y);
#endif
    vFloat magnitude = y - EXP_HW_COMPOSE_SCALE * inv;
    vFloat ax = setsgn(x_original, 0);
    vFloat lower = ax * (1.0f + EXP_HW_ORIGIN_CUBIC * ax * ax);
    ordered_min_max(magnitude, lower);
    y = copysgn(magnitude, x_original);
#elif defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    vFloat square = y * y;
    vFloat denominator =
        (2.0f * EXP_HW_TWO_SIDED_SCALE) * y +
        square +
        2.0f * EXP_HW_TWO_SIDED_SCALE * EXP_HW_TWO_SIDED_SCALE;
#if defined(USE_BF16)
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<1>(denominator);
#else
    vFloat inv = ckernel::sfpu::sfpu_reciprocal_iter<2>(denominator);
#endif
    vFloat negative_numerator =
        EXP_HW_TWO_SIDED_NEG_SLOPE * y + EXP_HW_TWO_SIDED_NEG_BIAS;
    vFloat negative = (x_original * y) * (negative_numerator * inv);
    vFloat positive =
        x_original * (1.0f - EXP_HW_TWO_SIDED_POS_SCALE * inv);
    y = positive;
    v_if(x_original < 0.0f) { y = negative; }
    v_endif;
#endif
    return y;
}
#endif  // HW_PRELOAD
#endif

#if EXP_HW_HYPERBOLIC_TTI_CANDIDATE && \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR)
// Cancellation-safe odd hyperbolic replay. The 29-slot recorded core owns the
// selected exponential magnitude. A 14-issue same-row suffix then reloads x,
// computes the global lower factor x*(1+c*x*x), restores sign, rounds, and
// stores. Deferring lower-factor writes until after slot 29 is the exact
// lifetime barrier: Horner's coefficient sources are never overwritten in the
// instruction immediately following their consuming MAD.
//
// The second ordinary DST reload is also the carrier for the existing BH
// load-macro fixed-store slot. Its delayed store reads the signed result after
// the unchanged RNE/SETSGN sequence, eliminating only the standalone STORE.
// The load source, latched row address, ADDR_MOD_6 walk, and one-row loop are
// unchanged; this is anonymous dependency-gap filling for this certified form.
inline void exp_hw_hyperbolic_odd_tti_lm_init() {
    // Sequence register 0: fixed STORE (mux 3), delay 2 => LM issue + 3.
    // Simple/MAD/Round are idle. Misc makes the store inherit the LM's
    // unchanged BF16 store modifier and count issued instructions.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_UPPER,
        0x1300);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_LOWER,
        0x0000);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
}

inline void exp_hw_hyperbolic_odd_tti_tile() {
    constexpr uint32_t kMultBits =
        __builtin_bit_cast(uint32_t, (float)EXP_HW_MULT);
    constexpr uint32_t kC2Bits = __builtin_bit_cast(
        uint32_t, EXP_HW_COEFFS[2] * exp_hw_fused_scale(2));
    constexpr uint32_t kC1Bits = __builtin_bit_cast(
        uint32_t, EXP_HW_COEFFS[1] * exp_hw_fused_scale(1));
    constexpr uint32_t kC0Bits =
        __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
    constexpr uint32_t kNegComposeScaleBits =
        __builtin_bit_cast(uint32_t, -(float)EXP_HW_COMPOSE_SCALE);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG1, kMultBits);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG2, kC2Bits);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG3, kC1Bits);
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, kC0Bits);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_FLOATB,
        0x42fc);  // exact 126.0 exponent bias
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_UPPER,
        (kNegComposeScaleBits >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_LOWER,
        (kNegComposeScaleBits & 0xffffu));
    TTI_SFPCONFIG(0x5555, 11, 8);

    TT_REPLAY_RECORD(kHyperbolicOddTtiCoreSlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // 1: x
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);  // 2: |x|
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG0,
        0);  // 3: xlog2
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);  // 4: 255
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, 1);  // 5: clamp
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);  // 6
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // 7
    TTI_SFPSHFT(0, 5, 0, 0);  // 8
    TTI_SFPEXEXP(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        sfpi::SFPEXEXP_MOD1_NODEBIAS);  // 9
    TTI_SFPEXMAN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPEXMAN_MOD1_PAD9);  // 10
    TTI_SFPCAST(0, 0, 0);  // 11
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG6,
        0);  // 12: Horner head
    TTI_SFPNOP;  // 13: preserve L7 and the Horner-head source lifetime
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        0);  // 14
    TTI_SFPNOP;  // 15: preserve L2 until the preceding Horner MAD retires
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG6,
        0);  // 16
    TTI_SFPNOP;  // 17: preserve L3 until the preceding Horner MAD retires
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        0);  // 18: Horner c0
    TTI_SFPSHFT(0x017, 5, 5, 7);  // 19: 2^i
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // 20: y
    TTI_SFPNOP;  // 21: y MAD hazard gap
    TTI_SFPARECIP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);  // 22
    TTI_SFPNOP;  // 23: conservative ARecip window
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG6,
        2);  // 24: reciprocal Newton residual
    TTI_SFPNOP;  // 25
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG6, 0, 0);  // 26
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        3);  // 27
    TTI_SFPENCC(3, 0, 0, 10);  // 28
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        0);  // 29: y-scale/y (L11 is programmed to -scale)
    auto same_row_suffix = []() {
        // 1: ordinary FP32 DST reload of effective x, not a raw-class shadow.
        TTI_SFPLOAD(ckernel::p_sfpu::LREG5, 0, TT_ADDRMOD_DST_HOLD, 0);
        // SFPLOAD is conservatively gap-1 in a replay-rate stream.
        TTI_SFPNOP;
        // 3..10: lower=|x|*(1+c*x^2), then max(exponential, lower).
        TTI_SFPSETSGN(
            0,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG7,
            1);
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG6,
            0);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG5,
            sfpi::SFPLOADI_MOD0_FLOATB,
            0x3e2a);  // exact descriptor cubic 0.166015625; x^2 gap
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG5,
            0);
        TTI_SFPNOP;
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG6,
            0);
        TTI_SFPNOP;
        TTI_SFPSWAP(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG6,
            9);
        // Reload x only after the cubic temporary is dead. This is the same
        // ordinary FP32 DST read as the plain suffix, now issued through the
        // BH load-macro so its delayed fixed STORE can retire this same row.
        // VD=L5 is overwritten by SETSGN before STORE fires at LM+3.
        TTI_SFPLOADMACRO(
            (0 << 2) | (ckernel::p_sfpu::LREG5 & 3),
            0,
            TT_ADDRMOD_DST_ADV,
            (ckernel::p_sfpu::LREG5 >> 2));
        // RNE is independent of the raw sign reload, so it is also the exact
        // load-consumer gap. Magnitude-round then sign-restore is bit-identical
        // to sign-restore then round for the finite numeric contract.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG6,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSETSGN(
            0,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG5,
            0);
        // L7 carried |x| in the suffix. Restore the exact exponent bias before
        // the next replay consumes this cross-row pin at core slot 3.
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_FLOATB,
            0x42fc);
        // The macro-fixed STORE fires here, after SETSGN's ordinary hazard
        // gap, and reads signed L5 using the row address latched above.
    };
    same_row_suffix();
    TT_REPLAY_REST_WITH_SAME_ROW_SUFFIX(
        kHyperbolicOddTtiCoreSlots, 31, same_row_suffix());
    TT_REPLAY_TILE_EPILOGUE();

    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xbf80);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPCONFIG(0x5555, 11, 8);  // restore architectural -1.0
}
#endif

#if defined(EXPONENT_ALU_LOG2)
// log2(x) = e + log2(m) for x = 2^e * m, m in [1,2). The fitter fits h(m)=log2(m)
// on [1,2) and emits NATURAL [1,2]-basis coeffs (LOG_HW_C0..); the kernel
// extracts e (integer log2, free via exexp) and m (exman/setexp), evaluates the
// poly, adds e, and applies an optional base scale (ln2 for log, log10(2) for
// log10) via LOG_HW_SCALE.
#ifndef LOG_HW_SCALE
#define LOG_HW_SCALE 1.0f
#endif
// log1p(x) = log(x + 1): the fitter tags expalu_input_offset = 1.0 so the log2
// decompose operates on (x + offset). Default 0.0 keeps plain log/log2/log10.
#ifndef LOG_HW_INPUT_OFFSET
#define LOG_HW_INPUT_OFFSET 0.0f
#endif
template <uint32_t DEG>
inline vFloat log_hw_eval(vFloat x) {
    vFloat x_in = x;  // original (used for special-case tests)
    constexpr float OFFSET = LOG_HW_INPUT_OFFSET;
    vFloat xd = (OFFSET != 0.0f) ? (x + OFFSET) : x;

    // Extract biased exponent e and mantissa m in [1,2) of the decompose input.
    vInt biased = exexp(xd, ExponentMode::Biased);
    vInt e_int = biased - 127;
    vFloat m = setexp(xd, 127);

    // Horner h(m)=log2(m). The fitter tags the polynomial basis: the natural
    // [1,2] basis evaluates in m, the m_minus_1 basis (the codegen default for
    // log2) evaluates in (m-1) so c0==0 and h(1)==0 exactly.
#ifdef LOG_HW_BASIS_M_MINUS_1
    vFloat u = m - 1.0f;
#else
    vFloat u = m;
#endif
    const float* c = LOG_HW_COEFFS;
    vFloat h = c[DEG];
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        h = h * u + c[k];
    }

    // e + h(m); int32_to_float wants sign-magnitude for negatives.
    v_if(e_int < 0) { e_int = as<vInt>(setsgn(as<vSMag>(~e_int + 1), 1)); }
    v_endif;
    vFloat e_float = convert<vFloat>(as<vSMag>(e_int), RoundMode::Nearest);
    vFloat result = (e_float + h) * LOG_HW_SCALE;

    // Special cases on the DECOMPOSE input (x + offset): log(0) = -inf, log(neg) = NaN.
    // For log1p this is the (x + 1) singularity at x = -1. Range-controlled
    // sweeps can opt out when metadata proves the decompose input is positive.
#ifndef LOG_HW_SKIP_SPECIALS
    v_if(xd < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(xd == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;
#endif
    (void)x_in;
    return result;
}

#if defined(HW_PRELOAD)
// GENERIC constant-pool preload variant of log_hw_eval (any degree).
// Ranking: prgm0 = LOG_HW_SCALE, prgm1 = c[DEG], prgm2 = c[DEG-1];
//   cva/cvb/cvc/cvd = c[DEG-2..DEG-5]  (caller-pinned LREGs, hw_preload_pin —
//   plain hoisted locals get constant-folded back into 2 SFPLOADI per coeff
//   per element, the exact schedule the disassembly showed for log/log10/log2);
//   cvspill = any deeper tail (in-body re-materialize, the logged spill).
// 127 / 1.0 are SFPMAD imm / const-lane. Math byte-identical to log_hw_eval
// (same op kinds, same order — an LREG-resident coeff MAD computes the same
// fp32 result as a loadi-resident one).
// When LOG_HW_SCALE == 1.0 (plain log2) the final scale multiply is skipped
// at compile time: (e+h)*1.0f is bit-identical to (e+h) for every input the
// specials-filtered path can produce.
template <uint32_t DEG>
inline vFloat log_hw_eval_preloaded(
    vFloat x, vFloat cva, vFloat cvb, vFloat cvc, vFloat cvd, const float* cvspill) {
    constexpr float OFFSET = LOG_HW_INPUT_OFFSET;
    vFloat xd = (OFFSET != 0.0f) ? (x + OFFSET) : x;

    vInt biased = exexp(xd, ExponentMode::Biased);
    vInt e_int = biased - 127;
    vFloat m = setexp(xd, 127);

#ifdef LOG_HW_BASIS_M_MINUS_1
    vFloat u = m - 1.0f;
#else
    vFloat u = m;
#endif
    // CC-free sign-magnitude conversion (the TTNN/log1p idiom): convert |e|
    // then copy the two's-complement sign bit. Bit-identical to the predicated
    // setsgn(~e+1) form for every |e| <= 254, and — unlike the v_if block,
    // whose SETCC/ENCC would predicate anything scheduled between them — these
    // three ops are independent of the Horner chain, so the compiler can slot
    // them into the SFPMAD latency shadows instead of SFPNOPs (hence they are
    // computed BEFORE the Horner in source order).
    vFloat e_float = convert<vFloat>(abs(e_int), RoundMode::Nearest);
    e_float = copysgn(e_float, as<vFloat>(e_int));

    vFloat h;
    if constexpr (DEG >= 1) {
        h = vConstFloatPrgm1 * u + vConstFloatPrgm2;  // c[DEG]*u + c[DEG-1]
        if constexpr (DEG >= 2) {
            h = h * u + cva;  // + c[DEG-2]
        }
        if constexpr (DEG >= 3) {
            h = h * u + cvb;  // + c[DEG-3]
        }
        if constexpr (DEG >= 4) {
            h = h * u + cvc;  // + c[DEG-4]
        }
        if constexpr (DEG >= 5) {
            h = h * u + cvd;  // + c[DEG-5]
        }
#pragma GCC unroll 16
        for (int k = (int)DEG - 6; k >= 0; k--) {
            h = h * u + cvspill[(int)DEG - 6 - k];  // c[DEG-6], c[DEG-7], ... c[0]
        }
    } else {
        h = vConstFloatPrgm1;  // degree-0: c[0]
    }

    vFloat result = e_float + h;
    if constexpr ((float)(LOG_HW_SCALE) != 1.0f) {
        result = result * vConstFloatPrgm0;  // * LOG_HW_SCALE
    }

#ifndef LOG_HW_SKIP_SPECIALS
    v_if(xd < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(xd == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;
#endif
    return result;
}
#endif  // HW_PRELOAD

// ----------------------------------------------------------------------------
// TTI REPLAY lowering for the log family (log / log10 / log2 via LOG2 kind).
//
// Interleave decision: the per-element body is 15-19 real ops (see below), so
// a 2-element interleave is 30-38 slots — at DEG >= 4 it does NOT fit the
// 32-slot replay buffer (DEG=6 + scale = 38; even DEG=5 no-scale is 34), and
// at the fitting low degrees the selected corpus has no configs. The lowering
// is therefore SINGLE-element with hazard gaps: the e-side ops (ABS/CAST/
// SETSGN), which are independent of the Horner chain, fill the first three
// MAD-latency gaps; the remaining gaps are SFPNOPs. Byte-identity: same op
// kinds, same order, same operands as log_hw_eval_preloaded (SFPADD/SFPMUL
// are MAD-unit ops with an exact-1.0/exact-0.0 operand — single rounding,
// bit-identical to the sfpi-emitted sfpadd/sfpmul).
//
// Per-element schedule (m_minus_1 basis, DEG=6, scale != 1 — the largest):
//   LOAD L2 | NOP | EXEXP L1<-L2 nodebias | SETEXP L2 imm127 | IADD L1 -127
//   | ADD L2+=(-1) [u] | ABS L0=|L1| | MAD h=P1*u+P2 -> L3 | CAST L0
//   | MAD +c[D-2](L7) | SETSGN L1={sgn L1, mag L0} | MAD +c[D-3](L6) | NOP
//   | MAD +c[D-4](L5) | NOP | MAD +c[D-5](L4) | NOP | MAD +c0(L9=0) | NOP
//   | ADD L3=L1*1+L3 [e+h] | NOP | MUL L3*=L12 [scale] | NOP | RND | STORE
// Every MAD-unit consumer sits >= 1 op after its producer; bit-unit results
// (EXEXP/SETEXP/IADD/ABS/CAST/SETSGN/RND) are consumed gap-0 (safe class).
// The final SFPSTORE walks dst via ADDR_MOD_6 dest += 2 (one vector).
//
// Gates: c0 must be exactly 0.0 (it rides the hardware zero register L9 —
// every selected log-family fit has c0 == 0 by construction of the m_minus_1
// basis); DEG in [2,6] so c1..c[DEG] fit prgm1/2 + the four per-tile pinned
// LREG7..LREG4; specials skipped (the NaN/-inf CC block is not scheduled);
// no input offset. LOG_HW_TTI_DISABLE=defined forces the sfpi path (A/B).
#if defined(HW_PRELOAD) && defined(LOG_HW_SKIP_SPECIALS) && defined(USE_BF16) && \
    !defined(LOG_HW_TTI_DISABLE) &&                                                \
    ((defined(ARCH_BLACKHOLE) &&                                                   \
      (!defined(TT_DOMAIN_ACTION_PROGRAM) ||                                       \
       defined(TT_EXPONENT_ALU_LOG2_NONPOSITIVE_CLASS))) ||                        \
     (defined(ARCH_WORMHOLE) && !defined(TT_DOMAIN_ACTION_PROGRAM) &&              \
      !defined(TT_EXPONENT_ALU_LOG2_NONPOSITIVE_CLASS)))
#define LOG_HW_TTI_CANDIDATE 1
constexpr uint32_t log_hw_tti_body_slots() {
    // DEG==2 m_minus_1 no-scale baseline = 16 (schedule above minus the
    // DEG>2 rungs and the scale pair).
    uint32_t n = 16;
    if (LOG_HW_DEGREE >= 3) n += 1;                            // rung gapped by SETSGN
    if (LOG_HW_DEGREE >= 4) n += 2u * (LOG_HW_DEGREE - 3u);    // NOP+MAD rungs
    if ((float)(LOG_HW_SCALE) != 1.0f) n += 2;                 // NOP + SFPMUL
#ifndef LOG_HW_BASIS_M_MINUS_1
    n -= 1;  // natural basis: no u = m - 1 SFPADD
#endif
    return n;
}
constexpr uint32_t kLogHwTtiBodySlots = log_hw_tti_body_slots();
constexpr uint32_t kLogHwTtiRouteSlots =
#if defined(TT_EXPONENT_ALU_LOG2_NONPOSITIVE_CLASS)
    7u;
#else
    0u;
#endif
constexpr bool kLogHwTtiReplay =
    (LOG_HW_DEGREE >= 2) && (LOG_HW_DEGREE <= 6) && (LOG_HW_COEFFS[0] == 0.0f) &&
    ((float)(LOG_HW_INPUT_OFFSET) == 0.0f) &&
    tti_replay_fits(kLogHwTtiBodySlots + kLogHwTtiRouteSlots);
#if defined(TT_EXPONENT_ALU_LOG2_NONPOSITIVE_CLASS)
static_assert(
    TT_EXPONENT_ALU_LOG2_INPUT_MIN_NORMAL_BITS == 0x00800000u &&
        TT_EXPONENT_ALU_LOG2_TERMINAL_SLOTS == kLogHwTtiRouteSlots &&
        TT_EXPONENT_ALU_LOG2_REPLAY_BODY_SLOTS ==
            kLogHwTtiBodySlots + kLogHwTtiRouteSlots,
    "typed log2 total-form descriptor does not match LOG_TTI replay");
#elif defined(TT_INLINE_PROGRAM_OVERLAY)
static_assert(
    !kLogHwTtiReplay,
    "typed core overlay requires route-aware LOG_TTI replay; refusing silent fallback");
#endif
// ----------------------------------------------------------------------------
// SFPLOADMACRO tail fold for the log-family replay body (floor-probe pilot).
//
// The plain replay body ends in NOP|ADD|NOP|MUL|NOP|RND|STORE (scale != 1;
// NOP|ADD|NOP|RND|STORE when scale == 1). One SFPLOADMACRO folds the
// MUL/RND/STORE (RND/STORE) tail into scheduled macro slots that fire during
// the NEXT replay iteration's CC-free head, deleting 4 (2) issue slots:
//
//   ... MAD +c0 | NOP | LM(seq0, VD=L0, ADDR_MOD_6) | ADD e_f+h -> L0
//
// ISA facts this construction rests on (Tensix SFPU ISA + "Using LOADMACRO
// Safely"; both under docs/quasar/):
//   - A macro-fired unit op ALWAYS has one source overridden with the LM's VD
//     register (SRCB_OVERRIDE selects VB vs VC) and its dest overridden with
//     VD (unless USE_STAGING). The value chain must therefore flow through
//     VD: the normally-issued ADD writes e_f+h into VD(=L0), the macro MUL
//     (srcB-override) reads and rewrites VD, the macro RND (srcC-override) reads VD
//     and writes the staging flop, the macro STORE reads staging.
//   - The LM's own load (input row -> VD) is dead data; the ADD overwrites it
//     before the first macro slot fires.
//   - The delayed macro STORE uses the address latched at LM issue time
//     (in-tree precedent: typecast LOADMACRO carries ADDR_MOD_6 and its
//     delay-4 store still hits the loaded row), so the LM itself carries the
//     ADDR_MOD_6 dest+=2 walk at the same body position the deleted SFPSTORE
//     held (D-RWC accounting unchanged; TT_REPLAY_TILE_EPILOGUE unchanged).
//   - Fire schedule (delays count issued SFPU instructions; the Misc config
//     sets UNIT_DEPENDENCY for all units): MUL@LM+3, RND@LM+5, STORE@LM+7
//     (scale) / RND@LM+3, STORE@LM+5 (no scale) — every macro consumer sits
//     2 instructions after its producer (the proven MAD|NOP|consumer gap) and
//     lands opposite Load/Simple-unit head ops of the next iteration (no
//     unit double-issue; RND routes to staging so the shared Simple/Round
//     write port stays with the head's SETEXP/EXEXP writes).
//   - VD = L0: dead from SETSGN to next iteration's ABS in every DEG 2..6
//     body; never a pinned coefficient register (L4-L7) — the LM neither
//     reads nor leaves live state across iterations.
//
// Drain: the STORE slot is the deepest in-flight op (LM+7 / LM+5, i.e. 6 / 4
// instructions past the body's last slot); 6 trailing SFPNOPs after the final
// TT_REPLAY_REST cover both variants before TT_REPLAY_TILE_EPILOGUE().
// Byte-identity: same op kinds/operands/order as the plain replay body (MUL
// operand order commuted a*b -> b*a: bit-identical; carrier register L0
// instead of L3: value-identical), so byte-identity vs BOTH the plain-replay
// arm (LOG_LM_DISABLE) and the sfpi twin (LOG_HW_TTI_DISABLE) is the gate.
// NOT ADOPTED (opt-in only): the LOADMACRO tail is silicon-proven safe and
// byte-identical, but macro-fired ops consume replay-stream slots 1:1, so
// the measured gain is inside run-to-run drift for log/log10 (1.67 vs 1.70,
// 1.74 vs 1.77) and log2 REGRESSES (1.74 vs 1.70). Kept as the documented
// negative + reference implementation; enable with LOG_LM_ENABLE for A/B.
#if defined(LOG_LM_ENABLE) && !defined(LOG_LM_DISABLE) && \
    !defined(TT_EXPONENT_ALU_LOG2_NONPOSITIVE_CLASS)
#define LOG_LM_CANDIDATE 1
constexpr uint32_t kLogHwLmBodySlots =
    kLogHwTtiBodySlots - (((float)(LOG_HW_SCALE) != 1.0f) ? 4u : 2u);
constexpr bool kLogHwLmReplay = kLogHwTtiReplay;
#else
#define LOG_LM_CANDIDATE 0
constexpr uint32_t kLogHwLmBodySlots = 0;
constexpr bool kLogHwLmReplay = false;
#endif
#else
#define LOG_HW_TTI_CANDIDATE 0
#define LOG_LM_CANDIDATE 0
constexpr bool kLogHwTtiReplay = false;
constexpr bool kLogHwLmReplay = false;
#endif
#endif

#if defined(EVAL_METHOD_NEWTON_ROOT) || \
    defined(EVAL_METHOD_ZONE_FACTORED_EXP_ROOT_CORRECTION_2)
// ============================================================================
// Newton-Raphson magic-seed square root (mirrors TTNN's native ckernel sqrt).
//
// This is the INTEGER-ROOT fast path. Instead of decomposing x = 2^e * m and
// rebuilding the exponent (the pow_hw path, ~39 SFPU instrs), we operate
// directly on the raw IEEE bits:
//
//   i  = bits(x) >> 1                                  (one shift)
//   y0 = bits( MAGIC - i )                             (one integer subtract)
//
// The magic-constant subtraction produces a seed y0 ~= 1/sqrt(x)-ish whose
// EXPONENT is already halved AND whose parity (the sqrt(2) odd-exponent scale)
// is already folded into the seed -- so there is NO exexp/setexp/cast/parity
// cascade at all. Two Newton-Raphson refinement steps recover full fp32
// precision. The 3 loop-invariant constants (MAGIC seed, two Newton coeffs)
// live in the programmable const registers (preloaded once), so the recorded
// per-element body reloads NONE of them.
//
// Algorithm: SQRT_23-bits from Kokosinski et al. (2024), the exact algorithm
// the native blackhole ckernel_sfpu_sqrt.h uses. The fitter owns the seed
// magic and Newton coefficients (metadata: newton_root_magic / newton_root_c1
// / newton_root_c2), so this generalizes to any even-root / accuracy target it
// chooses to tune -- it is not hand-wired for sqrt.
#ifndef NEWTON_ROOT_MAGIC
#define NEWTON_ROOT_MAGIC 0x5f1110a0
#endif
#ifndef NEWTON_ROOT_C1
#define NEWTON_ROOT_C1 2.2825186f
#endif
#ifndef NEWTON_ROOT_C2
#define NEWTON_ROOT_C2 2.2533049f
#endif
#ifndef NEWTON_ROOT_C0
#define NEWTON_ROOT_C0 0x1.c09806p0f
#endif
// Root order N (2 = sqrt/rsqrt, 3 = cbrt) and the rsqrt/inverse flavour are
// metadata-driven (newton_root_n / newton_root_reciprocal). Defaults keep the
// path byte-identical to native sqrt when no extra tags are emitted.
#ifndef NEWTON_ROOT_N
#define NEWTON_ROOT_N 2
#endif
#ifndef NEWTON_ROOT_ITERS
#define NEWTON_ROOT_ITERS 3
#endif

#if (NEWTON_ROOT_N == 2) && !defined(NEWTON_ROOT_RECIPROCAL)
// --- sqrt: magic seed + SQRT_23-bit double-Newton (native parity) ------------
inline vFloat newton_root_sqrt(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(vConstIntPrgm0 - i);  // MAGIC seed (preloaded)

    // SQRT_23-bits: two Newton-Raphson steps. prgm1/prgm2 preloaded.
    vFloat xy = x * y;
    vFloat negative_y = -y;
    vFloat c = negative_y * xy;
    y = y * (vConstFloatPrgm1 + c * (vConstFloatPrgm2 + c));
    xy = x * y;
#if NEWTON_ROOT_ITERS == 1
    return xy;
#elif NEWTON_ROOT_ITERS == 2
    negative_y = -y;
    vFloat one_minus_xyy = vFloat(1.0f) + (negative_y * xy);
    vFloat half_xy = addexp(xy, -1);  // 0.5*xy via exponent decrement (no 0.5 immediate)
    vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
    // Skip the final correction at x==inf (avoids inf-inf = NaN; y already inf).
    v_if(as<vInt>(x) < as<vInt>(infinity)) { y = one_minus_xyy * half_xy + xy; }
    v_endif;
    // sqrt of a negative is NaN.
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    return y;
#else
#error "SQRT23 newton_root requires NEWTON_ROOT_ITERS equal to one or two"
#endif
}
#endif  // sqrt variant

// ----------------------------------------------------------------------------
// TTI REPLAY lowering for newton_root sqrt (plain replay, NO LOADMACRO).
//
// 1:1 transcription of the sfpi/GCC BH schedule of newton_root_sqrt (verified
// against the compiled twin's objdump) with rule-1 hazard gaps after every
// MAD-unit producer. 26 slots, single element per body. Two CC windows, both
// CC-balanced inside the recorded body (the wave-6-proven pattern):
//   - IADD(mod2) sets CC on bits(x)-bits(+inf) < 0, one predicated MAD
//     (the final Newton correction), ENCC;
//   - SETCC on sign(x), one predicated LOADI (qNaN), ENCC.
// Neither CC-setting op reads a MAD-unit result (IADD reads the SFPLOAD'd x
// and an SFPLOADI constant; SETCC reads x). The +inf constant is hoisted to
// the load-gap slot and renamed L3 -> L6 (GCC reused L3; the hoist would
// otherwise collide with the C2+c temporary) — operand VALUES are identical,
// so results are byte-identical to the sfpi twin.
// Registers: L12 = MAGIC (vConstIntPrgm0), L13/L14 = C1/C2 (prgm1/2), all
// preloaded once in the preamble; L0-L6 body-owned, every register written
// before read within one iteration (no pins, no cross-iteration state).
// A verifier-bound typed Program may add a six-slot raw exponent-zero suffix:
// raw U16 load, exponent-mask load, AND, EQ0, predicated +0, ENCC.  Blackhole
// DST stores exponent bits in physical bits 7:0, so this restores the declared
// DAZ ingress quotient on exactly all 256 signed zero/subnormal words after the
// bit-seed Newton body.  The body is then exactly 32/32 slots and remains one
// replay and one tile loop.
// Gate: BH + bf16 + sqrt shape (N==2, no reciprocal); NEWTON_TTI_DISABLE=1
// vetoes (A/B escape: the sfpi loop compiles unchanged).
// NEWTON SQRT IS NOT WH-READY. Two Blackhole-only modifiers, one fixed and
// one still open. Do not re-open this gate until both are done.
//
//  FIXED  SFPSHFT_MOD1_ARG_IMM_USE_VC at the seed shift (see the body). WH
//         masked Mod1 &= 1 and sourced from VB == VD, so it ignored x and
//         shifted LREG0 -- which still held the PREVIOUS element's result from
//         slot 24's SFP_STOCH_RND. That self-feeding seed overflowed to inf and
//         was the measured 1598 non-finite outputs. Staging x with an SFPMOV
//         removes them (measured: no non-finites afterwards).
//
//  OPEN   SFPMAD_MOD1_NEGATE_VA on the `1 - y*xy` step. Wormhole has no negate
//         step (WormholeB0/.../SFPMAD.md:44-47), so it computes y*xy + 1 --
//         wrong sign, finite output. Measured after the shift fix:
//         pure ULP 6373.0498046875 @ 4.60us against the sfpi arm's
//         0.4995107650756836 @ 5.63us. The runtime moves, so the body IS
//         engaging; it is simply wrong.
//         The trig fix (LCONST_neg1 as the multiplicand) does not transfer
//         directly: here VA is a live value, not a constant, so negating it
//         needs a scratch LREG and this body's allocation must be re-derived
//         before claiming one. That analysis is not done.
#if defined(USE_BF16) && (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && \
    !defined(NEWTON_ROOT_RECIPROCAL) && \
    !defined(NEWTON_LOWERING_DISABLE) && (NEWTON_ROOT_N == 2) && (NEWTON_ROOT_ITERS == 2) && \
    (!defined(TT_INLINE_PROGRAM_OVERLAY) || \
     (defined(ARCH_BLACKHOLE) && \
      defined(TT_SELECTED_NEWTON_SQRT_RAW_ZERO_TERMINAL)))
#define NEWTON_TTI_CANDIDATE 1
// The Wormhole form stages the shift source with an extra SFPMOV (see the body).
#if defined(ARCH_WORMHOLE)
constexpr uint32_t kNewtonSqrtTtiBodySlots = 27;
#elif defined(TT_SELECTED_NEWTON_SQRT_RAW_ZERO_TERMINAL)
constexpr uint32_t kNewtonSqrtTtiBodySlots = 32;
static_assert(
    TT_SELECTED_NEWTON_SQRT_REPLAY_BODY_SLOTS == kNewtonSqrtTtiBodySlots,
    "verified Newton sqrt replay slot count changed");
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
static_assert(TT_SELECTED_NEWTON_SQRT_NEGATIVE_SIGN_MASK == 0x8000u);
static_assert(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_OUTPUT_BF16 == 0x7f80u);
static_assert(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_PATTERN_COUNT == 128u);
#endif
#else
constexpr uint32_t kNewtonSqrtTtiBodySlots = 26;
#endif
constexpr bool kNewtonSqrtTtiReplay = tti_replay_fits(kNewtonSqrtTtiBodySlots);
#if defined(TT_INLINE_PROGRAM_OVERLAY) && \
    !defined(TT_SELECTED_NEWTON_SQRT_RAW_ZERO_TERMINAL)
static_assert(
    !kNewtonSqrtTtiReplay,
    "typed core overlay requires route-aware NEWTON_TTI replay; refusing silent fallback");
#endif
#else
#define NEWTON_TTI_CANDIDATE 0
constexpr bool kNewtonSqrtTtiReplay = false;
#endif

// Reciprocal SQRT23 common-SFPI compiles to 25 issued operations per element.
// The exact transcription below adds five mandatory replay hazard gaps and
// therefore closes at 30/32 slots.  It is selected solely by the parsed root
// resource; the public native tile remains the escape arm.
#if defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    defined(NEWTON_ROOT_RECIPROCAL) && \
    defined(NEWTON_ROOT_ALGORITHM_SQRT_23BIT) && \
    defined(NEWTON_ROOT_BACKEND_NATIVE_SFPU_TILE) && \
    !defined(NEWTON_LOWERING_DISABLE) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    (NEWTON_ROOT_N == 2) && (NEWTON_ROOT_ITERS == 1)
#define NEWTON_RSQRT_TTI_CANDIDATE 1
constexpr uint32_t kNewtonRsqrtTtiBodySlots = 30;
constexpr bool kNewtonRsqrtTtiReplay = tti_replay_fits(kNewtonRsqrtTtiBodySlots);
#else
#define NEWTON_RSQRT_TTI_CANDIDATE 0
constexpr bool kNewtonRsqrtTtiReplay = false;
#endif

// The Blackhole public BF16 cbrt body compiles to exactly fifteen arithmetic,
// conversion, and store issues plus a separate destination increment.  The
// transcription below folds that increment into the store's ordinary
// ADDR_MOD_6 walk, so one 23-slot record covers all 32 destination rows.  Its
// constants are derived only from the parsed NativeRoot graph.  The validation
// escape directly issues the same graph-owned arithmetic body 32 times, so it
// changes only replay scheduling.  The eight explicit gaps are
// required by the replay framework's conservative producer/consumer contract;
// they materialize stalls the public direct stream otherwise leaves to the
// Blackhole scheduler and do not change an arithmetic operand or rounding.
#if defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    !defined(NEWTON_ROOT_RECIPROCAL) && \
    defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC) && \
    defined(NEWTON_ROOT_BACKEND_NATIVE_SFPU_TILE) && \
    !defined(NEWTON_LOWERING_DISABLE) && \
    !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    (NEWTON_ROOT_N == 3) && (NEWTON_ROOT_ITERS == 1)
#define NEWTON_CBRT_EXACT_DESCRIPTOR 1
#else
#define NEWTON_CBRT_EXACT_DESCRIPTOR 0
#endif

#if NEWTON_CBRT_EXACT_DESCRIPTOR
#define NEWTON_CBRT_TTI_CANDIDATE 1
constexpr uint32_t kNewtonCbrtTtiBodySlots = 23;
constexpr bool kNewtonCbrtTtiReplay = tti_replay_fits(kNewtonCbrtTtiBodySlots);

// One physical row of the public cbrt arithmetic schedule.  Both the replay
// path and NEWTON_TTI_DISABLE issue this exact function; the escape therefore
// changes transport only, not arithmetic, register operands, or rounding.
//
// Two distinct controls, per C-LOW-2 (applies to cbrt and to the N==2 sqrt/
// rsqrt shape above, which shares these gates):
//
//   NEWTON_LOWERING_DISABLE  vetoes selection of the structural Newton route
//                            entirely, recovering the untouched native/generic
//                            baseline -- the A/B control arm.
//   NEWTON_TTI_DISABLE       keeps the selected route and its exact template
//                            arithmetic, and directly issues the rows instead
//                            of recording and replaying them.
//
// These were one define.  That conflation meant the only available escape also
// changed which kernel was selected, so there was no way to A/B the transport
// against identical arithmetic -- which is precisely what C-LOW-2 asks for.
__attribute__((always_inline)) inline void newton_cbrt_tti_row() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
    TTI_SFPNOP;  // replay load-to-consumer gap
    TTI_SFPABS(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 1);  // ax
    TTI_SFPCAST(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        0);  // f = bits*negative_third_256 + magic
    TTI_SFPNOP;  // MAD-to-shift gap
    TTI_SFPSHFT(0x008, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 7);
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);  // y*y
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        0);  // d = ax*y*y
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // c = d*y
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG5,
        0);  // C2*c + C1
    TTI_SFPNOP;  // MAD-to-consumer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG0,
        0);  // t = c*(C2*c+C1)+C0
    TTI_SFPSETSGN(
        0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 0);  // signed d
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // t*t
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG3,
        0);  // y = signed d*(t*t)
    TTI_SFPNOP;  // MUL-to-RNE gap
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_ADV, 0);
}

inline void newton_cbrt_tti_tile() {
    // 2^-8 is the public SFPU seed-coordinate scale and 2^23 is its
    // mantissa extraction bias.  The variable quantities (N and MAGIC) come
    // solely from the NativeRoot graph emitted by S60.
    constexpr float negative_inv_n_scaled =
        (-1.0f / (float)NEWTON_ROOT_N) / 256.0f;
    constexpr float magic =
        ((float)NEWTON_ROOT_MAGIC) / 256.0f + 8388608.0f;
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG1,
        __builtin_bit_cast(uint32_t, negative_inv_n_scaled));
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG2, __builtin_bit_cast(uint32_t, magic));
#if defined(NEWTON_TTI_DISABLE)
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d) {
        newton_cbrt_tti_row();
    }
#else
    TT_REPLAY_RECORD(kNewtonCbrtTtiBodySlots);
    newton_cbrt_tti_row();
    TT_REPLAY_REST(kNewtonCbrtTtiBodySlots, 31);
#endif
    TT_REPLAY_TILE_EPILOGUE();
}
#else
#define NEWTON_CBRT_TTI_CANDIDATE 0
constexpr bool kNewtonCbrtTtiReplay = false;
#endif

#if (NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL)
#if defined(NEWTON_ROOT_ALGORITHM_SQRT_23BIT)
// SQRT_23-bit reciprocal reconstruction. This is the reciprocal member of
// the same metadata-declared fixed-root family as newton_root_sqrt: the seed
// and correction coefficients remain fitter-owned, while the final Newton
// coordinate is y/2 rather than x*y/2.
inline vFloat newton_root_rsqrt(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(vConstIntPrgm0 - i);
    vFloat xy = x * y;
    vFloat c = (-y) * xy;
    y = y * (vConstFloatPrgm1 + c * (vConstFloatPrgm2 + c));
#if NEWTON_ROOT_ITERS == 1
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    return y;
#elif NEWTON_ROOT_ITERS == 2
    xy = x * y;
    vFloat one_minus_xyy = vFloat(1.0f) + ((-y) * xy);
    vFloat half_y = addexp(y, -1);
    vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
    vInt x_bits = as<vInt>(x);
    vInt infinity_minus_x_bits = as<vInt>(infinity) - x_bits;
    v_if((infinity_minus_x_bits != 0) && (x_bits != 0)) {
        y = one_minus_xyy * half_y + y;
    }
    v_else { y = as<vFloat>(infinity_minus_x_bits); }
    v_endif;
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    return y;
#else
#error "SQRT23 reciprocal root requires one or two declared corrections"
#endif
}
#else
// --- rsqrt: classic inverse-sqrt magic seed + Newton (y = y*(1.5 - 0.5*x*y*y))-
// The seed IS 1/sqrt(x) directly, so NO final reciprocal op is needed. Each
// Newton step is one mul + two mads. 2 steps -> <=1 bf16 ULP, 3 -> exact.
inline vFloat newton_root_rsqrt(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(vConstIntPrgm0 - i);  // inverse-sqrt MAGIC seed
    vFloat half_x = addexp(x, -1);                       // 0.5*x (exponent decrement)
#pragma GCC unroll 4
    for (int s = 0; s < NEWTON_ROOT_ITERS; s++) {
        y = y * (vConstFloatPrgm1 - half_x * (y * y));  // prgm1 = 1.5
    }
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    v_if(x == 0.0f) { y = std::numeric_limits<float>::infinity(); }
    v_endif;
    return y;
}
#endif
#endif  // rsqrt variant

#if defined(EVAL_METHOD_ZONE_FACTORED_EXP_ROOT_CORRECTION_2)
// Closed anonymous four-role form: one x^2 core and one compact
// exp/SQRT23-rsqrt/correction tail.  Coefficients and bounds are ordinary
// fitter products; this common one-row body consults no activation identity.
inline vFloat zone_factored_exp_root_correction_2_eval(vFloat x) {
    static_assert(
        TT_EXP_ROOT_EXP_DEGREE == 2u && TT_EXP_ROOT_CORE_DEGREE == 4u &&
            TT_EXP_ROOT_CORRECTION_DEGREE == 1u &&
            TT_EXP_ROOT_CORRECTION_COEFFS[0] == 1.0f,
        "compact exp/root zone descriptor has unsupported leaf shape");

    vFloat magnitude = setsgn(x, 0);
    vFloat square = magnitude * magnitude;
    vFloat core = TT_EXP_ROOT_CORE_COEFFS[TT_EXP_ROOT_CORE_DEGREE];
#pragma GCC unroll 8
    for (int k = (int)TT_EXP_ROOT_CORE_DEGREE - 1; k >= 0; --k) {
        core = core * square + TT_EXP_ROOT_CORE_COEFFS[k];
    }
    if constexpr (TT_EXP_ROOT_ODD) {
        core = magnitude * core;
    }

    // Match the fitter's biased one-FMA exponent state exactly.  The fixed
    // shift moves the large coefficient scale into an exact exponent add.
    constexpr float kLog2E = 1.4426950408889634f;
    constexpr float kBiasedOffset =
        127.0f - TT_EXP_ROOT_EXPONENT_SHIFT * kLog2E;
    vFloat biased = magnitude * kLog2E + kBiasedOffset;
    vInt exponent = exexp(biased);
    vInt mantissa = exman(biased, MantissaMode::ImplicitOne);
    mantissa = shft(mantissa, exponent, ShiftMode::Logical);
    vFloat encoded = as<vFloat>(mantissa);
    vInt integer_part = exexp(encoded, ExponentMode::Biased);
    vMag fractional_magnitude = exman(encoded);
    vFloat fraction =
        convert<vFloat>(fractional_magnitude, RoundMode::Nearest) * 0x1p-23f;

    vFloat exp_polynomial = TT_EXP_ROOT_EXP_COEFFS[TT_EXP_ROOT_EXP_DEGREE];
#pragma GCC unroll 4
    for (int k = (int)TT_EXP_ROOT_EXP_DEGREE - 1; k >= 0; --k) {
        exp_polynomial =
            exp_polynomial * fraction + TT_EXP_ROOT_EXP_COEFFS[k];
    }
    vFloat root = newton_root_rsqrt(magnitude);
    vFloat inverse = root * root;
    vFloat scaled_root = exp_polynomial * root;
    vFloat correction_term = TT_EXP_ROOT_CORRECTION_COEFFS[1] * inverse;
    // c0 is canonically one, so the final scale multiply folds into this FMA.
    vFloat unscaled = correction_term * scaled_root + scaled_root;
    vInt unscaled_exponent = exexp(unscaled, ExponentMode::Biased);
    vFloat result =
        setexp(unscaled, integer_part + unscaled_exponent - 127);

    v_if(magnitude <= TT_EXP_ROOT_CORE_BOUNDARY) { result = core; }
    v_endif;
    v_if(magnitude >= TT_EXP_ROOT_FINITE_TERMINAL) {
        result = std::numeric_limits<float>::infinity();
    }
    v_endif;
    if constexpr (TT_EXP_ROOT_ODD) {
        result = copysgn(result, x);
    }
#if defined(TT_EXP_ROOT_BF16_ORIGIN_FTZ_RNE_CELL)
    // The joint fitter may certify one x*P(x^2) origin cell whose final FP32
    // product is flushed before destination BF16 RNE.  Compare its derived
    // normal-domain image, then build signed FP32 MIN_NORMAL by clearing the
    // original normal input's mantissa.  The certificate is bound to target
    // policy, the ordinary joint product, and exact c0 bytes; no activation
    // identity or raw-word fingerprint participates.
    vFloat origin_ftz_rne_key = addexp(
        magnitude, TT_EXP_ROOT_ORIGIN_FTZ_RNE_CELL_EXPONENT_SHIFT);
    v_if(
        origin_ftz_rne_key ==
        TT_EXP_ROOT_ORIGIN_FTZ_RNE_CELL_SCALED_MAGNITUDE) {
        result = setman(x, 0);
    }
    v_endif;
#endif
    return result;
}
#endif

#if (NEWTON_ROOT_N == 3)
#if defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)
// --- cbrt: native-style Moroz magic seed + correction ----------------------
//
// Metadata declares this algorithm through NEWTON_ROOT_ALGORITHM_CBRT_MAGIC.
// The body mirrors ckernel_sfpu_cbrt.h generically enough for any lowering that
// supplies the same payload: root_n=3, seed magic, seed scale, and correction
// coefficients. It does not inspect the activation name.
#ifndef NEWTON_ROOT_MAGIC_SCALE
#define NEWTON_ROOT_MAGIC_SCALE 256.0f
#endif
#ifndef NEWTON_ROOT_MAGIC_BIAS
#define NEWTON_ROOT_MAGIC_BIAS 8388608.0f
#endif
#ifndef NEWTON_ROOT_NEG_INV_N_SCALED
#define NEWTON_ROOT_NEG_INV_N_SCALED -0x1.555556p-10f
#endif
inline vFloat newton_root_cbrt_magic(vFloat x) {
    vFloat ax = setsgn(x, 0);

    vFloat f = convert<vFloat>(as<vSMag>(ax), RoundMode::Nearest);
    vFloat magic = ((float)NEWTON_ROOT_MAGIC) / NEWTON_ROOT_MAGIC_SCALE + NEWTON_ROOT_MAGIC_BIAS;
    f = f * NEWTON_ROOT_NEG_INV_N_SCALED + magic;

    vFloat y = as<vFloat>(as<vInt>(f) << 8);

#ifdef USE_BF16
    vFloat d = ax * (y * y);
    vFloat c = d * y;
    vFloat t = c * (vConstFloatPrgm2 * c + vConstFloatPrgm1) + vConstFloatPrgm0;
    d = copysgn(d, x);
    y = d * (t * t);
#else
    vFloat c = (ax * y) * (y * y);
    y = y * (c * (vConstFloatPrgm2 * c + vConstFloatPrgm1) + vConstFloatPrgm0);

    vFloat d = ax * (y * y);
    c = d * y + vFloat(-1.0f);
    vFloat negative_third = addexp(vFloat(NEWTON_ROOT_NEG_INV_N_SCALED), 8);
    vFloat t = c * negative_third + vFloat(1.0f);
    d = copysgn(d, x);
    y = d * (t * t);
#endif
    return y;
}
#else
// --- cbrt: minimal exponent seed + DIVISION-FREE cubic Householder ---------
// Odd function: work on |x|, restore sign. The SFPU has no 32-bit float->int (so
// the classic bits/3+magic seed isn't expressible) AND its reciprocal LLK is
// register-heavy (inlining it per-element overflows the SFPU register file ->
// trisc "maximum reload insns" ICE). So we iterate the INVERSE cube root
// w -> |x|^(-1/3) with a DIVISION-FREE cubic Householder step:
//     c = 1 - |x|*w^3 ;  w <- w * (1 + c/3 + 2c^2/9)
// then cbrt(|x|) = |x| * w^2. Seed w0 = 2^(-round(e/3)) * (-0.27*m + 1.25), a
// linear m^(-1/3) approx. The exponent-parity is NOT folded in (keeps the body
// register-light); 3 cubic steps recover <=1 bf16 ULP (validated).
inline vFloat newton_root_cbrt(vFloat x) {
    vInt sign_bits = as<vInt>(x) & (vInt)0x80000000;
    vFloat ax = setsgn(x, 0);  // work on |x|; cbrt is odd, sign restored at end

    vInt e_int = exexp(ax, ExponentMode::Biased) - 127;
    vFloat m = setexp(ax, 127);

    // q = round(e/3) via the 0x4B400000 float magic (e small -> exact in float).
    // Round-to-nearest (NOT floor+parity): keeps the body register-light enough
    // to avoid the trisc "maximum reload insns" ICE. The missing exponent parity
    // is absorbed by one extra cubic Householder step.
    const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat ef = convert<vFloat>(as<vSMag>(e_int), RoundMode::Nearest);
    v_if(e_int < 0) { ef = -convert<vFloat>(as<vSMag>(~e_int + 1), RoundMode::Nearest); }
    v_endif;
    vInt q = as<vInt>(ef * (1.0f / 3.0f) + magic) - as<vInt>(magic);

    // w0 = 2^(-q) * (-0.27*m + 1.25): linear m^(-1/3) seed, exponent shifted -q.
    vFloat wm = -0.27f * m + 1.25f;
    vFloat w = setexp(wm, exexp(wm, ExponentMode::Biased) - q);

    // Division-free cubic Householder on the inverse cube root.
    const vFloat a13 = 1.0f / 3.0f;
    const vFloat a29 = 2.0f / 9.0f;
    for (int s = 0; s < NEWTON_ROOT_ITERS; s++) {
        vFloat c = 1.0f - ax * (w * w * w);
        w = w * (1.0f + c * a13 + (c * c) * a29);
    }
    vFloat y = ax * w * w;  // cbrt(|x|) = |x| * w^2
    // restore sign (cbrt is odd); cbrt(0)=0 handled explicitly.
    y = as<vFloat>(as<vInt>(y) | sign_bits);
    v_if(x == 0.0f) { y = 0.0f; }
    v_endif;
    return y;
}
#endif
#endif  // cbrt variant

template <uint32_t DEG>
inline vFloat newton_root_eval(vFloat x) {
    (void)DEG;  // degree is irrelevant for the Newton path (kept for caller symmetry)
#if (NEWTON_ROOT_N == 3)
#if defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)
    return newton_root_cbrt_magic(x);
#else
    return newton_root_cbrt(x);
#endif
#elif defined(NEWTON_ROOT_RECIPROCAL)
    return newton_root_rsqrt(x);
#else
    return newton_root_sqrt(x);
#endif
}

#if defined(TT_NEWTON_TYPED_RAW_TERMINAL_PREPASS)
// Compose a typed encoded-raw terminal without editing either admitted Newton
// numeric body.  Exactly one encoded lane and one FP32 replacement are live.
inline void newton_typed_raw_terminal_prepass() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        vFloat terminal_or_input = dst_reg[d];
        finalize_encoded_raw_domain_actions(raw_u16, terminal_or_input);
        dst_reg[d] = terminal_or_input;
    }
}
#endif
#endif

#if defined(EVAL_METHOD_ASIN_ACOS)
#if defined(EVAL_METHOD_HALF_ANGLE_RATIO_1)
// Ordinary frontier descriptor aliases.  These keep the already measured
// common SFPI body byte-shaped while removing any dependency on a Program
// manifest/include from the selected execution path.
#define TT_PROGRAM_HALF_ANGLE_LIMIT HALF_ANGLE_LIMIT
#define TT_PROGRAM_HALF_ANGLE_SPLIT HALF_ANGLE_SPLIT
#define TT_PROGRAM_HALF_ANGLE_ONE HALF_ANGLE_ONE
#define TT_PROGRAM_HALF_ANGLE_COORD_SCALE HALF_ANGLE_COORD_SCALE
#define TT_PROGRAM_HALF_ANGLE_RESULT_SCALE HALF_ANGLE_RESULT_SCALE
#define TT_PROGRAM_HALF_ANGLE_RESULT_BIAS HALF_ANGLE_RESULT_BIAS
#define TT_PROGRAM_HALF_ANGLE_RATIO_DEGREE HALF_ANGLE_RATIO_DEGREE
#define TT_PROGRAM_HALF_ANGLE_SQRT_MAGIC HALF_ANGLE_SQRT_MAGIC
#define TT_PROGRAM_HALF_ANGLE_SQRT_C0 HALF_ANGLE_SQRT_C0
#define TT_PROGRAM_HALF_ANGLE_SQRT_C1 HALF_ANGLE_SQRT_C1
#define TT_PROGRAM_HALF_ANGLE_SQRT_C2 HALF_ANGLE_SQRT_C2
#define TT_PROGRAM_HALF_ANGLE_ROOT_RECIPROCAL HALF_ANGLE_ROOT_RECIPROCAL
#define TT_PROGRAM_HALF_ANGLE_RECONSTRUCT_FROM_VALUE HALF_ANGLE_RECONSTRUCT_FROM_VALUE
#if defined(HALF_ANGLE_POSTCOMPOSE)
#define TT_PROGRAM_HALF_ANGLE_POSTCOMPOSE 1
#define TT_PROGRAM_HALF_ANGLE_OUTPUT_SCALE HALF_ANGLE_OUTPUT_SCALE
#define TT_PROGRAM_HALF_ANGLE_OUTPUT_BIAS HALF_ANGLE_OUTPUT_BIAS
#endif
#endif
#if defined(EVAL_METHOD_HALF_ANGLE_RATIO_1)
inline vFloat asin_acos_sqrt(vFloat x) {
#if TT_PROGRAM_HALF_ANGLE_ROOT_RECIPROCAL
#if defined(TT_HALF_ANGLE_ROOT_PRGM_PRELOAD)
    // Same shared two-step target recurrence, with its S55-owned constants
    // programmed once per kernel instead of materialized for every lane.
    vFloat out = x;
    v_if(x != 0.0f) {
        vInt half_bits = as<vInt>(as<vUInt>(x) >> 1);
        vFloat approx = as<vFloat>(vConstIntPrgm0 - half_bits);
        vFloat negative_half_x = x * vConstFloatPrgm1;
        approx = ((approx * approx) * negative_half_x + vConstFloatPrgm2) * approx;
        approx = ((approx * approx) * negative_half_x + vConstFloatPrgm2) * approx;
        out = approx * x;
    }
    v_endif;
    return out;
#else
    return ckernel::sfpu::sfpu_sqrt_custom<false>(x);
#endif
#else
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(
        vInt(TT_PROGRAM_HALF_ANGLE_SQRT_MAGIC) - i);

    vFloat xy = x * y;
    vFloat c = -y * xy;
    y = y * (
        TT_PROGRAM_HALF_ANGLE_SQRT_C1 +
        c * (TT_PROGRAM_HALF_ANGLE_SQRT_C2 + c));
    xy = x * y;
    vFloat one_minus_xyy = vFloat(1.0f) + (-y * xy);
    // ``addexp(0, -1)`` wraps the zero exponent field on SFPU.  Use the
    // zero-safe exact power-of-two multiply already shared by the
    // sqrt-factored Newton correction so closed endpoint coordinates remain
    // zero without an endpoint-specific predicate.
    vFloat half_xy = xy * 0.5f;
    vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
    v_if(as<vInt>(x) < as<vInt>(infinity)) {
        y = one_minus_xyy * half_xy + xy;
    }
    v_endif;
    return y;
#endif
}

template <uint32_t DEGREE, uint32_t LUT_SIZE>
inline vFloat asin_acos_ratio_poly(
    vFloat coordinate, const std::array<float, LUT_SIZE>& lut) {
    static_assert(NUM_SEGMENTS == 1, "native half-angle ratio requires one LUT segment");
    static_assert(
        DEGREE == TT_PROGRAM_HALF_ANGLE_RATIO_DEGREE,
        "native half-angle ratio descriptor/LUT degree mismatch");
    static_assert(
        LUT_SIZE == 2 + DEGREE + 1,
        "native half-angle ratio descriptor/LUT shape mismatch");
    constexpr uint32_t COEFF_OFFSET = 2;
    vFloat ratio = lut[COEFF_OFFSET + DEGREE];
#pragma GCC unroll 12
    for (int k = DEGREE - 1; k >= 0; --k) {
        ratio = __builtin_rvtt_sfpmad(
            ratio.get(), coordinate.get(), vFloat(lut[COEFF_OFFSET + k]).get(),
            SFPMAD_MOD1_OFFSET_NONE);
    }
    return ratio;
}

template <uint32_t DEGREE, uint32_t LUT_SIZE>
inline vFloat asin_acos_eval(
    vFloat x, const std::array<float, LUT_SIZE>& lut) {
    vFloat ax = setsgn(x, 0);
    // Evaluate the arithmetic body without an outer condition frame.  The
    // root recurrence has its own target guard, and nesting it under a false
    // valid-domain predicate is not a verifier-safe Blackhole CC schedule.
    // Invalid lanes are overwritten after all arithmetic temporaries retire.
    vFloat coordinate =
        (TT_PROGRAM_HALF_ANGLE_ONE - ax) *
        TT_PROGRAM_HALF_ANGLE_COORD_SCALE;
    vFloat root = asin_acos_sqrt(coordinate);

    // Both zones consume the same verified ratio leaf.  Select its coordinate
    // and multiplicative root first, then issue one shared polynomial body;
    // duplicating that body in two SFPU predicates records both copies.
    v_if(ax < TT_PROGRAM_HALF_ANGLE_SPLIT) {
        coordinate = ax * ax;
        root = ax;
    }
    v_endif;

    vFloat ratio = asin_acos_ratio_poly<DEGREE, LUT_SIZE>(coordinate, lut);
    vFloat unsigned_result = root * ratio;
    v_if(ax >= TT_PROGRAM_HALF_ANGLE_SPLIT) {
#if TT_PROGRAM_HALF_ANGLE_RECONSTRUCT_FROM_VALUE
        unsigned_result = __builtin_rvtt_sfpmad(
            unsigned_result.get(),
            vFloat(TT_PROGRAM_HALF_ANGLE_RESULT_SCALE).get(),
            vFloat(TT_PROGRAM_HALF_ANGLE_RESULT_BIAS).get(),
            SFPMAD_MOD1_OFFSET_NONE);
#else
        vFloat scaled_ratio = TT_PROGRAM_HALF_ANGLE_RESULT_SCALE * ratio;
        unsigned_result = __builtin_rvtt_sfpmad(
            root.get(), scaled_ratio.get(),
            vFloat(TT_PROGRAM_HALF_ANGLE_RESULT_BIAS).get(),
            SFPMAD_MOD1_OFFSET_NONE);
#endif
    }
    v_endif;
    vFloat signed_result = copysgn(unsigned_result, x);
#if defined(TT_PROGRAM_HALF_ANGLE_POSTCOMPOSE)
    signed_result = __builtin_rvtt_sfpmad(
        signed_result.get(),
        vFloat(TT_PROGRAM_HALF_ANGLE_OUTPUT_SCALE).get(),
        vFloat(TT_PROGRAM_HALF_ANGLE_OUTPUT_BIAS).get(),
        SFPMAD_MOD1_OFFSET_NONE);
#endif
    v_if(ax > TT_PROGRAM_HALF_ANGLE_LIMIT) {
        static_assert(
            TT_HALF_ANGLE_INVALID_RESULT <= 4u,
            "half-angle invalid exterior requires a class result");
        if constexpr (TT_HALF_ANGLE_INVALID_RESULT == 0u) {
            signed_result = std::numeric_limits<float>::quiet_NaN();
        } else if constexpr (TT_HALF_ANGLE_INVALID_RESULT == 1u) {
            signed_result = std::numeric_limits<float>::infinity();
        } else if constexpr (TT_HALF_ANGLE_INVALID_RESULT == 2u) {
            signed_result = -std::numeric_limits<float>::infinity();
        } else if constexpr (TT_HALF_ANGLE_INVALID_RESULT == 3u) {
            signed_result = vFloat(0.0f);
        } else {
            signed_result = setsgn(vFloat(0.0f), 1);
        }
    }
    v_endif;
    return signed_result;
}
#else
inline vFloat asin_acos_sqrt(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(0x5f1110a0 - i);

    vFloat xy = x * y;
    vFloat c = -y * xy;
    y = y * (2.2825186f + c * (2.2533049f + c));
    xy = x * y;
    vFloat one_minus_xyy = vFloat(1.0f) + (-y * xy);
    vFloat half_xy = addexp(xy, -1);
    vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
    v_if(as<vInt>(x) < as<vInt>(infinity)) { y = one_minus_xyy * half_xy + xy; }
    v_endif;
    return y;
}

inline vFloat asin_acos_ratio_poly(vFloat z) {
    vFloat u = z * z;
    vFloat ratio = ASIN_ACOS_COEFFS[ASIN_ACOS_DEGREE];
#pragma GCC unroll 8
    for (int k = ASIN_ACOS_DEGREE - 1; k >= 0; --k) {
        ratio = ratio * u + ASIN_ACOS_COEFFS[k];
    }
    return z * ratio;
}

inline vFloat asin_acos_eval(vFloat x) {
    vFloat ax = setsgn(x, 0);
    vFloat asin_abs = ASIN_ACOS_PI_2;

    v_if(ax <= ASIN_ACOS_DIRECT_THRESHOLD) { asin_abs = asin_acos_ratio_poly(ax); }
    v_else {
        vFloat t = (vFloat(1.0f) - ax) * 0.5f;
        vFloat root = asin_acos_sqrt(t);
        asin_abs = ASIN_ACOS_PI_2 - 2.0f * asin_acos_ratio_poly(root);
    }
    v_endif;

    vFloat asin_signed = copysgn(asin_abs, x);
#if defined(ASIN_ACOS_OP_ACOS)
    return ASIN_ACOS_PI_2 - asin_signed;
#else
    return asin_signed;
#endif
}
#endif
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
// Native-Blackhole magic-seed + 1 Newton square root (same algorithm as
// asin_acos_sqrt / ckernel_sfpu_sqrt.h). Used by the sqrt_factored basis to
// reconstruct acos(x) = sqrt(1-|x|) * P(|x|). File-scope so it compiles
// independently of the EVAL_METHOD_ASIN_ACOS block above.
inline vFloat basis_sqrt(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(0x5f1110a0 - i);
    vFloat xy = x * y;
    vFloat c = -y * xy;
    y = y * (2.2825186f + c * (2.2533049f + c));
    xy = x * y;
    vFloat one_minus_xyy = vFloat(1.0f) + (-y * xy);
    // ``addexp(0, -1)`` wraps the zero exponent field on SFPU and produces
    // infinity.  The exact power-of-two multiply is zero-safe and preserves
    // the same value for every positive coordinate admitted by this basis.
    vFloat half_xy = xy * 0.5f;
    y = one_minus_xyy * half_xy + xy;
    return y;
}
#endif

#if defined(EXPONENT_ALU_LOG1P) || \
    (defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD) && defined(POLY_TTI_DISABLE))
// log1p(x) = log(1+x) via EXACT-reconstruction range reduction (Norbert Juffa's
// method, ported to our exponent-ALU dialect). The naive log2 path (form u=1+x,
// decompose the ROUNDED u) rounds away the low bits of small x -> catastrophic
// pure ULP near 0 (the reverted 598383e5f regression). Here u=1+x is formed ONLY
// to read the exponent k; the reduced argument is reconstructed from the ORIGINAL
// x so no low bits are lost:
//     r = 2^-k * x  +  (2^-k - 1)  =  2^-k*(1+x) - 1  in [0,1)
// with  t = 2^-k - 1  formed SEPARATELY (exact pow2-1) so the k=0 case is r = x+0
// = x exactly, never (x+1)-1. Then
//     log1p(x) = k*ln2 + [ r + r^2 * P(r) ],   P(r) = (log1p(r)-r)/r^2
// The explicit linear term r keeps the near-0 relative error controlled (a single
// wide poly/rational cannot); the fitter supplies P in the natural [0,1) basis.
#ifndef LOG1P_LN2
#define LOG1P_LN2 0.69314718055994530942f
#endif
#if defined(EXPONENT_ALU_LOG1P)
template <uint32_t DEG>
inline vFloat log1p_hw_eval(vFloat x) {
    vFloat u = x + 1.0f;                                    // form 1+x only for its exponent

    // Juffa exponent isolation on the [0.75, 1.5) window (NOT the [1,2) window):
    // pick k so 2^-k*u in [0.75,1.5) -> reduced arg in [-0.25,0.5). Crucially this
    // keeps small |x| (u near 1) at k=0, so log1p(x) needs no k*ln2 term there and
    // avoids the -ln2 + log1p(r) cancellation the [1,2) window suffers for x<0.
    // e = k<<23: subtract the encoding of 0.75, then zero the mantissa bits.
    vInt e = as<vInt>(vFloat(0.75f));
    e = as<vInt>(u) - e;
    e = as<vInt>(setman(as<vFloat>(e), 0));

    // Reconstruct the reduced argument from the ORIGINAL x (no 1+x cancellation):
    //   2^-k * x        = reinterpret(bits(x) - e)   (integer subtract == 2^-k scale)
    //   2^-k - 1        = -0.25 * (-4 * 2^-k) - 1
    vFloat two_neg_k_x = as<vFloat>(as<vInt>(x) - e);
    vFloat s = as<vFloat>(as<vInt>(vFloat(-4.0f)) - e);    // -4 * 2^-k
    vFloat t = -0.25f * s - 1.0f;                          // 2^-k - 1  (exact)
    vFloat r = two_neg_k_x + t;                            // 2^-k*(1+x) - 1 in [-0.25,0.5)

    // log1p(r) = r + r^2 * P(r); P = (log1p(r)-r)/r^2, fitter's natural basis Horner.
    const float* c = LOG1P_HW_COEFFS;
    vFloat p = c[DEG];
#pragma GCC unroll 16
    for (int i = (int)DEG - 1; i >= 0; i--) {
        p = p * r + c[i];
    }
    vFloat result = r + (r * r) * p;

    // + k*ln2.  e is k<<23 (exponent-bit units); convert(|e|) = |k|*2^23, restore
    // the sign from e, scale by ln2*2^-23 to recover k*ln2.
    vFloat e_float = convert<vFloat>(abs(e), RoundMode::Nearest);
    e_float = copysgn(e_float, as<vFloat>(e));
    result = e_float * (LOG1P_LN2 * 0x1p-23f) + result;

    // Specials on the reduction input (1+x): log1p(-1) = -inf, x < -1 -> NaN.
#ifndef LOG1P_HW_SKIP_SPECIALS
    v_if(u < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(u == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;
#endif
    return result;
}
#endif

#if defined(HW_PRELOAD) || \
    (defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD) && defined(POLY_TTI_DISABLE))
// GENERIC constant-pool preload variant of log1p_hw_eval (Juffa exact
// reconstruction). The non-preloaded evaluator re-materializes FOUR fp32
// constants inside the per-element loop (2 SFPLOADI each: c[2], c[1], c[0],
// LN2*2^-23 -> 8 in-body instructions per element at degree 2). Ranking:
//   prgm0 = LOG1P_LN2 * 2^-23 (k*ln2 recombine multiplier, touched every elem)
//   prgm1 = c[DEG], prgm2 = c[DEG-1]  (top-2 Horner coeffs)
//   cva/cvb/cvc = c[DEG-2..DEG-4]     (caller-pinned LREGs, hw_preload_pin)
//   cvspill     = any deeper tail     (in-body re-materialize, the logged spill)
// The bf16-immediate constants (0.75, -4.0, -0.25) stay as 1-op immediates:
// their consumers are destructive 2-operand SFPIADD/SFPMULI, so an LREG copy
// would cost the same one op (SFPMOV) per element.
// Math is byte-identical to log1p_hw_eval (same op kinds, same order).
template <uint32_t DEG>
inline vFloat log1p_hw_eval_preloaded(vFloat x, vFloat cva, vFloat cvb, vFloat cvc, const float* cvspill) {
    vFloat u = x + 1.0f;  // form 1+x only for its exponent

    // Juffa exponent isolation on the [0.75, 1.5) window (see log1p_hw_eval).
    vInt e = as<vInt>(vFloat(0.75f));
    e = as<vInt>(u) - e;
    e = as<vInt>(setman(as<vFloat>(e), 0));

    // Reconstruct the reduced argument from the ORIGINAL x (no 1+x cancellation).
    vFloat two_neg_k_x = as<vFloat>(as<vInt>(x) - e);
    vFloat s = as<vFloat>(as<vInt>(vFloat(-4.0f)) - e);  // -4 * 2^-k
    vFloat t = -0.25f * s - 1.0f;                        // 2^-k - 1  (exact)
    vFloat r = two_neg_k_x + t;                          // 2^-k*(1+x) - 1 in [-0.25,0.5)

    // log1p(r) = r + r^2 * P(r); Horner over prgm-resident + pinned coeffs.
    vFloat p;
    if constexpr (DEG >= 1) {
        p = vConstFloatPrgm1 * r + vConstFloatPrgm2;  // c[DEG]*r + c[DEG-1]
        if constexpr (DEG >= 2) {
            p = p * r + cva;  // + c[DEG-2]
        }
        if constexpr (DEG >= 3) {
            p = p * r + cvb;  // + c[DEG-3]
        }
        if constexpr (DEG >= 4) {
            p = p * r + cvc;  // + c[DEG-4]
        }
#pragma GCC unroll 16
        for (int i = (int)DEG - 5; i >= 0; i--) {
            // Spill tail (DEG>4): cvspill index walks c[DEG-5] ... c[0].
            p = p * r + cvspill[(int)DEG - 2 - i];
        }
    } else {
        p = vConstFloatPrgm1;  // degree-0: c[0]
    }
    vFloat result = r + (r * r) * p;

    // + k*ln2: prgm0 holds LN2*2^-23 (same constant the non-preloaded path
    // re-materializes as an in-body SFPLOADI pair every element).
    vFloat e_float = convert<vFloat>(abs(e), RoundMode::Nearest);
    e_float = copysgn(e_float, as<vFloat>(e));
    result = e_float * vConstFloatPrgm0 + result;

    // Specials on the reduction input (1+x): log1p(-1) = -inf, x < -1 -> NaN.
#if !defined(LOG1P_HW_SKIP_SPECIALS) && !defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD)
    v_if(u < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(u == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;
#endif
    return result;
}
#endif  // HW_PRELOAD

// ----------------------------------------------------------------------------
// TTI REPLAY fast path for log1p (Juffa exact-reconstruction reduction).
// Same replay mechanism as kExpHwTtiReplay; the base single-element P2 body is
// 29 slots and the class-aware normalized form is 31 (a 2x interleave cannot
// fit 32), with explicit gaps after every MAD-family
// producer (see the trig replay comment for the gap discipline). The body is
// a 1:1 transcription of the sfpi/GCC BH loop of log1p_hw_eval_preloaded
// degree-1/2 (objdump of log1p_p2_s1, with the leading Horner rung deleted
// for P1): identical op kinds, order and operand
// roles (SFPIADD 2s-comp reductions, exact -0.25*s SFPMULI, SFPABS/SFPCAST/
// SFPSETSGN k*ln2 recombine) — bit-identical by construction. Requires
// LOG1P_HW_SKIP_SPECIALS (no CC specials in the recorded body; the selected
// config defines it). LOG1P_TTI_DISABLE=defined forces the sfpi path.
#if defined(HW_PRELOAD) && defined(LOG1P_HW_SKIP_SPECIALS) && defined(USE_BF16) && \
    (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && !defined(LOG1P_TTI_DISABLE) && \
    !defined(TT_DOMAIN_ACTION_PROGRAM)
#define LOG1P_TTI_CANDIDATE 1
constexpr bool kLog1pTtiReplay =
    (LOG1P_HW_DEGREE == 1) || (LOG1P_HW_DEGREE == 2);
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
static_assert(
    TT_NORMALIZED_LOG1P_DEGREE == LOG1P_HW_DEGREE &&
        TT_NORMALIZED_LOG1P_SEGMENTS == 1u &&
        TT_NORMALIZED_LOG1P_LUT_SIZE == TT_NORMALIZED_LOG1P_DEGREE + 1u &&
        TT_NORMALIZED_LOG1P_C0_INDEX == 0u &&
        TT_NORMALIZED_LOG1P_C1_INDEX == 1u &&
        TT_NORMALIZED_LOG1P_C2_INDEX == 2u &&
        TT_NORMALIZED_LOG1P_BODY_SLOTS <= 32u &&
        TT_NORMALIZED_LOG1P_SUFFIX_SLOTS <= 2u &&
        TT_NORMALIZED_LOG1P_PEAK_LIVE <= 8u,
    "normalized residual graph exceeds replay/resource closure");
static_assert(
    TT_NORMALIZED_LOG1P_INVALID_PATTERNS > 0u &&
        TT_NORMALIZED_LOG1P_SPECIAL_PATTERNS > 0u,
    "normalized residual class certificate is empty");
#endif
#if defined(TT_INLINE_PROGRAM_OVERLAY)
static_assert(
    !kLog1pTtiReplay,
    "typed core overlay requires route-aware LOG1P_TTI replay; refusing silent fallback");
#endif
#else
#define LOG1P_TTI_CANDIDATE 0
constexpr bool kLog1pTtiReplay = false;
#endif
#endif

#if defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD) && defined(POLY_TTI_DISABLE)
// Baseline for the typed composition, not the selected outer leaf on raw x.
// The SFPI compiler owns scheduling here; no replay instructions are issued.
inline void poly_log_square_factorized_odd_sfpi_tile() {
    static_assert(TT_SELECTED_LOG_SQUARE_TERMINAL_BOUND_BITS == 0x3f800000u);
    static_assert(TT_SELECTED_LOG_SQUARE_OUTER_DEGREE == 3u ||
                  TT_SELECTED_LOG_SQUARE_OUTER_DEGREE == 4u);
    const float c0 = __builtin_bit_cast(float, TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[0]);
    const float bound = __builtin_bit_cast(float, TT_SELECTED_LOG_SQUARE_TERMINAL_BOUND_BITS);
    for (int d = 0; d < 32; ++d) {
        vFloat x = dst_reg[d];
        vFloat square = x * x;
        vFloat negative_square = -square;
        vFloat coordinate = setsgn(log1p_hw_eval_preloaded<2>(
            negative_square, vFloat(c0), vFloat(0.0f), vFloat(0.0f), nullptr), 0);
        vFloat outer = __builtin_bit_cast(float,
            TT_SELECTED_LOG_SQUARE_OUTER_COEFF_BITS[TT_SELECTED_LOG_SQUARE_OUTER_DEGREE]);
#pragma GCC unroll 4
        for (int i = (int)TT_SELECTED_LOG_SQUARE_OUTER_DEGREE - 1; i >= 0; --i) {
            outer = outer * coordinate + __builtin_bit_cast(float, TT_SELECTED_LOG_SQUARE_OUTER_COEFF_BITS[i]);
        }
        vFloat twice_x = x + x;
        vFloat unscaled = twice_x * outer;
        // The Program's SCALE_POW2(-1) retains the same post-round BF16
        // MIN_NORMAL boundary as the selected TTI suffix, including zero.
        vInt exponent = exexp(unscaled, ExponentMode::Biased);
        vFloat result = setexp(unscaled, exponent - 1);
        v_if(exponent <= 1) {
            result = copysgn(vFloat(0.0f), unscaled);
            v_if(exponent == 1) {
                vFloat normalized = setexp(setsgn(unscaled, 0), 127);
                v_if(normalized >= 1.9921875f) {
                    result = copysgn(vFloat(std::numeric_limits<float>::min()), unscaled);
                }
                v_endif;
            }
            v_endif;
        }
        v_endif;
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        // Install exact declared terminals after the sole narrowing, just as
        // the replay suffix does. The original decoded input owns the sign.
        vFloat magnitude = setsgn(x, 0);
#if defined(TT_SELECTED_LOG_SQUARE_EXTERIOR_SIGNED_INF)
        vFloat terminal = copysgn(vFloat(std::numeric_limits<float>::infinity()), x);
#elif defined(TT_SELECTED_LOG_SQUARE_EXTERIOR_NAN)
        vFloat terminal = std::numeric_limits<float>::quiet_NaN();
#else
#error "log-square baseline requires a typed exterior action"
#endif
        v_if(magnitude < bound) { terminal = result; }
        v_elseif(magnitude == bound) {
            terminal = copysgn(vFloat(std::numeric_limits<float>::infinity()), x);
        }
        v_endif;
        v_if(x == 0.0f) {
#if defined(TT_SELECTED_LOG_SQUARE_WH_DECODED_TERMINALS)
            terminal = 0.0f;
#else
            terminal = copysgn(vFloat(0.0f), x);
#endif
        }
        v_endif;
        dst_reg[d] = terminal;
    }
}
#endif

#if defined(EVAL_METHOD_ROOT_LOG1P_HANDOFF_1)
static_assert(TT_ROOT_LOG1P_PRODUCER_SLOTS == 24u);
static_assert(TT_ROOT_LOG1P_CONSUMER_SLOTS == 29u);
static_assert(TT_ROOT_LOG1P_SUFFIX_SLOTS == 8u);
static_assert(TT_ROOT_LOG1P_TOTAL_ISSUES == 61u);
static_assert(TT_ROOT_LOG1P_TOTAL_ISSUES < TT_ROOT_LOG1P_HARD_CEILING);
static_assert(TT_ROOT_LOG1P_PEAK_LIVE <= 8u);
static_assert(TT_ROOT_LOG1P_ROWS == 32u);
static_assert(TT_ROOT_LOG1P_DST_ROW_INCREMENT == 2u);
static_assert(TT_ROOT_LOG1P_DST_HALF_ROWS == 128u);
static_assert(TT_ROOT_LOG1P_SCRATCH_OFFSET == 64u);
static_assert(
    TT_ROOT_LOG1P_SCRATCH_OFFSET +
        (TT_ROOT_LOG1P_ROWS - 1u) * TT_ROOT_LOG1P_DST_ROW_INCREMENT <
    TT_ROOT_LOG1P_DST_HALF_ROWS);
static_assert(TT_ROOT_LOG1P_SPECIAL_BIASED_EXPONENT == 255u);
static_assert(TT_ROOT_LOG1P_VALID_PATTERNS == 16384u);
static_assert(TT_ROOT_LOG1P_DECODED_EXPFF_PATTERNS == 256u);
static_assert(TT_ROOT_LOG1P_POS_SPECIAL_PATTERNS == 128u);
static_assert(
    TT_ROOT_LOG1P_VALID_PATTERNS + TT_ROOT_LOG1P_INVALID_PATTERNS +
        TT_ROOT_LOG1P_POS_SPECIAL_PATTERNS ==
    65536u);
#endif

#if defined(EXPONENT_ALU_POW)
// pow path for sqrt/rsqrt/cbrt. For root order N (POW_HW_ROOT_N) and x = 2^e * m
// with m in [1,2): root_N(x) = 2^(e/N) * root_N(2^r) * root_N(m), where
// e = N*q + r, r in {0..N-1}. The fitter fits p(m)=root_N(m) on [1,2)
// (NATURAL [1,2]-basis coeffs POW_HW_COEFFS); the kernel splits e into q (folded
// into the exponent via setexp) and parity remainder r (selects the scale
// constant POW_HW_SCALE_C{r} = root_N(2^r)), evaluates the poly, recombines.
//
// Odd roots (cbrt, N=3) are odd functions: cbrt(-x) = -cbrt(x). The kernel
// strips the sign (works on |x|) and restores it on the result. Even roots
// (sqrt, N=2) of negatives are NaN.
//
// expalu_reciprocal (POW_HW_RECIPROCAL) folds the final 1/result for rsqrt
// (= 1/sqrt(x)): one Newton-Raphson reciprocal applied AFTER recombine.
//
// Backward-compatible defaults: N=2, scales = {1, sqrt(2)}.
#ifndef POW_HW_ROOT_N
#define POW_HW_ROOT_N 2
#endif
#ifndef POW_HW_SCALE_C0
#define POW_HW_SCALE_C0 1.0f
#endif
#ifndef POW_HW_SCALE_C1
#define POW_HW_SCALE_C1 1.4142135623730951f  // sqrt(2)
#endif
// C2 only consulted when ROOT_N >= 3 (cbrt). Default cbrt(4) is harmless otherwise.
#ifndef POW_HW_SCALE_C2
#define POW_HW_SCALE_C2 1.5874010519681994f  // cbrt(4)
#endif
template <uint32_t DEG>
inline vFloat pow_hw_eval(vFloat x) {
    constexpr int ROOT_N = POW_HW_ROOT_N;
    constexpr bool ODD_ROOT = (ROOT_N % 2 == 1);

    // Odd roots act on |x|; sign restored at the end. Even roots: neg -> NaN.
    vInt sign_bits = as<vInt>(x) & (vInt)0x80000000;
    vFloat ax = ODD_ROOT ? setsgn(x, 0) : x;

    // Extract biased exponent e and mantissa m in [1,2).
    vInt biased = exexp(ax, ExponentMode::Biased);
    vInt e_int = biased - 127;
    vFloat m = setexp(ax, 127);

    // Horner p(m)=root_N(m) in the natural [1,2] basis.
    const float* c = POW_HW_COEFFS;
    vFloat s = c[DEG];
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        s = s * m + c[k];
    }

    // e = N*q + r, r in {0..N-1}. Compute q = floor(e/N) and r via float arith
    // (e is small, exact in float), matching the cbrt software path's structure.
    const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat ef = convert<vFloat>(as<vSMag>(e_int), RoundMode::Nearest);
    v_if(e_int < 0) {
        vInt mag = ~e_int + 1;
        ef = -convert<vFloat>(as<vSMag>(mag), RoundMode::Nearest);
    }
    v_endif;

    vInt q;
    vInt r;
    if constexpr (ROOT_N == 2) {
        r = e_int & 1;  // two's-complement low bit is parity for both signs
        vFloat qf = (ef - convert<vFloat>(as<vSMag>(r), RoundMode::Nearest)) * 0.5f;
        q = as<vInt>(qf + magic) - as<vInt>(magic);
    } else {
        // General floor-divide by N via round-to-nearest of e/N then correct r.
        constexpr float INV_N = 1.0f / (float)ROOT_N;
        vFloat q_approx = ef * INV_N;
        vFloat q_rounded = q_approx + magic;
        q = as<vInt>(q_rounded) - as<vInt>(magic);
        vFloat q_back = q_rounded - magic;
        // r = e - N*q (in float, exact), then to int via magic.
        vFloat nq = q_back;
#pragma GCC unroll 8
        for (int i = 1; i < ROOT_N; i++) {
            nq = nq + q_back;
        }  // nq = N*q_back
        vFloat r_float = ef - nq;
        r = as<vInt>(r_float + magic) - as<vInt>(magic);
        // Normalize r into [0, N-1], adjusting q accordingly.
        v_if(r < 0) {
            q = q - 1;
            r = r + ROOT_N;
        }
        v_endif;
        v_if(r >= ROOT_N) {
            q = q + 1;
            r = r - ROOT_N;
        }
        v_endif;
    }

    // Apply scale root_N(2^r) for the parity remainder r.
    v_if(r == 1) { s = s * POW_HW_SCALE_C1; }
    v_endif;
    if constexpr (ROOT_N >= 3) {
        v_if(r == 2) { s = s * POW_HW_SCALE_C2; }
        v_endif;
    }

    // Multiply by 2^q via exponent addition.
    vInt s_exp = exexp(s, ExponentMode::Biased);
    vFloat result = setexp(s, s_exp + q);

#if defined(POW_HW_RECIPROCAL)
    // rsqrt etc.: final 1/result. Newton-Raphson reciprocal, 3 iters. NOT
    // init-free: sfpu_reciprocal_iter reads Prgm0 on BH (2.0) and Prgm0/1/2 on
    // WH (Sollya seeds) — sfpu_reciprocal_init runs in the preamble (see the
    // PRGM CONSTANT OWNERSHIP RULE there).
    result = ckernel::sfpu::sfpu_reciprocal_iter<3>(result);
#endif

    if constexpr (ODD_ROOT) {
        // Restore original sign (root is an odd function): cbrt(-x) = -cbrt(x).
        result = as<vFloat>(as<vInt>(result) | sign_bits);
        // x == 0 -> 0 (sign already carried; |0| poly path is finite).
        v_if(ax == 0.0f) { result = as<vFloat>(sign_bits); }
        v_endif;
    } else {
        // Special cases: even root of negative is NaN; root(0)=0.
        v_if(x < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x == 0.0f) {
#if defined(POW_HW_RECIPROCAL)
            result = std::numeric_limits<float>::infinity();  // rsqrt(0) = +inf
#else
            result = 0.0f;  // sqrt(0) = 0
#endif
        }
        v_endif;
    }
    return result;
}

#if defined(HW_PRELOAD)
// GENERIC constant-pool preload variant of pow_hw_eval (any degree).
// Ranking: prgm0 = SQRT2, prgm1 = c[DEG], prgm2 = c[DEG-1]; cv[] = c[DEG-2..0].
// 'magic' (round helper) is hoisted into an LREG by the caller and passed by value.
// Math byte-identical to pow_hw_eval.
// cvspill: caller-hoisted LREG array of the below-prgm coeffs (c[DEG-2..0]).
template <uint32_t DEG>
inline vFloat pow_hw_eval_preloaded(vFloat x, vFloat magic_hoist, const float* cvspill) {
    // The preload optimization (prgm0 = sqrt(2)) only applies to plain sqrt
    // (ROOT_N == 2, no reciprocal). For odd roots (cbrt) and the reciprocal
    // fold (rsqrt) the scale/sign/recip logic doesn't fit the 3-prgm budget, so
    // fall back to the fully-general evaluator — those kernels are correctness-
    // first and still avoid the segment cascade.
#if (POW_HW_ROOT_N != 2) || defined(POW_HW_RECIPROCAL)
    (void)magic_hoist;
    (void)cvspill;
    return pow_hw_eval<DEG>(x);
#else
    vInt biased = exexp(x, ExponentMode::Biased);
    vInt e_int = biased - 127;
    vFloat m = setexp(x, 127);

    vFloat s;
    if constexpr (DEG >= 1) {
        s = vConstFloatPrgm1 * m + vConstFloatPrgm2;  // c[DEG]*m + c[DEG-1]
#pragma GCC unroll 16
        for (int k = (int)DEG - 2; k >= 0; k--) {
            s = s * m + cvspill[(int)DEG - 2 - k];  // c[DEG-2], c[DEG-3], ... c[0]
        }
    } else {
        s = vConstFloatPrgm1;  // degree-0: c[0]
    }

    vInt r = e_int & 1;
    vFloat ef = convert<vFloat>(as<vSMag>(e_int), RoundMode::Nearest);
    v_if(e_int < 0) {
        vInt mag = ~e_int + 1;
        ef = -convert<vFloat>(as<vSMag>(mag), RoundMode::Nearest);
    }
    v_endif;
    vFloat qf = (ef - convert<vFloat>(as<vSMag>(r), RoundMode::Nearest)) * 0.5f;
    vInt q = as<vInt>(qf + magic_hoist) - as<vInt>(magic_hoist);

    // Apply sqrt(2) when e is odd (preloaded in vConstFloatPrgm0).
    v_if(r == 1) { s = s * vConstFloatPrgm0; }
    v_endif;

    vInt s_exp = exexp(s, ExponentMode::Biased);
    vFloat result = setexp(s, s_exp + q);

    v_if(x < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x == 0.0f) { result = 0.0f; }
    v_endif;
    return result;
#endif
}
#endif  // HW_PRELOAD
#endif

// ============================================================================
// Asymptotic factoring: inline Cody-Waite exp for dominant factor computation
// Used when correction polynomials are multiplied by exp-based dominant factors
// to handle extreme dynamic range in function tails (e.g., GELU, sigmoid).
// Mutually exclusive with range reduction (never combined).
// ============================================================================
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)

// Inline Cody-Waite exp(arg) — no overflow/underflow checks needed because
// the asymptotic region has known bounded input range.
// Degree-5 Taylor for exp(r), |r| < ln(2)/2: relative error < 3.3e-9.
inline vFloat asymptotic_exp(vFloat arg) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    vFloat z = arg * INV_LN2;

    // Round to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    vInt k_int = as<vInt>(tmp) - as<vInt>(c231);

    // Cody-Waite extended precision: r = arg - k*ln2
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vFloat r = k * NEG_LN2_HI + arg;
    r = k * NEG_LN2_LO + r;

    // Degree-5 Taylor for exp(r)
    vFloat p = 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    // Scale by 2^k via exponent bit manipulation.
    //
    // UNDERFLOW GUARD: setexp writes the exponent FIELD directly, so a biased
    // exponent that lands at or below 0 wraps into a large positive exponent
    // and returns ~1e38 where the true value has underflowed to ~0.  Measured
    // on Blackhole via erf_bw = (2/sqrt(pi))*exp(-x^2): the wrap begins exactly
    // where round(-x^2/ln2) < -127, i.e. |x| >= 9.4375, which returned 2.72e38
    // against a golden of 2.35e-39 (150 grid points, symmetric in +/-x).  The
    // clamp must come AFTER setexp: setexp(0.0f, e) rebuilds a nonzero value
    // from the zero mantissa, so zeroing p beforehand does not help.
    vInt p_exp = exexp(p, ExponentMode::Biased);
    vInt new_exp = p_exp + k_int;
    vFloat out = setexp(p, new_exp);
    v_if(new_exp <= 0) { out = 0.0f; }
    v_endif;
    return out;
}

#endif  // ASYMPTOTIC_FACTOR_EXP_*

inline vFloat sfpu_mad(vFloat a, vFloat b, vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), SFPMAD_MOD1_OFFSET_NONE);
}

// Generic polynomial evaluation using Horner's method with compile-time optimization
template <uint32_t DEGREE>
inline vFloat eval_polynomial(const float* coeffs, vFloat x) {
    if constexpr (DEGREE == 0) {
        // Constant: y = c0
        return coeffs[0];
    } else if constexpr (DEGREE == 1) {
        // Linear: y = c0 + c1*x
        return sfpu_mad(coeffs[1], x, coeffs[0]);
    } else if constexpr (DEGREE == 2) {
        // Quadratic: y = c0 + c1*x + c2*x²
        // Horner: (c2*x + c1)*x + c0
        return sfpu_mad(sfpu_mad(coeffs[2], x, coeffs[1]), x, coeffs[0]);
    } else if constexpr (DEGREE == 3) {
        // Cubic: y = c0 + c1*x + c2*x² + c3*x³
        // Horner: ((c3*x + c2)*x + c1)*x + c0
        return sfpu_mad(sfpu_mad(sfpu_mad(coeffs[3], x, coeffs[2]), x, coeffs[1]), x, coeffs[0]);
    } else if constexpr (DEGREE == 4) {
        // Quartic: y = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴
        // Horner: ((((c4*x + c3)*x + c2)*x + c1)*x + c0
        vFloat result = coeffs[4];
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 5) {
        // Quintic: y = c0 + ... + c5*x⁵
        vFloat result = coeffs[5];
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 6) {
        // Hexic: y = c0 + ... + c6*x⁶
        vFloat result = coeffs[6];
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 7) {
        // Septic: y = c0 + ... + c7*x⁷
        vFloat result = coeffs[7];
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 8) {
        // Octic: y = c0 + ... + c8*x⁸
        vFloat result = coeffs[8];
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 9) {
        // Nonic: y = c0 + ... + c9*x⁹
        vFloat result = coeffs[9];
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 10) {
        // Decic: y = c0 + ... + c10*x¹⁰
        vFloat result = coeffs[10];
        result = sfpu_mad(result, x, coeffs[9]);
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 11) {
        // Undecic: y = c0 + ... + c11*x¹¹
        vFloat result = coeffs[11];
        result = sfpu_mad(result, x, coeffs[10]);
        result = sfpu_mad(result, x, coeffs[9]);
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 12) {
        // Dodecic: y = c0 + ... + c12*x¹²
        vFloat result = coeffs[12];
        result = sfpu_mad(result, x, coeffs[11]);
        result = sfpu_mad(result, x, coeffs[10]);
        result = sfpu_mad(result, x, coeffs[9]);
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 13) {
        // Tridecic: y = c0 + ... + c13*x¹³
        vFloat result = coeffs[13];
        result = sfpu_mad(result, x, coeffs[12]);
        result = sfpu_mad(result, x, coeffs[11]);
        result = sfpu_mad(result, x, coeffs[10]);
        result = sfpu_mad(result, x, coeffs[9]);
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 14) {
        // Tetradecic: y = c0 + ... + c14*x¹⁴
        vFloat result = coeffs[14];
        result = sfpu_mad(result, x, coeffs[13]);
        result = sfpu_mad(result, x, coeffs[12]);
        result = sfpu_mad(result, x, coeffs[11]);
        result = sfpu_mad(result, x, coeffs[10]);
        result = sfpu_mad(result, x, coeffs[9]);
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 15) {
        // Pentadecic: y = c0 + ... + c15*x¹⁵
        vFloat result = coeffs[15];
        result = sfpu_mad(result, x, coeffs[14]);
        result = sfpu_mad(result, x, coeffs[13]);
        result = sfpu_mad(result, x, coeffs[12]);
        result = sfpu_mad(result, x, coeffs[11]);
        result = sfpu_mad(result, x, coeffs[10]);
        result = sfpu_mad(result, x, coeffs[9]);
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 16) {
        // Hexadecic: y = c0 + ... + c16*x¹⁶
        vFloat result = coeffs[16];
        result = sfpu_mad(result, x, coeffs[15]);
        result = sfpu_mad(result, x, coeffs[14]);
        result = sfpu_mad(result, x, coeffs[13]);
        result = sfpu_mad(result, x, coeffs[12]);
        result = sfpu_mad(result, x, coeffs[11]);
        result = sfpu_mad(result, x, coeffs[10]);
        result = sfpu_mad(result, x, coeffs[9]);
        result = sfpu_mad(result, x, coeffs[8]);
        result = sfpu_mad(result, x, coeffs[7]);
        result = sfpu_mad(result, x, coeffs[6]);
        result = sfpu_mad(result, x, coeffs[5]);
        result = sfpu_mad(result, x, coeffs[4]);
        result = sfpu_mad(result, x, coeffs[3]);
        result = sfpu_mad(result, x, coeffs[2]);
        result = sfpu_mad(result, x, coeffs[1]);
        result = sfpu_mad(result, x, coeffs[0]);
        return result;
    } else if constexpr (DEGREE == 32) {
        // Duotrigesic: y = c0 + ... + c32*x³²
        vFloat result = coeffs[32];
        for (int i = 31; i >= 0; i--) {
            result = sfpu_mad(result, x, coeffs[i]);
        }
        return result;
    }
}

// Linear-term-SPLIT reconstruction for the REDUCE_LOG sqrt2-centered route
// (TTNN ckernel_sfpu_log.h / Juffa log1p_hw_eval structure, line ~4092's
// `result = r + (r*r)*p`): instead of the monolithic Horner P(f) -- whose
// dominant ~unit-magnitude linear term accumulates every per-FMA rounding --
// evaluate only the degree-(DEGREE-2) correction Q(f) = coeffs[2..DEGREE] and
// inject the dominant linear term c1*f EXACTLY, ONCE, OUTSIDE the Horner chain:
//     f = m-1 (Sterbenz-exact reduced arg);  P(f) = c1*f + f^2 * Q(f)
// Same degree, same single interval, SAME fitted coefficients as the monolithic
// route -- ONLY the evaluation structure differs. Host-proven under fma_bh +
// ulp_error_pure to break the monolithic-Horner floor (log 1.5330 -> 0.9047,
// log2 2.3400 -> 1.8956, log10 2.1846 -> 1.8938 max pure ULP over the full fp32
// domain incl binade 126), all three BEATING their TTNN targets. This is the
// byte-for-byte arithmetic ttpoly.precision.rangered models for
// log_reduce_compose="sqrt2_centered_split" (the committed conformance gate).
// +1 SFPMAD over eval_polynomial<DEGREE> (well under the 32-slot budget).
template <uint32_t DEGREE>
inline vFloat eval_polynomial_log_split(const float* coeffs, vFloat f) {
    static_assert(DEGREE >= 2, "REDUCE_LOG linear-split needs degree >= 2");
    // Q(f) = coeffs[2] + coeffs[3]*f + ... + coeffs[DEGREE]*f^(DEGREE-2)
    vFloat q = coeffs[DEGREE];
#pragma GCC unroll 16
    for (int i = (int)DEGREE - 1; i >= 2; i--) {
        q = sfpu_mad(q, f, coeffs[i]);
    }
    vFloat s = f * f;                     // f^2 (reduced arg is exact, so is f*f-ish)
    vFloat corr = s * q;                  // f^2 * Q(f) -- the small correction term
    return sfpu_mad(coeffs[1], f, corr);  // c1*f (injected EXACTLY) + correction
}

// x²-Horner evaluation for polynomials with parity structure.
// Halves Horner step count by evaluating in x² basis with stride-2 coefficient access.
//
// Odd parity (c0=c2=c4=...=0):  P(x) = x * Horner([c1,c3,c5,...], x²)
// Even parity (c1=c3=c5=...=0): P(x) = Horner([c0,c2,c4,...], x²)
//
// DEGREE is the nominal (max) degree — the LUT stores DEGREE+1 coefficients per segment,
// but zero-valued coefficients are skipped via stride-2 access.
#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
inline vFloat eval_polynomial_parity(const float* coeffs, vFloat x, vFloat x2) {
    // Odd: only odd-index coefficients are nonzero (c1,c3,c5,...)
    // TOP = highest odd index ≤ DEGREE
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;  // number of FMA steps after init
    vFloat result = coeffs[TOP];
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        result = result * x2 + coeffs[TOP - 2 * k];
    }
    return result * x;  // final *x for odd parity
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
inline vFloat eval_polynomial_parity(const float* coeffs, vFloat x, vFloat x2) {
    // Even: only even-index coefficients are nonzero (c0,c2,c4,...)
    // TOP = highest even index ≤ DEGREE
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;  // number of FMA steps after init
    vFloat result = coeffs[TOP];
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        result = result * x2 + coeffs[TOP - 2 * k];
    }
    return result;
}
#endif

#ifdef USE_DUAL_EVAL
// Dual polynomial evaluation using Horner's method with interleaved operations.
// Evaluates two independent x-vectors simultaneously to exploit SFPU instruction-level
// parallelism, hiding the 2-cycle SFPMAD latency by interleaving independent chains.
//
// Usage: eval_polynomial_dual<DEGREE>(coeffs, x1, x2, result1, result2)
// The strict alternation (result1 op, result2 op, result1 op, ...) ensures
// the compiler sees independent computation chains that can be pipelined.
template <uint32_t DEGREE>
inline void eval_polynomial_dual(const float* coeffs, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    // Each coefficient is hoisted into a named vFloat so the compiler emits one sfploadi
    // pair and reuses the register as src2 for BOTH MADs, halving coefficient load count.
    if constexpr (DEGREE == 0) {
        vFloat c = coeffs[0];
        result1 = c;
        result2 = c;
    } else if constexpr (DEGREE == 1) {
        vFloat c1 = coeffs[1];
        vFloat c0 = coeffs[0];
        result1 = c0 + c1 * x1;
        result2 = c0 + c1 * x2;
    } else if constexpr (DEGREE == 2) {
        {
            vFloat c = coeffs[2];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 3) {
        {
            vFloat c = coeffs[3];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 4) {
        {
            vFloat c = coeffs[4];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 5) {
        {
            vFloat c = coeffs[5];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 6) {
        {
            vFloat c = coeffs[6];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 7) {
        {
            vFloat c = coeffs[7];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 8) {
        {
            vFloat c = coeffs[8];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 9) {
        {
            vFloat c = coeffs[9];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 10) {
        {
            vFloat c = coeffs[10];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 11) {
        {
            vFloat c = coeffs[11];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 12) {
        {
            vFloat c = coeffs[12];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 13) {
        {
            vFloat c = coeffs[13];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 14) {
        {
            vFloat c = coeffs[14];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[13];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 15) {
        {
            vFloat c = coeffs[15];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[14];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[13];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 16) {
        {
            vFloat c = coeffs[16];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[15];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[14];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[13];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    }
}

// Dual-eval x²-Horner for polynomials with parity structure.
// Two interleaved chains in x² basis for ILP, same stride-2 pattern.
#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
inline void eval_polynomial_dual_parity(
    const float* coeffs, vFloat x1, vFloat x2, vFloat x1_sq, vFloat x2_sq, vFloat& result1, vFloat& result2) {
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;
    {
        vFloat c = coeffs[TOP];
        result1 = c;
        result2 = c;
    }
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        vFloat c = coeffs[TOP - 2 * k];
        result1 = result1 * x1_sq + c;
        result2 = result2 * x2_sq + c;
    }
    result1 = result1 * x1;
    result2 = result2 * x2;
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
inline void eval_polynomial_dual_parity(
    const float* coeffs, vFloat x1, vFloat x2, vFloat x1_sq, vFloat x2_sq, vFloat& result1, vFloat& result2) {
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;
    {
        vFloat c = coeffs[TOP];
        result1 = c;
        result2 = c;
    }
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        vFloat c = coeffs[TOP - 2 * k];
        result1 = result1 * x1_sq + c;
        result2 = result2 * x2_sq + c;
    }
}
#endif

#endif  // USE_DUAL_EVAL

#if defined(BASIS_NORMALIZED_LOG_ODDS_SEPARABLE)
template <uint32_t LUT_SIZE>
inline void normalized_log_odds_separable_eval(
    const std::array<float, LUT_SIZE>& lut) {
    static_assert(BASIS_LOG_ODDS_CORRECTION_DEGREE == 1,
                  "first qualified separable correction is P1");
    static_assert(BASIS_LOG_ODDS_LOG_DEGREE == 2,
                  "first qualified separable log ratio is P2");
    static_assert(LUT_SIZE == 7,
                  "unit-interval boundaries plus five flattened coefficients");
    constexpr uint32_t kCoeff = 2;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = prepare_raw_domain_input(dst_reg[d]);
        // q=min(x,1-x), with q born from the computed complement so it cannot
        // alias the DST-backed x carrier.  The tempting centered identity
        // 0.5-abs(x-0.5) loses x below the fp32 half-ULP of 0.5 (2^-25), so
        // the predicate is required to preserve every bottom-binade normal.
        vFloat complement = vFloat(1.0f) - x;
        vFloat q = complement;
        v_if(x < 0.5f) { q = x; }
        v_endif;

        vFloat m;
        vInt exponent;
        log_reduce(q, m, exponent);
        vFloat r = m - vFloat(1.0f);
        vFloat u = q - 0.5f;

        // CSV layout: C0,C1,L0,L1,L2.  Both chains are ordinary low-to-high
        // fitted coefficients; no activation identity participates.
        vFloat correction = sfpu_mad(lut[kCoeff + 1], u, lut[kCoeff + 0]);
        vFloat log_ratio = sfpu_mad(lut[kCoeff + 4], r, lut[kCoeff + 3]);
        log_ratio = sfpu_mad(log_ratio, r, lut[kCoeff + 2]);
        vFloat log_q = log_expand(r * log_ratio, exponent);
        constexpr float kLn2 = 0.69314718055994530942f;
        vFloat log_one_minus_q = sfpu_mad(u, correction, -kLn2);
        // Reconstruction is complete.  Reload the decoded and encoded source
        // coordinates from DST now, after the arithmetic temporaries retire.
        // Keeping either classifier live across log reduction exceeds the
        // eight-LREG Blackhole budget and corrupts endpoint/raw predicates.
        vFloat x_effective = prepare_raw_domain_input(dst_reg[d]);
        // The magnitude is positive on both halves of the unit interval.
        // Restore the lower-half sign with the freshly reloaded coordinate;
        // this keeps the sign carrier out of reduction/Horner liveness while
        // preserving the exact anonymous basis semantics.
        vFloat result = log_one_minus_q - log_q;
        v_if(x_effective < 0.5f) { result = -result; }
        v_endif;

        // Close the unit interval in the decoded coordinate. NaNs already
        // flow through the log arithmetic to +Inf. The remaining encoded
        // distinction shared by both profiles is exponent-zero: Blackhole's raw
        // BF16 ingress can expose negative subnormal payloads as nonzero SFPU
        // values even though the declared input policy is DAZ.  A late masked
        // equality closes that whole zero/subnormal class without a raw sign,
        // mantissa, NaN, or exterior classifier.
        v_if(x_effective < 0.0f) {
            result = std::numeric_limits<float>::infinity();
        }
        v_endif;
        v_if(x_effective >= 1.0f) {
            result = std::numeric_limits<float>::infinity();
        }
        v_endif;
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        vUInt raw_exponent = raw_u16 & vUInt(0x00ffu);
#if defined(TT_TARGET_LOG_ODDS_RAW_INTERVAL_UNION)
        // S55 exhaustively evaluates the declared rounded target graph and
        // proves that its differing exterior class is one sign-mirrored raw
        // interval ending at exponent-FF. S60 emits those derived endpoints;
        // this anonymous lowering consumes the quotient, not the source graph.
        static_assert(TT_TARGET_LOG_ODDS_INTERVAL_COUNT == 2u);
        static_assert(
            TT_TARGET_LOG_ODDS_POS_END_RAW == 0x7f80u &&
            TT_TARGET_LOG_ODDS_NEG_FIRST_RAW ==
                (TT_TARGET_LOG_ODDS_POS_FIRST_RAW | 0x8000u) &&
            TT_TARGET_LOG_ODDS_NEG_END_RAW ==
                (TT_TARGET_LOG_ODDS_POS_END_RAW | 0x8000u) &&
            TT_TARGET_LOG_ODDS_NEG_INF_PATTERN_COUNT ==
                2u * (TT_TARGET_LOG_ODDS_POS_END_RAW -
                      TT_TARGET_LOG_ODDS_POS_FIRST_RAW));
        constexpr float target_phase_lower = __builtin_bit_cast(
            float, TT_TARGET_LOG_ODDS_POS_FIRST_RAW << 16);
        vFloat magnitude = setsgn(x_effective, 0);
        v_if(magnitude >= target_phase_lower) {
            // Exponent-FF begins exactly at the interval's exclusive end.
            // Reject it in encoded space so NaN/Inf retain the declared +Inf.
            v_if(raw_exponent != 0xffu) {
                result = -std::numeric_limits<float>::infinity();
            }
            v_endif;
        }
        v_endif;
#endif
        v_if(raw_exponent == 0u) {
            result = -std::numeric_limits<float>::infinity();
        }
        v_endif;
#ifdef USE_BF16
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
        dst_reg[d] = result;
    }
}
#endif

// Generic piecewise polynomial LUT for any degree/depth combination
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // Skip boundary points

    // Compile-time validation
    static_assert(
        LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
        "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Process all 32 destination registers
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL) || \
    defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RAW_NEG_EXP_FF_INGRESS)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
#if defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RAW_NEG_EXP_FF_INGRESS)
        vFloat x_orig = prepare_raw_domain_input(x_raw, raw_u16);
#else
        vFloat x_orig = prepare_raw_domain_input(x_raw);
#endif

#if defined(RANGE_REDUCTION_EXP)
        // Exp range reduction to [-ln(2)/2, ln(2)/2].  Overflow is decided
        // per-arm, and exp_expand's reconstructed-exponent guard
        // (new_exp >= 255 -> +inf) is the precise saturator.
        //
        // PLAIN arm: the old EXP_OVERFLOW=88.5f clamped the finite band
        // [88.5, ln(FLT_MAX)=88.7228) to +inf, which is wrong for the exhaustive
        // pure-ULP metric (exp(88.6)=2.9e38 << FLT_MAX is representable) and is
        // exactly what made exp/sigmoid return exhaustive_max_pure_ulp=inf on
        // job 78410. EXP_OVERFLOW is now only an extreme anti-wrap sentinel
        // (256, well above every family overflow so k_int cannot integer-wrap);
        // exp_expand owns the true plain-exp / silu-swish saturation.
        //
        // HYPERBOLIC composes keep their own empirically-verified shoulder:
        // cosh/sinh stay finite far past exp(x)'s overflow because th=0.5*exp(|x|)
        // does not hit FLT_MAX until x=ln(2*FLT_MAX)=89.41598... (mpmath-verified).
        // EXP_OVERFLOW_HYPERBOLIC=89.416f sits inside the grid-verified
        // [89.41573, 89.41603] gap -- it neither clips a computable input nor
        // lets a genuinely-overflowing one reach the exponent-fold. Each constant
        // is declared only in the arm(s) that read it (-Wunused-variable otherwise:
        // hyperbolic never reads plain EXP_OVERFLOW, plain/sigmoid never read
        // EXP_OVERFLOW_HYPERBOLIC).
#if defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD)
        constexpr float EXP_OVERFLOW_HYPERBOLIC = 89.416f;
#elif defined(REDUCE_EXP_BASE2)
        // exp2's shoulders are owned by the pre-evaluation return_class
        // terminals (x<=-126.5 -> +0, x>=128 -> +inf) and exp_expand's own
        // new_exp guards, so this arm reads no EXP_OVERFLOW constant.
#elif defined(REDUCE_EXP_COMPOSE_ELU) || defined(REDUCE_EXP_COMPOSE_SELU)
        // elu/celu/selu own both shoulders themselves: the x>=0 identity/affine
        // branch and the exact -1 / -lambda*alpha saturation tail (at -25*ln2 /
        // its selu analogue). No natural-exp EXP_OVERFLOW constant is read.
#else
        constexpr float EXP_OVERFLOW = 256.0f;
#endif
        vFloat x;
        vInt k_int;
#if defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD)
        // cosh(x) = 0.5*exp(|x|) + 0.5*exp(-|x|), sinh(x) = sign(x)*(0.5*
        // exp(|x|) - 0.5*exp(-|x|)): both reduce on the ABS coordinate (one
        // reduction serves both signed exponentials) and reconstruct via the
        // reciprocal-composite combine after the segment cascade -- see the
        // Expand block below. x_abs is reused there for the overflow test
        // (and, for sinh, the small-|x| dual-schedule crossover and the
        // final copysign) so setsgn only runs once per element.
        vFloat x_abs = setsgn(x_orig, 0);
        exp_reduce(x_abs, x, k_int);
#elif defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT) || defined(REDUCE_EXP_COMPOSE_SIGMOID)
        // silu/swish = x/(1+exp(-x)) and plain sigmoid = 1/(1+exp(-x)): both
        // reduce on -x directly (no abs -- the sign is already carried by which
        // side of zero -x lands on). The two arms diverge only at reconstruct.
        vFloat x_neg = -x_orig;
        exp_reduce(x_neg, x, k_int);
#elif defined(REDUCE_EXP_BASE2)
        // exp2(x) = 2^k * 2^r: exact integer reduction (k=round(x), r=x-k),
        // NOT the Cody-Waite ln2 split.
        exp2_reduce(x_orig, x, k_int);
#else
        exp_reduce(x_orig, x, k_int);
#endif
#elif defined(RANGE_REDUCTION_TRIG)
        // Trig range reduction: reduce to [-π/2, π/2]
        vFloat x;
        vInt q_int;
        trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        // Tan range reduction: reduce to [-π/4, π/4] (Cody-Waite)
        vFloat x;
        vInt j_int;
        tan_reduce(x_orig, x, j_int);
#elif defined(RANGE_REDUCTION_LOG)
        // Log range reduction: extract mantissa m ∈ [1, 2) and exponent e via
        // exponent-FIELD bit manipulation (never an fp adder).
        // log(x) = e*ln(2) + log(m)
        vFloat x;
        vInt e_int;
        log_reduce(x_orig, x, e_int);
#if defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED) || defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT)
        // sqrt(2)-centered mantissa split: log_reduce above already gives m in
        // [1,2); fold the [sqrt(2),2) half back down by exactly one more
        // exponent-FIELD step (setexp, not an fp subtract) whenever
        // m >= sqrt(2), so the two half-open pieces [1,sqrt(2)) and
        // [sqrt(2)/2,1) become one symmetric-about-1 residual range
        // [sqrt(2)/2, sqrt(2)) instead of the one-sided [1,2). log_expand
        // below is UNCHANGED (it is just e*ln(2)+poly(m), agnostic to which
        // split produced m/e_int) -- only the reduce side needs this arm.
        // This is what lets the single-segment minimax fit clear TTNN: the
        // worst-case |m-1| the polynomial must cover roughly halves (m-1 in
        // [-0.2929,0.4142) vs [0,1) for the plain split).
        constexpr float LOG_SQRT2 = 1.4142135623730951f;
        v_if(x >= LOG_SQRT2) {
            e_int = e_int + 1;
            x = setexp(x, 126);
        }
        v_endif;
        // The fitted polynomial is pinned at the log(1)=0 root: it is fit in
        // the residual f = m-1 (single-segment minimax on log(1+f), c0
        // constrained to exactly 0.0), not in m directly -- so f=0 evaluates
        // the polynomial's own EXACT constant term instead of whatever
        // near-zero residual an unconstrained fit would leave at m=1. This
        // subtract is exact (no rounding): m is in [sqrt(2)/2, sqrt(2)),
        // which is within a factor of 2 of 1.0 on both sides, so Sterbenz's
        // lemma guarantees m-1.0f has no rounding error. log_expand below is
        // unaffected -- it only ever sees the final poly VALUE (which equals
        // log(m) either way), never the coordinate it was evaluated in.
        x = x - 1.0f;
#endif
#elif defined(RANGE_REDUCTION_CBRT)
        // Extract BIASED exponent (always non-negative) and sign BEFORE poly eval.
        // Use exexp_nodebias to avoid SFPU's sign-magnitude format for negative debiased values.
        vInt cbrt_biased_e = exexp(setsgn(x_orig, 0), ExponentMode::Biased);
        vInt cbrt_sign = as<vInt>(x_orig) & 0x80000000;
        vFloat x = cbrt_reduce_m(x_orig);
#elif defined(RANGE_REDUCTION_RECIP_COMPLEMENT)
        // Signed-abs reciprocal/complement reduction (atan). reduced =
        // min(|x|, 1/|x|) in [0,1]; the poly coordinate is s = reduced^2, so ONE
        // even core ratio P(s) ~= f(sqrt(s))/sqrt(s) serves both the direct leaf
        // (|x|<=1) and the reciprocal-folded tail (|x|>1). recip_reduced persists
        // to the reconstruction block below (same scope, like cbrt's e/sign).
        vFloat recip_reduced;
        vFloat x;
        {
            vFloat abs_x = setsgn(x_orig, 0);                                // |x|
            vFloat inv_abs = ckernel::sfpu::sfpu_reciprocal_iter<2>(abs_x);  // 1/|x|
            recip_reduced = abs_x;
            v_if(abs_x > 1.0f) { recip_reduced = inv_abs; }                  // min(|x|, 1/|x|)
            v_endif;
            x = recip_reduced * recip_reduced;                              // s = reduced^2 in [0,1]
        }
#else
        vFloat x = x_orig;
#endif

#if defined(BASIS_INPUT_ABS_X)
        vFloat x_eval = setsgn(x_orig, 0);
        vFloat& x_clamped = x_eval;
#else
        vFloat& x_eval = x;
        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x_clamped = x;
#endif

        // Start with segment 0 coefficients
#if defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT)
        // TTNN / log1p-style linear-term-split reconstruction (see
        // eval_polynomial_log_split): the dominant linear term c1*f is injected
        // EXACTLY, outside the Horner chain, breaking the monolithic-Horner
        // floor. Single-segment route only (the sqrt2-centered log family is
        // fit as one interval); fail-closed if a multi-segment cascade is ever
        // paired with this compose, since the split's exact-linear-term
        // reconstruction is only defined per single fitted polynomial.
        static_assert(NUM_SEGMENTS == 1,
                      "REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT requires a single segment");
        vFloat result = eval_polynomial_log_split<POLY_DEGREE>(&lut[COEFF_OFFSET], x_eval);
#else
        vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x_eval);

        // Update with correct segment using plain for-loop
        // Works correctly on all architectures without kernel bloat
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if(x_clamped >= lut[seg]) {
                result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x_eval);
            }
            v_endif;
        }
#endif

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        result = result * x_eval;
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
        // sqrt_factored basis: acos(|x|) = sqrt(1-|x|) * P(|x|)
        result = result * basis_sqrt(vFloat(1.0f) - x_eval);
#endif

#if defined(BASIS_AFFINE_EVEN)
        result =
            sfpu_mad(BASIS_AFFINE_EVEN_SCALE * x_eval, result, sfpu_mad(BASIS_AFFINE_SCALE, x_orig, BASIS_AFFINE_BIAS));
#endif

#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
        // Keep the canonical single-lane cascade feature-complete with the
        // specialized body: action-bearing basis artifacts route here so the
        // existing abs-coordinate tail must survive unchanged.
        v_if(x_eval > BASIS_RIGHT_TAIL_ABS_THRESHOLD) {
            result = BASIS_RIGHT_TAIL_ABS_A * x_eval + BASIS_RIGHT_TAIL_ABS_B;
        }
        v_endif;
#endif

#if defined(BASIS_CLAMP_MAX)
        vFloat basis_clamp_max_value = BASIS_CLAMP_MAX_VALUE;
        ordered_min_max(result, basis_clamp_max_value);
#endif

#if defined(BASIS_POST_SIGN_X)
        result = copysgn(result, x_orig);
#endif

#if defined(BASIS_POST_REFLECT_PI)
        // sqrt_factored acos: acos(-|x|) = pi - acos(|x|)
        v_if(x_orig < 0.0f) { result = BASIS_PI_VALUE - result; }
        v_endif;
#endif

#if defined(BASIS_LEFT_TAIL_ZERO)
        v_if(x_orig < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { result = 0.0f; }
        v_endif;
#endif

#if defined(BASIS_RIGHT_TAIL_IDENTITY)
        v_if(x_orig > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { result = x_orig; }
        v_endif;
#endif

#if defined(ASYMPTOTIC_FACTOR_X) && defined(ASYMPTOTIC_REGION_ALL)
        // Whole-domain odd factoring: the selected CSV approximates f(x)/x.
        // The compact loop-row evaluator must perform the same declared
        // reconstruction as the specialized evaluator before terminal actions.
        result = (result * x_orig) * ASYMPTOTIC_SCALE;
#endif

#if defined(ASYMPTOTIC_FACTOR_QUADRATIC)
        // Interior-root factoring: f(x) = (x-A)(x-B) * ASYMPTOTIC_SCALE * P(x),
        // whole-domain. `result` is the smooth correction P(x) = f(x)/((x-A)(x-B));
        // multiply by the two-root quadratic so the interior roots (lgamma: 1,2)
        // are structurally EXACT and the near-root pure-ULP artifact vanishes.
        // Guarded -> inert for ops that do not define it.
#ifndef ASYMPTOTIC_SCALE
#define ASYMPTOTIC_SCALE 1.0f
#endif
        {
            vFloat q = (x_orig - ASYMPTOTIC_QUAD_ROOT_A) * (x_orig - ASYMPTOTIC_QUAD_ROOT_B);
            result = (result * q) * ASYMPTOTIC_SCALE;
        }
#endif

#if defined(RANGE_REDUCTION_EXP)
#if defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN)
        // cosh(x): fold the 0.5 scale into the exponent field of BOTH
        // reconstructed terms instead of an explicit multiply --
        // th = P * 2^(k-1) = 0.5*exp(|x|), rh = (1/P) * 2^(-k-1) =
        // 0.5*exp(-|x|). exp_expand(poly, e) already does exactly
        // "setexp(poly, exexp_nodebias(poly) + e)", so reusing it with
        // (k_int - 1) / (-k_int - 1) IS the addexp-style fold -- no new
        // primitive needed. `result` here is still the raw Horner output
        // P(s) ~= exp(s); exp_expand takes it by value, so both calls below
        // read the same pre-expand P. rh underflows to 0.0f gracefully via
        // exp_expand's own new_exp<=0 guard as |x| grows (0.5*exp(-|x|) -> 0),
        // while th is only reachable once the EXP_OVERFLOW_HYPERBOLIC
        // shoulder below has already routed the +inf case away, so its own
        // exponent field never wraps.
        v_if(x_abs > EXP_OVERFLOW_HYPERBOLIC) { result = std::numeric_limits<float>::infinity(); }
        v_else {
            vFloat recip_poly = ckernel::sfpu::sfpu_reciprocal_iter<2>(result);
            vFloat th = exp_expand(result, k_int - 1);
            vFloat rh = exp_expand(recip_poly, -k_int - 1);
            result = th + rh;
        }
        v_endif;
#elif defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD)
        // sinh(x) = sign(x)*(0.5*exp(|x|) - 0.5*exp(-|x|)), same th/rh
        // exponent-fold as cosh, but SUBTRACTED and sign-restored via
        // copysgn(., x_orig) -- th and rh are each computed from |x|, so the
        // combine itself is always the even/positive-branch magnitude.
        //
        // CRITICAL (unlike cosh's addition): th-rh catastrophically cancels
        // for small |x|. At k_int==0 (|x| < ln(2)/2, the ENTIRE reduced
        // segment) th=0.5*P and rh=0.5*(1/P) are both ~0.5 and nearly equal
        // -- their difference (~|x| itself) is computed from two values
        // whose OWN rounding error (~2^-24, from th/rh's ~0.5 magnitude) does
        // not shrink as |x|->0, so the relative/ULP error of the difference
        // blows up (measured: tens to hundreds of ULP right at this
        // boundary). At k_int>=1 (|x| >= ln(2)/2) th and rh are already
        // separated by a factor of >=2 (Sterbenz-safe, no amplification), so
        // ONLY the k_int==0 region needs the fallback -- SINH_SMALL_X_
        // COEFFS is a from-scratch odd minimax fit of sinh(x) itself
        // (Sollya fpminimax, not reused from the exp(s) fit) over exactly
        // that same reduced range, evaluated directly in x_orig (already
        // correctly signed, an odd polynomial -- no copysgn needed there).
        v_if(x_abs > EXP_OVERFLOW_HYPERBOLIC) { result = copysgn(vFloat(std::numeric_limits<float>::infinity()), x_orig); }
        v_elseif(x_abs < SINH_SMALL_X_THRESHOLD) {
            result = eval_polynomial<SINH_SMALL_X_DEGREE>(SINH_SMALL_X_COEFFS, x_orig);
        }
        v_else {
            vFloat recip_poly = ckernel::sfpu::sfpu_reciprocal_iter<2>(result);
            vFloat th = exp_expand(result, k_int - 1);
            vFloat rh = exp_expand(recip_poly, -k_int - 1);
            result = copysgn(th - rh, x_orig);
        }
        v_endif;
#elif defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT)
        // silu/swish = x/(1+exp(-x)). t = exp(-x) via the reduction above
        // (fed -x_orig); full 2^k expand (no 0.5 exponent-fold here, unlike
        // the hyperbolic composes). As x -> -inf, t overflows and the naive
        // limit is x*reciprocal(inf) = -inf*0 = NaN in IEEE, not the true
        // limit 0 -- guarded explicitly. As x -> +inf, t underflows to
        // exactly 0.0f via exp_expand's own new_exp<=0 guard, denom becomes
        // exactly 1.0f, and result = x*1 = x -- BUT only while the Cody-Waite
        // reduction is still valid. The magic-number round in exp_reduce loses
        // its round once |z=(-x)*INV_LN2| exceeds 2^23 (x >= 2^24 ~= 1.68e7),
        // so k_int (hence t, denom, recip) goes to garbage and silu returns 0
        // instead of the true limit x. Guard the whole large-+x tail the same
        // way the -inf tail is guarded: silu(x)=x*sigmoid(x) -> x*1 = x for
        // every x > EXP_OVERFLOW (sigmoid(x)==1.0f exactly there), which also
        // matches the reduction-valid band [EXP_OVERFLOW, 2^24) bit-for-bit.
        v_if(x_orig < -EXP_OVERFLOW) { result = 0.0f; }
        v_elseif(x_orig > EXP_OVERFLOW) { result = x_orig; }
        v_else {
            vFloat t = exp_expand(result, k_int);
            vFloat denom = t + 1.0f;
            vFloat recip = ckernel::sfpu::sfpu_reciprocal_iter<2>(denom);
            result = x_orig * recip;
        }
        v_endif;
#elif defined(REDUCE_EXP_COMPOSE_SIGMOID)
        // sigmoid(x) = 1/(1+exp(-x)). Identical to the sigmoid_product arm
        // above WITHOUT the final ``x *`` factor. t = exp(-x) via the reduction
        // above (fed -x_orig), full 2^k expand. As x -> -inf, t overflows and
        // 1/(1+inf) underflows to a NaN in IEEE, not the true limit 0 --
        // guarded explicitly. As x -> +inf, t underflows to exactly 0.0f via
        // exp_expand's own new_exp<=0 guard, denom becomes exactly 1.0f, and
        // result = 1/1 = 1 -- BUT only while the Cody-Waite reduction is still
        // valid. The magic-number round in exp_reduce loses its round once
        // |z=(-x)*INV_LN2| exceeds 2^23 (x >= 2^24 ~= 1.68e7), so k_int (hence
        // t, denom, recip) goes to garbage and sigmoid returns 0 instead of the
        // true limit 1. Guard the whole large-+x tail the same way the -inf
        // tail is guarded: sigmoid(x) -> 1.0f for every x > EXP_OVERFLOW, which
        // also matches the reduction-valid band [EXP_OVERFLOW, 2^24) bit-for-bit
        // (t underflows to 0, denom==1, recip==1.0f there).
        v_if(x_orig < -EXP_OVERFLOW) { result = 0.0f; }
        v_elseif(x_orig > EXP_OVERFLOW) { result = 1.0f; }
        v_else {
            vFloat t = exp_expand(result, k_int);
            vFloat denom = t + 1.0f;
            vFloat recip = ckernel::sfpu::sfpu_reciprocal_iter<2>(denom);
            result = recip;
        }
        v_endif;
#elif defined(REDUCE_EXP_BASE2)
        // exp2(x) = 2^k * 2^r: the reduced leaf P(r) ~= 2^r, reconstructed with
        // the same exact 2^k exponent fold as the plain arm. The pre-evaluation
        // return_class terminals already own the +0/+inf shoulders (x<=-126.5,
        // x>=128), and exp_expand's new_exp<=0 guard covers residual underflow,
        // so no separate overflow/underflow branch is needed here.
        result = exp_expand(result, k_int);
#elif defined(REDUCE_EXP_COMPOSE_ELU) || defined(REDUCE_EXP_COMPOSE_SELU)
        // elu/celu/selu. The reduced leaf here is the CANCELLATION-FREE
        // expm1(r)/r core (NOT exp(r)): result == P(r) ~= expm1(r)/r.  Rebuild
        //     expm1(x) = (2^k - 1) + 2^k * (r * P(r))
        // The (2^k - 1) term is EXACT for every k in the reconstruction band
        // (x in [tail, 0] => k in [-25, 0] => 2^k <= 1), so the leading -1 never
        // catastrophically cancels against the small residual the way a direct
        // polynomial of e^x-1 does on the negative shelf (that direct form floors
        // at ~1500 fp32 ULP; this reconstruction reaches ~1 ULP).  `x` is the
        // reduced coordinate r (exp_reduce output, fed x_orig); k_int the
        // Cody-Waite integer.  For x_orig>=0 / x_orig<=tail the middle-band value
        // is discarded by the identity/affine and saturation overwrites below,
        // so any 2^k overflow there is harmless.
        {
            vFloat reduced_expm1 = result * x;                  // r * P(r) = expm1(r)
            vFloat two_to_k = exp_expand(vFloat(1.0f), k_int);  // 2^k (exact power of two)
            vFloat base = two_to_k - 1.0f;                      // 2^k - 1 (exact for k<=0)
            vFloat expm1_x = base + two_to_k * reduced_expm1;   // expm1(x)
#if defined(REDUCE_EXP_COMPOSE_SELU)
            // selu(x) = lambda*x (x>=0) | lambda*alpha*expm1(x) (x<0); saturates
            // to -lambda*alpha.  SELU_SAT_BOUND is the greatest fp32 x whose
            // RN32(selu(x)) is exactly -lambda*alpha.
            constexpr float SELU_LAMBDA = 1.0507010221481323f;
            constexpr float SELU_LAMBDA_ALPHA = 1.7580993175506592f;
            constexpr float SELU_SAT_BOUND = -16.869848251342773f;
            result = SELU_LAMBDA_ALPHA * expm1_x;
            v_if(x_orig >= 0.0f) { result = SELU_LAMBDA * x_orig; }
            v_elseif(x_orig <= SELU_SAT_BOUND) { result = -SELU_LAMBDA_ALPHA; }
            v_endif;
#else
            // elu/celu(alpha=1): elu(x) = x (x>=0) | expm1(x) (x<0); saturates to
            // -1.  ELU_SAT_BOUND is the greatest fp32 x <= -25*ln2 (RN32(elu)==-1).
            constexpr float ELU_SAT_BOUND = -17.32868003845215f;
            result = expm1_x;
            v_if(x_orig >= 0.0f) { result = x_orig; }
            v_elseif(x_orig <= ELU_SAT_BOUND) { result = -1.0f; }
            v_endif;
#endif
            // elu/celu/selu are SIGN-CONCORDANT: sign(f(x)) == sign(x) for every x,
            // including +-0. The expm1 reconstruction yields +0 for expm1(-0)
            // (2^0-1=+0, plus 2^0*(-0)=-0, and +0 + -0 == +0), and the >=0 identity
            // select does not always recover -0 on device (the sign-preserving-DAZ
            // subnormal population + the -0 input flush to +0, ~2^23 mismatches).
            // Force the output sign from x_orig -- a no-op for every normal case and
            // independent of the device signed-zero compare semantics.
            result = copysgn(result, x_orig);
        }
#else
        // Expand: result = poly_result * 2^k, with overflow/underflow clamping.
        // EXP_UNDERFLOW is local to this (plain-exp) arm only -- the compose
        // arms above never underflow to a hard 0.0f the same way (cosh/sinh's
        // rh vanishes gracefully through exp_expand's own guard instead, and
        // silu/swish's guard is the -EXP_OVERFLOW branch), so declaring it
        // unconditionally at file scope left it unused (-Wunused-variable)
        // whenever a compose macro was defined.
        constexpr float EXP_UNDERFLOW = -88.5f;
        v_if(x_orig > EXP_OVERFLOW) { result = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig < EXP_UNDERFLOW) { result = 0.0f; }
        v_else { result = exp_expand(result, k_int); }
        v_endif;
#endif
#elif defined(RANGE_REDUCTION_TRIG)
        // Trig expand: negate result if quotient is odd
        result = trig_expand(result, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        // Tan expand: j even → poly(a), j odd → -1/poly(a)
        result = tan_expand(result, j_int);
#elif defined(RANGE_REDUCTION_LOG)
        // Log expand: log(x) = e*ln(2) + log(m)
        // Handle special cases: log(0) = -inf, log(negative) = NaN
        v_if(x_orig < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
        v_else { result = log_expand(result, e_int); }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Compute q = floor(e/3) and r = e mod 3 using biased exponent.
        // All arithmetic stays in float to avoid SFPU's sign-magnitude int format.
        {
            constexpr float ONE_THIRD_C = 0.3333333333333333f;
            const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);

            // Convert biased exponent to float and debias: e_float = biased - 127
            vFloat e_float = convert<vFloat>(as<vSMag>(cbrt_biased_e), RoundMode::Nearest) - 127.0f;

            // q_float ≈ e/3, then round to nearest integer
            vFloat q_approx = e_float * ONE_THIRD_C;
            vFloat q_rounded = q_approx + magic;
            vInt q = as<vInt>(q_rounded) - as<vInt>(magic);

            // Get q as float WITHOUT int32_to_float (avoids sign-magnitude)
            vFloat q_back = q_rounded - magic;

            // r = e - 3*q in float, then convert to int via magic number
            vFloat r_float = e_float - (q_back + q_back + q_back);
            vInt r = as<vInt>(r_float + magic) - as<vInt>(magic);

            // Corrections for r ∈ {0,1,2}
            v_if(r < 0) {
                q = q - 1;
                r = r + 3;
            }
            v_endif;
            v_if(r > 2) {
                q = q + 1;
                r = r - 3;
            }
            v_endif;

            // Handle x=0 (biased_exp = 0 for zero/denorm)
            v_if(cbrt_biased_e < 1) { result = 0.0f; }
            v_else { result = cbrt_expand(result, q, r, cbrt_sign); }
            v_endif;
        }
#elif defined(RANGE_REDUCTION_RECIP_COMPLEMENT)
        // result = P(s) = core ratio f(sqrt(s))/sqrt(s). Reconstruct atan-style:
        //   |x|<=1: mag = |x| * P                 (recip_reduced = |x|)
        //   |x|>1 : mag = pi/2 - (1/|x|) * P      (recip_reduced = 1/|x|)
        // As |x|->inf, recip_reduced->0, P->1, core_value->0, mag->pi/2 EXACTLY,
        // so the reduction owns its own +-inf shoulders (no domain action). atan
        // is odd, so copysign restores the input sign. Host twin:
        // ttpoly/precision/rangered.py::_eval_reciprocal_complement (2.5 ULP,
        // 0 invalid over the full fp32 domain).
        {
            constexpr float RECIP_COMPLEMENT_PI_HALF = 1.5707963267948966f;
            vFloat core_value = recip_reduced * result;      // reduced * P(reduced^2)
            vFloat magnitude = core_value;
            v_if(setsgn(x_orig, 0) > 1.0f) {
                magnitude = RECIP_COMPLEMENT_PI_HALF - core_value;
            }
            v_endif;
            result = copysgn(magnitude, x_orig);
        }
#endif

#ifdef HAS_CRITICAL_POINT
        // Override with exact value at critical point for perfect accuracy
        // (e.g., cosh(0)=1, tanh(0)=0, sigmoid(0)=0.5)
        v_if(x == lut[CRITICAL_IDX]) { result = CRITICAL_VALUE; }
        v_endif;
#endif

        result = apply_output_postcompose(result);
        finalize_raw_domain_actions(x_raw, result);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        // The non-specialized loop also retains the untouched source DST row
        // until its final store. Keep raw-class terminals identical across
        // compact-loop, specialized, and structural evaluator selections.
        finalize_encoded_raw_domain_actions(
            dst_reg[d].mode<DataLayout::U16>(), result);
#endif

        // bf16 dst: RNE-round before the store. SFPSTORE narrows fp32->bf16 by
        // truncation (RTZ) in hardware; rounding here (sfpstochrnd RND_EVEN)
        // makes the already-bf16 value lossless under SFPSTORE and recovers the
        // half-ULP that RTZ would otherwise drop (the ML-pass output bias).
#ifdef USE_BF16
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
        dst_reg[d] = result;
    }
}

#ifdef USE_DUAL_EVAL
// Dual-evaluation version of piecewise_generic_lut.
// Processes two destination registers per iteration (d and d+1), exploiting
// SFPU instruction-level parallelism by interleaving two independent Horner chains.
//
// Segment 0 uses eval_polynomial_dual for full ILP benefit.
// Segments 1..N-1 use independent single-eval (x1 and x2 may be in different segments).
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dual(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    static_assert(
        LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
        "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Process pairs of destination registers (32 total → 16 iterations)
    for (int d = 0; d < 32; d += 2) {
        vFloat x_raw1 = dst_reg[d];
        vFloat x_raw2 = dst_reg[d + 1];
        vFloat x_orig1 = prepare_raw_domain_input(x_raw1);
        vFloat x_orig2 = prepare_raw_domain_input(x_raw2);

#if defined(RANGE_REDUCTION_EXP)
        // Anti-wrap sentinels only; exp_expand owns precise per-function
        // saturation via the reconstructed exponent field (see single-lane note).
        constexpr float EXP_OVERFLOW = 256.0f;
        constexpr float EXP_UNDERFLOW = -256.0f;
        vFloat x1, x2;
        vInt k_int1, k_int2;
        exp_reduce(x_orig1, x1, k_int1);
        exp_reduce(x_orig2, x2, k_int2);
#elif defined(RANGE_REDUCTION_TRIG)
        vFloat x1, x2;
        vInt q_int1, q_int2;
        trig_reduce(x_orig1, x1, q_int1);
        trig_reduce(x_orig2, x2, q_int2);
#elif defined(RANGE_REDUCTION_TAN)
        vFloat x1, x2;
        vInt j_int1, j_int2;
        tan_reduce(x_orig1, x1, j_int1);
        tan_reduce(x_orig2, x2, j_int2);
#elif defined(RANGE_REDUCTION_LOG)
        vFloat x1, x2;
        vInt e_int1, e_int2;
        log_reduce(x_orig1, x1, e_int1);
        log_reduce(x_orig2, x2, e_int2);
#elif defined(RANGE_REDUCTION_CBRT)
        // Extract BIASED exponent and sign BEFORE poly eval
        vInt cbrt_biased_e1 = exexp(setsgn(x_orig1, 0), ExponentMode::Biased);
        vInt cbrt_sign1 = as<vInt>(x_orig1) & 0x80000000;
        vInt cbrt_biased_e2 = exexp(setsgn(x_orig2, 0), ExponentMode::Biased);
        vInt cbrt_sign2 = as<vInt>(x_orig2) & 0x80000000;
        vFloat x1 = cbrt_reduce_m(x_orig1);
        vFloat x2 = cbrt_reduce_m(x_orig2);
#else
        vFloat x1 = x_orig1;
        vFloat x2 = x_orig2;
#endif

#if defined(BASIS_INPUT_ABS_X)
        x1 = setsgn(x_orig1, 0);
        x2 = setsgn(x_orig2, 0);
#endif

        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x1_clamped = x1;
        vFloat& x2_clamped = x2;

        // Segment 0: dual evaluation exploits ILP for interleaved Horner chains
        vFloat result1, result2;
        eval_polynomial_dual<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);

        // Segments 1..N-1: independent selection (x1 and x2 may be in different segments)
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if(x1_clamped >= lut[seg]) {
                result1 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x1);
            }
            v_endif;
            v_if(x2_clamped >= lut[seg]) {
                result2 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x2);
            }
            v_endif;
        }

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        result1 = result1 * x1;
        result2 = result2 * x2;
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
        result1 = result1 * basis_sqrt(vFloat(1.0f) - x1);
        result2 = result2 * basis_sqrt(vFloat(1.0f) - x2);
#endif

#if defined(BASIS_AFFINE_EVEN)
        result1 =
            sfpu_mad(BASIS_AFFINE_EVEN_SCALE * x1, result1, sfpu_mad(BASIS_AFFINE_SCALE, x_orig1, BASIS_AFFINE_BIAS));
        result2 =
            sfpu_mad(BASIS_AFFINE_EVEN_SCALE * x2, result2, sfpu_mad(BASIS_AFFINE_SCALE, x_orig2, BASIS_AFFINE_BIAS));
#endif

#if defined(BASIS_CLAMP_MAX)
        vFloat basis_clamp_max_value1 = BASIS_CLAMP_MAX_VALUE;
        vFloat basis_clamp_max_value2 = BASIS_CLAMP_MAX_VALUE;
        ordered_min_max(result1, basis_clamp_max_value1);
        ordered_min_max(result2, basis_clamp_max_value2);
#endif

#if defined(BASIS_POST_SIGN_X)
        result1 = copysgn(result1, x_orig1);
        result2 = copysgn(result2, x_orig2);
#endif

#if defined(BASIS_POST_REFLECT_PI)
        v_if(x_orig1 < 0.0f) { result1 = BASIS_PI_VALUE - result1; }
        v_endif;
        v_if(x_orig2 < 0.0f) { result2 = BASIS_PI_VALUE - result2; }
        v_endif;
#endif

#if defined(BASIS_LEFT_TAIL_ZERO)
        v_if(x_orig1 < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { result1 = 0.0f; }
        v_endif;
        v_if(x_orig2 < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { result2 = 0.0f; }
        v_endif;
#endif

#if defined(BASIS_RIGHT_TAIL_IDENTITY)
        v_if(x_orig1 > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { result1 = x_orig1; }
        v_endif;
        v_if(x_orig2 > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { result2 = x_orig2; }
        v_endif;
#endif

#if defined(RANGE_REDUCTION_EXP)
        v_if(x_orig1 > EXP_OVERFLOW) { result1 = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig1 < EXP_UNDERFLOW) { result1 = 0.0f; }
        v_else { result1 = exp_expand(result1, k_int1); }
        v_endif;
        v_if(x_orig2 > EXP_OVERFLOW) { result2 = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig2 < EXP_UNDERFLOW) { result2 = 0.0f; }
        v_else { result2 = exp_expand(result2, k_int2); }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        result1 = trig_expand(result1, q_int1);
        result2 = trig_expand(result2, q_int2);
#elif defined(RANGE_REDUCTION_TAN)
        result1 = tan_expand(result1, j_int1);
        result2 = tan_expand(result2, j_int2);
#elif defined(RANGE_REDUCTION_LOG)
        // Log expand: handle special cases and reconstruct log(x) = e*ln(2) + log(m)
        v_if(x_orig1 < 0.0f) { result1 = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig1 == 0.0f) { result1 = -std::numeric_limits<float>::infinity(); }
        v_else { result1 = log_expand(result1, e_int1); }
        v_endif;
        v_if(x_orig2 < 0.0f) { result2 = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig2 == 0.0f) { result2 = -std::numeric_limits<float>::infinity(); }
        v_else { result2 = log_expand(result2, e_int2); }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Cbrt expand using biased exponent — all arithmetic in float
        {
            constexpr float ONE_THIRD_C = 0.3333333333333333f;
            const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);

            // Result 1
            vFloat e_float1 = convert<vFloat>(as<vSMag>(cbrt_biased_e1), RoundMode::Nearest) - 127.0f;
            vFloat q_approx1 = e_float1 * ONE_THIRD_C;
            vFloat q_rounded1 = q_approx1 + magic;
            vInt q1 = as<vInt>(q_rounded1) - as<vInt>(magic);
            vFloat q_back1 = q_rounded1 - magic;
            vFloat r_float1 = e_float1 - (q_back1 + q_back1 + q_back1);
            vInt r1 = as<vInt>(r_float1 + magic) - as<vInt>(magic);
            v_if(r1 < 0) {
                q1 = q1 - 1;
                r1 = r1 + 3;
            }
            v_endif;
            v_if(r1 > 2) {
                q1 = q1 + 1;
                r1 = r1 - 3;
            }
            v_endif;
            v_if(cbrt_biased_e1 < 1) { result1 = 0.0f; }
            v_else { result1 = cbrt_expand(result1, q1, r1, cbrt_sign1); }
            v_endif;

            // Result 2
            vFloat e_float2 = convert<vFloat>(as<vSMag>(cbrt_biased_e2), RoundMode::Nearest) - 127.0f;
            vFloat q_approx2 = e_float2 * ONE_THIRD_C;
            vFloat q_rounded2 = q_approx2 + magic;
            vInt q2 = as<vInt>(q_rounded2) - as<vInt>(magic);
            vFloat q_back2 = q_rounded2 - magic;
            vFloat r_float2 = e_float2 - (q_back2 + q_back2 + q_back2);
            vInt r2 = as<vInt>(r_float2 + magic) - as<vInt>(magic);
            v_if(r2 < 0) {
                q2 = q2 - 1;
                r2 = r2 + 3;
            }
            v_endif;
            v_if(r2 > 2) {
                q2 = q2 + 1;
                r2 = r2 - 3;
            }
            v_endif;
            v_if(cbrt_biased_e2 < 1) { result2 = 0.0f; }
            v_else { result2 = cbrt_expand(result2, q2, r2, cbrt_sign2); }
            v_endif;
        }
#endif

#ifdef HAS_CRITICAL_POINT
        v_if(x1 == lut[CRITICAL_IDX]) { result1 = CRITICAL_VALUE; }
        v_endif;
        v_if(x2 == lut[CRITICAL_IDX]) { result2 = CRITICAL_VALUE; }
        v_endif;
#endif

        result1 = apply_output_postcompose(result1);
        result2 = apply_output_postcompose(result2);
        finalize_raw_domain_actions(x_raw1, result1);
        finalize_raw_domain_actions(x_raw2, result2);

        // bf16 dst: RNE-round before the store (see single-eval note above).
#ifdef USE_BF16
        result1 = convert<vFloat16b>(result1, RoundMode::Nearest);
        result2 = convert<vFloat16b>(result2, RoundMode::Nearest);
#endif
        dst_reg[d] = result1;
        dst_reg[d + 1] = result2;
    }
}
#endif  // USE_DUAL_EVAL

#if EVAL_METHOD_IS_STANDALONE
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
#ifdef USE_DUAL_EVAL
    piecewise_generic_lut_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#else
    piecewise_generic_lut<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#endif
}
#endif

// ----------------------------------------------------------------------------
// TTI REPLAY lowering for the SINGLE-SEGMENT polynomial cascade.
//
// Three supported shapes (everything else falls back to the sfpi dispatch):
//   PLAIN       y = P(x)                        (e.g. expm1 p7)
//   SIGNED_ABS  y = copysgn(min?(P(|x|), CM), x) (signed_abs_poly basis:
//               erf p7 / tanh p6 / atan p8; optional BASIS_CLAMP_MAX == 1.0)
//   PARITY_EVEN y = H(x*x), the POLY_PARITY_EVEN x^2-Horner over the even
//               coefficients only (e.g. cosh p16: 9 coefficients, 8 rungs)
//
// Interleave choice, PLAIN: 2-element interleave — the body is pure Horner, so
// interleave supplies every MAD gap for free and halves the replay count.
// Body = 2*(LOAD + DEG MADs + STOCH_RND + STORE) = 2*DEG+6 slots (p7 = 20).
// Coefficient budget: prgm1/2 = c[D], c[D-1]; prgm0 = c[D-2]; per-tile pinned
// LREG7..LREG4 = c[D-3]..c[D-6]; c0 == 0 rides the hw zero (L9). Lanes use
// L0/L2 (A: x, acc) and L1/L3 (B) — gate: DEG <= 6, or DEG == 7 with c0 == 0.
//
// Interleave choice, SIGNED_ABS: SINGLE element. Each lane needs THREE live
// registers (x for the final sign restore, |x| for the Horner, acc) so a
// 2-element interleave leaves only L6/L7 + prgm for coefficients (5 < DEG+1
// for every selected config) — the register file, not the 32-slot capacity,
// forces 1x. Gaps: NOPs (the shape has no independent side-computation); at
// DEG 8 the c1 SFPLOADI pair doubles as one gap. c0 must be 0 (odd basis).
// Body slots = 3 + 1 + 2*(DEG-1) + (DEG==8) + (clamp ? 2 : 0) + 4:
//   erf p7+clamp = 22, tanh p6+clamp = 20, atan p8 = 23.
//
// Byte-identity: same op kinds / order / single-rounding MADs as the sfpi
// eval_polynomial dual loop (head LOADI+MAD == MAD from a register: identical
// fp32 result); abs == the same SFPSETSGN imm-0 form; clamp == the same
// SFPSWAP-vs-LCONST_1 vec_min_max; sign restore == the same SFPSETSGN mod-0.
// PARITY_EVEN: same s = x*x single MUL and the same stride-2 fma chain as
// eval_polynomial_parity (odd-index LUT residue skipped identically).
// POLY_TTI_DISABLE (env -> codegen) forces the sfpi path for A/B.
#if defined(EVAL_METHOD_POLY_CASCADE) && defined(EMBEDDED_LUT) &&                         \
    ((defined(USE_BF16) &&                                                               \
      !defined(TT_INLINE_PROGRAM_TTI_FACTORIZED_ODD_UNIT_ORIGIN_REASSOCIATION)) ||       \
     (!defined(USE_BF16) &&                                                              \
      defined(TT_INLINE_PROGRAM_TTI_FACTORIZED_ODD_UNIT_ORIGIN_REASSOCIATION))) &&       \
    (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && !defined(POLY_TTI_DISABLE) && \
    !defined(RANGE_REDUCTION_EXP) && !defined(RANGE_REDUCTION_TRIG) &&                      \
    !defined(RANGE_REDUCTION_TAN) && !defined(RANGE_REDUCTION_LOG) &&                       \
    !defined(RANGE_REDUCTION_CBRT) && !defined(POLY_PARITY_ODD) &&                          \
    !defined(HAS_CRITICAL_POINT) &&                                                         \
    !defined(POSTCOMPOSE_AFFINE_Y) && !defined(ASYMPTOTIC_FACTOR_QUADRATIC) &&              \
    !defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) && !defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) &&  \
    !defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) && !defined(BASIS_MUL_SQRT_1_MINUS_ABS) &&     \
    !defined(BASIS_POST_REFLECT_PI) && !defined(BASIS_AFFINE_EVEN) &&                       \
    !defined(BASIS_MUL_ABS_X_BEFORE_POST) && !defined(BASIS_LEFT_TAIL_ZERO) &&              \
    !defined(BASIS_RIGHT_TAIL_IDENTITY) &&                                      \
    (!defined(BASIS_RIGHT_TAIL_ABS_AFFINE) ||                                   \
     defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)) &&                       \
    (!defined(TT_DOMAIN_ACTION_PROGRAM) ||                                      \
     defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY))
#define POLY_TTI_CANDIDATE 1
#if defined(TT_INLINE_PROGRAM_TTI_FACTORIZED_ODD_UNIT_ORIGIN_REASSOCIATION)
// FP32 selected-core shape y=x*P(x^2), with exact P(0)=1. The certified
// reassociation evaluates Q from the immutable c1..cd suffix and forms
// x+(x*s)*Q. A manual replay owns x/s/Q in L0/L1/L2, so the SFPI compiler
// cannot CSE away the coordinate as it did in every high-level spelling.
#define POLY_TTI_SHAPE_FACTORIZED_UNIT 1
#define POLY_TTI_SHAPE_PARITY_EVEN 0
#define POLY_TTI_SHAPE_SIGNED_ABS 0
#define POLY_TTI_SHAPE_PLAIN 0
#define POLY_TTI_SHAPE_LOG_SQUARE 0
#elif defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD)
#define POLY_TTI_SHAPE_LOG_SQUARE 1
#define POLY_TTI_SHAPE_FACTORIZED_UNIT 0
#define POLY_TTI_SHAPE_PARITY_EVEN 0
#define POLY_TTI_SHAPE_SIGNED_ABS 0
#define POLY_TTI_SHAPE_PLAIN 0
#elif defined(POLY_PARITY_EVEN)
// Parity-even x^2-Horner shape: no abs/sign/clamp basis (parity inference is
// disabled for those upstream in the codegen, but keep the guard explicit).
#if !defined(BASIS_INPUT_ABS_X) && !defined(BASIS_POST_SIGN_X) && !defined(BASIS_CLAMP_MAX)
#define POLY_TTI_SHAPE_PARITY_EVEN 1
#else
#define POLY_TTI_SHAPE_PARITY_EVEN 0
#endif
#define POLY_TTI_SHAPE_SIGNED_ABS 0
#define POLY_TTI_SHAPE_PLAIN 0
#define POLY_TTI_SHAPE_FACTORIZED_UNIT 0
#define POLY_TTI_SHAPE_LOG_SQUARE 0
#elif defined(BASIS_INPUT_ABS_X) && defined(BASIS_POST_SIGN_X)
#define POLY_TTI_SHAPE_SIGNED_ABS 1
#define POLY_TTI_SHAPE_PLAIN 0
#define POLY_TTI_SHAPE_PARITY_EVEN 0
#define POLY_TTI_SHAPE_FACTORIZED_UNIT 0
#define POLY_TTI_SHAPE_LOG_SQUARE 0
#elif !defined(BASIS_INPUT_ABS_X) && !defined(BASIS_POST_SIGN_X) && !defined(BASIS_CLAMP_MAX)
#define POLY_TTI_SHAPE_SIGNED_ABS 0
#define POLY_TTI_SHAPE_PLAIN 1
#define POLY_TTI_SHAPE_PARITY_EVEN 0
#define POLY_TTI_SHAPE_FACTORIZED_UNIT 0
#define POLY_TTI_SHAPE_LOG_SQUARE 0
#else
#define POLY_TTI_SHAPE_SIGNED_ABS 0
#define POLY_TTI_SHAPE_PLAIN 0
#define POLY_TTI_SHAPE_PARITY_EVEN 0
#define POLY_TTI_SHAPE_FACTORIZED_UNIT 0
#define POLY_TTI_SHAPE_LOG_SQUARE 0
#endif

constexpr uint32_t kPolyTtiCoeffOff = NUM_SEGMENTS + 1;  // boundary points first
constexpr float poly_tti_c(uint32_t k) { return LUT_DATA[kPolyTtiCoeffOff + k]; }
constexpr bool kPolyTtiC0Zero = (poly_tti_c(0) == 0.0f);
#if defined(BASIS_CLAMP_MAX)
constexpr bool kPolyTtiClamp = true;
constexpr bool kPolyTtiClampIsOne = ((float)(BASIS_CLAMP_MAX_VALUE) == 1.0f);
#else
constexpr bool kPolyTtiClamp = false;
constexpr bool kPolyTtiClampIsOne = true;
#endif

constexpr uint32_t kPolyTtiPlainBodySlots = 2u * POLY_DEGREE + 6u;
constexpr uint32_t kPolyTtiAbsBodySlots =
    3u + 1u + 2u * (POLY_DEGREE - 1u) + ((POLY_DEGREE == 8u) ? 1u : 0u) +
    (kPolyTtiClamp ? 2u : 0u) + 4u;

// PARITY_EVEN 2x shape: highest even coefficient index TOP, STEPS = TOP/2 MADs
// per element (head absorbs c[TOP] and c[TOP-2]; STEPS-1 rungs follow). The
// coefficient budget mirrors PLAIN — prgm1/2 = c[TOP], c[TOP-2]; prgm0 = first
// rung addend; per-tile pins L7/L6/L5 for rungs 1-3 — but rungs past that use
// SELF-CONTAINED in-body SFPLOADI pairs into L4 (a per-tile L4 pin would be
// stale on replay iterations 2..16 once the body overwrites it). Body =
// 2 LOAD + 2 MUL(s=x*x) + 2*STEPS MADs + 2*max(0, STEPS-5) LOADI + 2 RND +
// 2 STORE. cosh p16: STEPS=8 -> 30 slots.
constexpr uint32_t kPolyTtiParityTop = (POLY_DEGREE % 2u == 0u) ? POLY_DEGREE : POLY_DEGREE - 1u;
constexpr uint32_t kPolyTtiParitySteps = kPolyTtiParityTop / 2u;
constexpr uint32_t kPolyTtiParityInBody =
    (kPolyTtiParitySteps > 5u) ? (kPolyTtiParitySteps - 5u) : 0u;
constexpr uint32_t kPolyTtiParityBodySlots =
    4u + 2u * kPolyTtiParitySteps + 2u * kPolyTtiParityInBody + 4u;

constexpr bool kPolyTtiPlain2x = POLY_TTI_SHAPE_PLAIN && (NUM_SEGMENTS == 1) &&
                                 (POLY_DEGREE >= 2) &&
                                 ((POLY_DEGREE <= 6) || (POLY_DEGREE == 7 && kPolyTtiC0Zero)) &&
                                 tti_replay_fits(kPolyTtiPlainBodySlots);
constexpr bool kPolyTtiAbs1x = POLY_TTI_SHAPE_SIGNED_ABS && (NUM_SEGMENTS == 1) &&
                               (POLY_DEGREE >= 2) && (POLY_DEGREE <= 8) && kPolyTtiC0Zero &&
                               kPolyTtiClampIsOne && tti_replay_fits(kPolyTtiAbsBodySlots);
constexpr bool kPolyTtiParityEven2x = POLY_TTI_SHAPE_PARITY_EVEN && (NUM_SEGMENTS == 1) &&
                                      (kPolyTtiParitySteps >= 2u) && (kPolyTtiParitySteps <= 8u) &&
                                      tti_replay_fits(kPolyTtiParityBodySlots);
#if defined(TT_INLINE_PROGRAM_TTI_FACTORIZED_ODD_UNIT_ORIGIN_REASSOCIATION)
constexpr uint32_t kPolyTtiFactorizedUnitBodySlots =
    TT_PROGRAM_FACTORIZED_ODD_ISSUE_LOWER_BOUND;
constexpr bool kPolyTtiFactorizedUnit1x =
    POLY_TTI_SHAPE_FACTORIZED_UNIT && (NUM_SEGMENTS == 1u) &&
    (POLY_DEGREE >= 2u) && (POLY_DEGREE <= 11u) &&
    (__builtin_bit_cast(uint32_t, poly_tti_c(0u)) == 0x3f800000u) &&
    (TT_PROGRAM_FACTORIZED_ODD_DEGREE == POLY_DEGREE) &&
    (TT_PROGRAM_FACTORIZED_ODD_DIRECT_ARITHMETIC_ISSUES == POLY_DEGREE + 2u) &&
    (TT_PROGRAM_FACTORIZED_ODD_REASSOCIATED_ARITHMETIC_ISSUES == POLY_DEGREE + 2u) &&
    (TT_PROGRAM_FACTORIZED_ODD_TTI_PEAK_LIVE == 4u) &&
    (kPolyTtiFactorizedUnitBodySlots == 2u * POLY_DEGREE + 10u) &&
    tti_replay_fits(kPolyTtiFactorizedUnitBodySlots);
#else
constexpr bool kPolyTtiFactorizedUnit1x = false;
#endif
#if defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD)
constexpr bool kPolyTtiLogSquare1x =
    POLY_TTI_SHAPE_LOG_SQUARE && (NUM_SEGMENTS == 1u) &&
    (POLY_DEGREE == TT_SELECTED_LOG_SQUARE_OUTER_DEGREE) &&
    (POLY_DEGREE == 3u || POLY_DEGREE == 4u) &&
    (TT_SELECTED_LOG_SQUARE_BODY_SLOTS == 31u) &&
    (TT_SELECTED_LOG_SQUARE_RAW_PATTERN_COUNT == 65536u) &&
    tti_replay_fits(TT_SELECTED_LOG_SQUARE_BODY_SLOTS);
#else
constexpr bool kPolyTtiLogSquare1x = false;
#endif
#if defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION)
#if defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_INTRINSIC_RAW_CLOSURE) && \
    defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
#error "intrinsic square-affine raw closure cannot retain an encoded suffix"
#endif
constexpr bool kPolyTtiSquareAffine1x =
    POLY_TTI_SHAPE_PLAIN && (NUM_SEGMENTS == 1) &&
    (POLY_DEGREE == 8u) &&
    (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_SCALE_BITS == 0x3f800000u) &&
    (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BIAS_BITS == 0x3f000000u) &&
    (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RAW_EQUAL_MASK == 0x00ffu) &&
    (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RAW_EQUAL_VALUE == 0x00ffu) &&
    (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RAW_NONZERO_MASK == 0xff00u) &&
    (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_REPAIRED_PATTERNS == 255u) &&
#if defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_INTRINSIC_RAW_CLOSURE)
    true &&
#endif
    (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BODY_SLOTS == 28u) &&
    tti_replay_fits(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BODY_SLOTS);
#else
constexpr bool kPolyTtiSquareAffine1x = false;
#endif
#if defined(TT_SELECTED_CORE_TTI_SCALED_SQUARE_RECONSTRUCTION)
constexpr bool kPolyTtiScaledSquare1x =
    POLY_TTI_SHAPE_PLAIN && (NUM_SEGMENTS == 1) &&
    (POLY_DEGREE == 9u) &&
    (TT_SELECTED_CORE_TTI_SCALED_SQUARE_BIAS_BITS == 0xbf800000u) &&
    (TT_SELECTED_CORE_TTI_SCALED_SQUARE_BODY_SLOTS == 31u) &&
    tti_replay_fits(TT_SELECTED_CORE_TTI_SCALED_SQUARE_BODY_SLOTS);
#else
constexpr bool kPolyTtiScaledSquare1x = false;
#endif

// Programmable-constant allocation is part of replay closure, not merely an
// init detail. A typed suffix can structurally require sfpu reciprocal after
// the selected evaluator has run. On BH that primitive owns Prgm0/L12; on WH
// it owns Prgm0/1/2. sfpu_reciprocal_init is deliberately the last writer of
// those rows, so admit replay only when its actual coefficient allocation is
// disjoint. This is descriptor/shape based: it neither knows nor needs the
// activation name or coefficient values.
constexpr uint32_t kPolyTtiPrgm0Mask = 1u << 0;
constexpr uint32_t kPolyTtiPrgm1Mask = 1u << 1;
constexpr uint32_t kPolyTtiPrgm2Mask = 1u << 2;
constexpr uint32_t kPolyTtiPrgm12Mask =
    kPolyTtiPrgm1Mask | kPolyTtiPrgm2Mask;

// Every admitted body keeps its Horner head in Prgm1/2. Most bodies also
// keep their first below-head addend in Prgm0. The square-affine schedule
// streams that addend into L3, while the signed-abs reciprocal schedule
// streams its displaced addends into L3; neither allocates Prgm0.
#if POLY_TTI_SHAPE_LOG_SQUARE || POLY_TTI_SHAPE_PARITY_EVEN
constexpr uint32_t kPolyTtiPrgmWriteMask =
    kPolyTtiPrgm12Mask | kPolyTtiPrgm0Mask;
#elif defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION) || \
    defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
constexpr uint32_t kPolyTtiPrgmWriteMask = kPolyTtiPrgm12Mask;
#else
constexpr bool kPolyTtiFirstAddendIsHardwareZero =
    (POLY_DEGREE == 2u) && kPolyTtiC0Zero;
constexpr uint32_t kPolyTtiPrgmWriteMask =
    kPolyTtiPrgm12Mask |
    (kPolyTtiFirstAddendIsHardwareZero ? 0u : kPolyTtiPrgm0Mask);
#endif

#if defined(TT_INLINE_PROGRAM_NEEDS_RECIPROCAL) || \
    defined(TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL) || \
    defined(TT_SELECTED_CORE_NEEDS_RECIPROCAL)
#if defined(ARCH_WORMHOLE)
constexpr uint32_t kPolyTtiPrgmOwnerMask =
    kPolyTtiPrgm0Mask | kPolyTtiPrgm1Mask | kPolyTtiPrgm2Mask;
#else
constexpr uint32_t kPolyTtiPrgmOwnerMask = kPolyTtiPrgm0Mask;
#endif
#else
constexpr uint32_t kPolyTtiPrgmOwnerMask = 0u;
#endif

constexpr bool kPolyTtiProgrammableConstantOwnershipClosed =
    (kPolyTtiPrgmWriteMask & kPolyTtiPrgmOwnerMask) == 0u;
constexpr bool kPolyTtiCapacityClosed =
    kPolyTtiPlain2x || kPolyTtiAbs1x || kPolyTtiParityEven2x ||
    kPolyTtiSquareAffine1x || kPolyTtiFactorizedUnit1x ||
    kPolyTtiScaledSquare1x || kPolyTtiLogSquare1x;
constexpr bool kPolyTtiReplay =
    kPolyTtiCapacityClosed && kPolyTtiProgrammableConstantOwnershipClosed;
#if defined(TT_INLINE_PROGRAM_OVERLAY) || \
    defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION) || \
    defined(TT_SELECTED_CORE_TTI_SCALED_SQUARE_RECONSTRUCTION) || \
    defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY) || \
    defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL) || \
    defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD)
#if defined(TT_INLINE_PROGRAM_TTI_FACTORIZED_ODD_UNIT_ORIGIN_REASSOCIATION)
static_assert(
    kPolyTtiFactorizedUnit1x,
    "typed exact-unit factorized reconstruction exceeds P2..P11 replay closure");
#elif defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD)
static_assert(
    kPolyTtiLogSquare1x,
    "typed log-square factorized reconstruction exceeds P3/P4 replay closure");
#elif defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION)
static_assert(
    kPolyTtiSquareAffine1x,
    "typed square-affine reconstruction exceeds P8 replay closure");
#elif defined(TT_SELECTED_CORE_TTI_SCALED_SQUARE_RECONSTRUCTION)
static_assert(
    kPolyTtiScaledSquare1x,
    "typed scaled-square reconstruction exceeds P9 replay closure");
#elif defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
// Verifier-bound affine endpoint continuation for one signed-abs P8 leaf.
// The selected Horner stays byte/order-identical.  The body clamps only its
// magnitude coordinate, forms the typed affine magnitude after c2's L4 dies,
// and selects its certified max continuation before the unchanged
// sign/RNE/store LOADMACRO tail.
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
constexpr uint32_t kPolyTtiSignedAbsAffineBodySlots = 32u;
constexpr uint32_t kPolyTtiSignedAbsAffineLmSequenceBits = 0x13850000u;
#else
constexpr uint32_t kPolyTtiSignedAbsAffineBodySlots = 28u;
constexpr uint32_t kPolyTtiSignedAbsAffineLmSequenceBits = 0x63550087u;
#endif
constexpr bool kPolyTtiSignedAbsAffineDescriptor =
    (TT_TYPED_TTI_ABS_AFFINE_SCALE_BITS == 0x3f800000u) &&
    (TT_TYPED_TTI_ABS_AFFINE_RIGHT_BIAS_BITS == 0xbf800000u) &&
    (TT_TYPED_TTI_ABS_AFFINE_LEFT_BIAS_BITS == 0x3f800000u) &&
    (TT_TYPED_TTI_ABS_AFFINE_BODY_SLOTS ==
     kPolyTtiSignedAbsAffineBodySlots) &&
    (TT_TYPED_TTI_ABS_AFFINE_LM_SEQUENCE_BITS ==
     kPolyTtiSignedAbsAffineLmSequenceBits) &&
    (__builtin_bit_cast(uint32_t, (float)(BASIS_RIGHT_TAIL_ABS_THRESHOLD)) ==
     TT_TYPED_TTI_ABS_AFFINE_CORE_BOUND_BITS) &&
    (__builtin_bit_cast(uint32_t, (float)(BASIS_RIGHT_TAIL_ABS_A)) ==
     TT_TYPED_TTI_ABS_AFFINE_SCALE_BITS) &&
    (__builtin_bit_cast(uint32_t, (float)(BASIS_RIGHT_TAIL_ABS_B)) ==
     TT_TYPED_TTI_ABS_AFFINE_RIGHT_BIAS_BITS);
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
constexpr bool kPolyTtiSignedAbsAffineRawIngressDescriptor =
    (TT_TYPED_TTI_ABS_AFFINE_RAW_EQUAL_VALUE == 0x80ffu) &&
    (TT_TYPED_TTI_ABS_AFFINE_REPAIRED_PATTERNS == 127u) &&
    (TT_TYPED_TTI_ABS_AFFINE_LM_SEQUENCE_BITS == 0x13850000u) &&
    (__builtin_bit_cast(uint32_t, poly_tti_c(0u)) == 0u) &&
    (__builtin_bit_cast(uint32_t, poly_tti_c(1u)) == 0u) &&
    (__builtin_bit_cast(uint32_t, poly_tti_c(2u)) == 0u);
#else
constexpr bool kPolyTtiSignedAbsAffineRawIngressDescriptor = true;
#endif
constexpr bool kPolyTtiSignedAbsAffineReplay =
    kPolyTtiReplay && POLY_TTI_SHAPE_SIGNED_ABS &&
    (POLY_DEGREE == 8u) && !kPolyTtiClamp &&
    kPolyTtiSignedAbsAffineDescriptor &&
    kPolyTtiSignedAbsAffineRawIngressDescriptor &&
    tti_replay_fits(kPolyTtiSignedAbsAffineBodySlots);
static_assert(
    !kPolyTtiReplay || kPolyTtiSignedAbsAffineReplay,
    "typed signed-abs affine overlay exceeds replay/register closure");
#elif defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
// Verifier-bound reciprocal/complement fold for a signed-abs P8 leaf.  The
// exact selected Horner stays in this body; only its coordinate and typed
// post-reconstruction are added.  A LOADMACRO schedules the unchanged final
// SETSGN/RNE/STORE chain and recovers the two slots needed for exact 32-slot
// closure.  See the body for the complete CC/macro non-overlap schedule.
constexpr uint32_t kPolyTtiSignedAbsRecipBodySlots = 32u;
constexpr bool kPolyTtiSignedAbsRecipReplay =
    kPolyTtiReplay && POLY_TTI_SHAPE_SIGNED_ABS &&
    (POLY_DEGREE == 8u) && !kPolyTtiClamp &&
    (TT_SELECTED_TTI_RECIP_CORE_BOUND_BITS == 0x3f800000u) &&
    (TT_SELECTED_TTI_RECIP_BODY_SLOTS == kPolyTtiSignedAbsRecipBodySlots) &&
    (TT_SELECTED_TTI_RECIP_LM_SEQUENCE_BITS == 0x63550087u) &&
    tti_replay_fits(kPolyTtiSignedAbsRecipBodySlots);
static_assert(
    kPolyTtiProgrammableConstantOwnershipClosed,
    "typed reciprocal replay has no disjoint programmable-constant allocation");
#if defined(TT_SELECTED_TTI_RAW_SHADOW_LATE_SCHEDULE)
#if !defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION) && \
    !defined(TT_SELECTED_CORE_TTI_SCALED_SQUARE_RECONSTRUCTION)
static_assert(
    kPolyTtiSignedAbsRecipReplay,
    "TTI raw shadow requires a replay with an explicit late terminal suffix");
#endif
static_assert(
    TT_SELECTED_TTI_RAW_SHADOW_ROWS == 32u &&
        TT_SELECTED_TTI_RAW_SHADOW_ROW_BASE +
                TT_SELECTED_TTI_RAW_SHADOW_ROWS <=
            TT_SELECTED_TTI_RAW_SHADOW_DST_CAPACITY,
    "TTI raw shadow exceeds the verifier-bound destination capacity");
static_assert(
    TT_SELECTED_TTI_RAW_SHADOW_PRESERVE_PEAK_LIVE <= 8u &&
        TT_SELECTED_TTI_RAW_SHADOW_SUFFIX_PEAK_LIVE <= 8u,
    "TTI raw shadow or terminal suffix exceeds SFPU liveness capacity");
#endif
static_assert(
    !kPolyTtiReplay || kPolyTtiSignedAbsRecipReplay,
    "typed signed-abs reciprocal overlay exceeds replay/register closure");
#elif defined(TT_INLINE_PROGRAM_TTI_SYMMETRIC_SATURATION_OVERLAY)
// A verifier-certified two-sided fold for signed-abs bases: typed tails are
// exactly the target-rounded value already produced by the mature
// clamp/sign/RNE reconstruction at the core endpoint.  Only the evaluator
// coordinate is clamped; its Horner and reconstruction instructions remain
// unchanged.  L2 is dead before the Horner head, so it carries the generated
// absolute core bound without consuming a coefficient register.
constexpr uint32_t kPolyTtiSymmetricOverlayBodySlots =
    kPolyTtiAbsBodySlots + 4u;  // full-fp32 bound, magnitude min, hazard gap
constexpr bool kPolyTtiSymmetricOverlayReplay =
    kPolyTtiReplay && POLY_TTI_SHAPE_SIGNED_ABS &&
    tti_replay_fits(kPolyTtiSymmetricOverlayBodySlots);
static_assert(
    !kPolyTtiReplay || kPolyTtiSymmetricOverlayReplay,
    "typed symmetric POLY_TTI overlay exceeds replay/register closure or has an unsupported basis");
#elif defined(ARCH_BLACKHOLE) && \
    defined(TT_INLINE_PROGRAM_TTI_COMMON_SFPU_BITWISE_INGRESS_CLOSURE)
// Verifier-bound physical ingress normalization for a mature signed-abs P7
// replay.  The selected coefficient transport, Horner, clamp, sign, RNE, and
// store remain the established one-loop path.
constexpr bool kPolyTtiCommonSfpuBitwiseIngressClosure =
    kPolyTtiReplay && POLY_TTI_SHAPE_SIGNED_ABS && kPolyTtiClamp &&
    kPolyTtiClampIsOne && (POLY_DEGREE == 7u) &&
    (TT_PROGRAM_TTI_COMMON_SFPU_BASE_SLOTS == 22u) &&
    (TT_PROGRAM_TTI_COMMON_SFPU_BODY_SLOTS == 30u) &&
    (TT_PROGRAM_TTI_COMMON_SFPU_REPAIRED_PATTERNS == 127u) &&
    (TT_PROGRAM_TTI_RAW_EQUAL_VALUE == 0x80ffu) &&
    tti_replay_fits(TT_PROGRAM_TTI_COMMON_SFPU_BODY_SLOTS);
static_assert(
    !kPolyTtiReplay || kPolyTtiCommonSfpuBitwiseIngressClosure,
    "common-SFPU bitwise ingress certificate exceeds signed-abs replay closure");
#elif defined(TT_INLINE_PROGRAM_TTI_INTRINSIC_TERMINAL_CLOSURE)
// Verifier exhaustively proved that the unchanged selected Horner plus its
// mature clamp/sign/RNE finalizer already equals the typed terminal Program
// on every raw BF16 encoding after Float16_b ingress.  This descriptor emits
// no body instruction: it retains the naked selected replay as the performance
// ABI.
constexpr bool kPolyTtiIntrinsicTerminalClosure =
    kPolyTtiReplay && POLY_TTI_SHAPE_SIGNED_ABS && kPolyTtiClamp &&
    kPolyTtiClampIsOne &&
    (TT_PROGRAM_TTI_INTRINSIC_TERMINAL_SLOTS == 0u);
#if defined(ARCH_BLACKHOLE)
static_assert(
    POLY_DEGREE != 7u,
    "BH signed-abs P7 requires a verifier-issued bitwise ingress certificate");
#endif
static_assert(
    !kPolyTtiReplay || kPolyTtiIntrinsicTerminalClosure,
    "intrinsic terminal certificate does not match the mature selected finalizer");
#elif !defined(TT_INLINE_PROGRAM_TTI_ENDPOINT_CONTINUATION_OVERLAY)
static_assert(
    !kPolyTtiReplay,
    "typed core overlay has no verifier-certified POLY_TTI endpoint descriptor; refusing silent fallback");
#else
// Verifier-owned endpoint fold for a plain, two-vector, single-segment Horner
// body.  Kind 0 clamps only the evaluator coordinate: target-rounded P(B) is
// the typed constant tail.  Kinds 1/2 additionally form the exact certified
// graph r=fma(-1,B,x), r=max(r,0), y=fma(a,r,P(min(x,B))).  No lattice compare,
// CC window, branch, second loop, raw-lane postselect, or replacement evaluator
// is introduced; the selected CSV Horner subsequence below stays unchanged.
constexpr bool kPolyTtiEndpointConstant = (TT_PROGRAM_TTI_ENDPOINT_KIND == 0u);
constexpr bool kPolyTtiEndpointHardwareScale =
    (TT_PROGRAM_TTI_ENDPOINT_SCALE_BITS == 0x00000000u) ||
    (TT_PROGRAM_TTI_ENDPOINT_SCALE_BITS == 0x3f800000u) ||
    (TT_PROGRAM_TTI_ENDPOINT_SCALE_BITS == 0xbf800000u);
constexpr uint32_t kPolyTtiEndpointBodySlots =
    kPolyTtiPlainBodySlots + (kPolyTtiEndpointConstant ? 2u : 8u);
constexpr bool kPolyTtiEndpointRegisterClosure =
    POLY_TTI_SHAPE_PLAIN && (POLY_DEGREE >= 2u) &&
    (kPolyTtiEndpointConstant ||
     (kPolyTtiEndpointHardwareScale && POLY_DEGREE <= 4u) ||
     (!kPolyTtiEndpointHardwareScale && POLY_DEGREE <= 3u));
// B=1 is an ISA/register capability gate, not an activation-domain fact.
// LCONST_1 supplies both min and residual-FMA operands without displacing any
// selected-CSV coefficient. Other unsupported bounds fail during typed lowering.
constexpr bool kPolyTtiEndpointDescriptorClosure =
    (TT_PROGRAM_TTI_ENDPOINT_KIND <= 2u) &&
    (TT_PROGRAM_TTI_ENDPOINT_CORE_BOUND_BITS == 0x3f800000u) &&
    (TT_PROGRAM_TTI_ENDPOINT_CORE_BF16_BITS ==
     TT_PROGRAM_TTI_ENDPOINT_TAIL_BF16_BITS);
constexpr bool kPolyTtiEndpointReplay =
    kPolyTtiReplay && kPolyTtiEndpointRegisterClosure &&
    kPolyTtiEndpointDescriptorClosure &&
    tti_replay_fits(kPolyTtiEndpointBodySlots);
static_assert(
    !kPolyTtiReplay || kPolyTtiEndpointReplay,
    "certified POLY_TTI endpoint fold exceeds replay/register closure or has an unsupported descriptor");
#endif
#endif

// Below-head Horner addend register for 0-based rung index J (addend c[D-2-J]):
// prgm0 first, then the per-tile pinned L7..L4, then the in-body scratch L3
// (signed-abs DEG 8 only), with c0 == 0 riding the hw zero L9.
constexpr uint32_t poly_tti_addend_reg(uint32_t j) {
    const uint32_t power = POLY_DEGREE - 2u - j;
    // The raw-ingress closure is verifier-bound to exact +0 c2/c1/c0, so all
    // three addends may use the hardware zero register without changing the
    // Horner FMA or its ordering.  Keep this capacity optimization scoped to
    // that descriptor: the ordinary P8 replay slot count still includes the
    // streamed c1 immediate.  Outside the closure preserve the prior c0-only
    // rule (including its historical -0 treatment).
    if (power == 0u && kPolyTtiC0Zero) {
        return 9;  // LCONST_0
    }
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
    if (__builtin_bit_cast(uint32_t, poly_tti_c(power)) == 0u) {
        return 9;  // LCONST_0
    }
#endif
#if defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
    if (j == 0u || j == 4u) {
        // Prgm0 is the reciprocal's live 2.0 Newton constant.  The displaced
        // frozen c6 is loaded in-body into L3.  L4 carries the complement, so
        // c2 is split-loaded into L3 through the earlier J1/J2 Horner gaps.
        // The ordinary c1 load overwrites it only after J4 consumes c2.
        return 3;
    }
#endif
    switch (j) {
        case 0: return 12;  // prgm0
        case 1: return 7;
        case 2: return 6;
        case 3: return 5;
        case 4: return 4;
        default: return 3;  // in-body SFPLOADI scratch (abs shape, DEG 8)
    }
}
constexpr uint32_t poly_tti_addend_bits(uint32_t j) {
    return __builtin_bit_cast(uint32_t, poly_tti_c(POLY_DEGREE - 2u - j));
}

// Per-tile pinned-coefficient prelude shared by both shapes: L7..L4 carry the
// rung-1..4 addends when those rungs exist and are not L9/L3-sourced.
inline void poly_tti_pin_coeffs() {
    if constexpr (POLY_DEGREE >= 3 && poly_tti_addend_reg(1) == 7) {
        TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, poly_tti_addend_bits(1));
    }
    if constexpr (POLY_DEGREE >= 4 && poly_tti_addend_reg(2) == 6) {
        TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG6, poly_tti_addend_bits(2));
    }
    if constexpr (POLY_DEGREE >= 5 && poly_tti_addend_reg(3) == 5) {
        TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG5, poly_tti_addend_bits(3));
    }
    if constexpr (POLY_DEGREE >= 6 && poly_tti_addend_reg(4) == 4) {
        TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, poly_tti_addend_bits(4));
    }
#if defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
    // c2 is streamed through L3 in-body; L4 remains the exact complement for
    // every replay iteration and is never overwritten by the body.
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG4,
        TT_SELECTED_TTI_RECIP_COMPLEMENT_BITS);
#endif
}

#if POLY_TTI_SHAPE_FACTORIZED_UNIT
inline void poly_tti_pin_factorized_unit_coeffs() {
    // Q(s)=cD*s^(D-1)+...+c1. Prgm1/2/0 own cD/cD-1/cD-2;
    // L7..L4 own the next four addends. Degrees 8..11 stream any remaining
    // low addends through L3 after the fixed-register rungs have retired.
    if constexpr (POLY_DEGREE >= 4u) {
        TT_REPLAY_PIN_FP32(
            ckernel::p_sfpu::LREG7,
            poly_tti_addend_bits(1u));
    }
    if constexpr (POLY_DEGREE >= 5u) {
        TT_REPLAY_PIN_FP32(
            ckernel::p_sfpu::LREG6,
            poly_tti_addend_bits(2u));
    }
    if constexpr (POLY_DEGREE >= 6u) {
        TT_REPLAY_PIN_FP32(
            ckernel::p_sfpu::LREG5,
            poly_tti_addend_bits(3u));
    }
    if constexpr (POLY_DEGREE >= 7u) {
        TT_REPLAY_PIN_FP32(
            ckernel::p_sfpu::LREG4,
            poly_tti_addend_bits(4u));
    }
}

template <uint32_t J>
inline void poly_tti_factorized_unit_register_rung() {
    constexpr uint32_t addend_reg =
        J == 0u ? ckernel::p_sfpu::LREG12
                : 8u - J;  // J1..J4 -> L7..L4
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        addend_reg,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPNOP;
}

template <uint32_t J>
inline void poly_tti_factorized_unit_stream_rung() {
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_UPPER,
        (poly_tti_addend_bits(J) >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_LOWER,
        (poly_tti_addend_bits(J) & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        0);
}

template <uint32_t N>
inline void poly_tti_factorized_unit_padding() {
    if constexpr (N > 0u) {
        TTI_SFPNOP;
        poly_tti_factorized_unit_padding<N - 1u>();
    }
}

inline void poly_tti_replay_tile_factorized_unit() {
    poly_tti_pin_factorized_unit_coeffs();
    TT_REPLAY_RECORD(kPolyTtiFactorizedUnitBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // s=x*x
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);  // Q head=cD*s+cD-1
    if constexpr (POLY_DEGREE <= 7u) {
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            0);  // xs; independent Q-head hazard filler
    } else {
        TTI_SFPNOP;
    }
    if constexpr (POLY_DEGREE >= 3u) {
        poly_tti_factorized_unit_register_rung<0u>();
    }
    if constexpr (POLY_DEGREE >= 4u) {
        poly_tti_factorized_unit_register_rung<1u>();
    }
    if constexpr (POLY_DEGREE >= 5u) {
        poly_tti_factorized_unit_register_rung<2u>();
    }
    if constexpr (POLY_DEGREE >= 6u) {
        poly_tti_factorized_unit_register_rung<3u>();
    }
    if constexpr (POLY_DEGREE >= 7u) {
        poly_tti_factorized_unit_register_rung<4u>();
    }
    if constexpr (POLY_DEGREE >= 8u) {
        poly_tti_factorized_unit_stream_rung<5u>();
    }
    if constexpr (POLY_DEGREE >= 9u) {
        poly_tti_factorized_unit_stream_rung<6u>();
    }
    if constexpr (POLY_DEGREE >= 10u) {
        poly_tti_factorized_unit_stream_rung<7u>();
    }
    if constexpr (POLY_DEGREE >= 11u) {
        poly_tti_factorized_unit_stream_rung<8u>();
    }
    if constexpr (POLY_DEGREE >= 8u) {
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            0);  // xs after streamed L3 coefficient dies
        TTI_SFPNOP;
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        0);  // y=x+(x*s)*Q
    TTI_SFPNOP;
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_ADV, 0);
    // Pad exactly to the verifier's 2*D+10 replay lower bound. At D<=7 the
    // compact fixed-register schedule has six spare slots; streamed D8..D11
    // consume them monotonically (three down to zero).
    if constexpr (POLY_DEGREE <= 7u) {
        poly_tti_factorized_unit_padding<6u>();
    } else {
        poly_tti_factorized_unit_padding<11u - POLY_DEGREE>();
    }
    TT_REPLAY_REST(kPolyTtiFactorizedUnitBodySlots, 31);
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if POLY_TTI_SHAPE_PLAIN || POLY_TTI_SHAPE_LOG_SQUARE
#if defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION)
template <uint32_t J>
inline void poly_tti_square_affine_rung() {
    // Prgm0 is owned by the typed overlay's reciprocal, and that suffix may
    // use every allocatable LREG.  Make each recorded row independent of both
    // programmable-register and cross-row LREG state: c[D-2] was loaded into
    // L3 in the preamble, and every lower addend is streamed through that same
    // now-dead scratch immediately before its consuming MAD.
    constexpr uint32_t addend_reg = ckernel::p_sfpu::LREG3;
    if constexpr (J > 0u) {
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            SFPLOADI_MOD0_UPPER,
            (poly_tti_addend_bits(J) >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            SFPLOADI_MOD0_LOWER,
            (poly_tti_addend_bits(J) & 0xffffu));
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        addend_reg,
        ckernel::p_sfpu::LREG2,
        0);
}

inline void poly_tti_replay_tile_square_affine() {
    // Unlike the ordinary polynomial replay, this schedule cannot keep its
    // below-head addends in L7..L4 across rows: the same-row Mills/reciprocal
    // tail is allowed to consume every allocatable LREG.  Every addend is
    // consequently streamed into L3 inside the recorded body below.  Do not
    // run poly_tti_pin_coeffs() here -- those eight per-tile LOADI issues were
    // dead (the recorded body never reads the pins) and only delayed the
    // required typed suffix.  This is shape/resource driven and changes no
    // recorded arithmetic or raw-class action.
    TT_REPLAY_RECORD(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BODY_SLOTS);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_UPPER,
        (poly_tti_addend_bits(0) >> 16));
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_LOWER,
        (poly_tti_addend_bits(0) & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);
    poly_tti_square_affine_rung<0>();
    poly_tti_square_affine_rung<1>();
    poly_tti_square_affine_rung<2>();
    poly_tti_square_affine_rung<3>();
    poly_tti_square_affine_rung<4>();
    poly_tti_square_affine_rung<5>();
    poly_tti_square_affine_rung<6>();
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG4,
        SFPLOADI_MOD0_UPPER,
        (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BIAS_BITS >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG4,
        SFPLOADI_MOD0_LOWER,
        (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BIAS_BITS & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG2,
        0);
    auto same_row_suffix = []() {
        // The recorded core deliberately leaves its unrounded result in the
        // verifier-fixed L2 and does not advance or overwrite DST.  Reload the
        // original coordinate from this same row, bridge L2 into the existing
        // typed overlay, and perform the one final store only after every zone
        // and encoded-raw action has retired.
        {
            vFloat y = l_reg[LRegs::LReg2];
            vFloat x_raw = dst_reg[0];
            finalize_raw_domain_actions(x_raw, y);
            // End the numeric-tail live range at an architectural register.
            // The encoded classifier below reloads the still-untouched DST
            // row only after the Mills/exp/reciprocal graph has retired.
            l_reg[LRegs::LReg2] = y;
        }
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        {
            vUInt raw_u16 = dst_reg[0].mode<DataLayout::U16>();
            vFloat y = l_reg[LRegs::LReg2];
            finalize_encoded_raw_domain_actions(raw_u16, y);
            l_reg[LRegs::LReg2] = y;
        }
#endif
        vFloat y = l_reg[LRegs::LReg2];
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[0] = y;
        dst_reg++;
    };
    same_row_suffix();
    TT_REPLAY_REST_WITH_SAME_ROW_SUFFIX(
        TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BODY_SLOTS, 31, same_row_suffix());
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD)
template <uint32_t COEFF_INDEX>
inline void poly_tti_log_square_outer_rung() {
    constexpr uint32_t coefficient_bits =
        TT_SELECTED_LOG_SQUARE_OUTER_COEFF_BITS[COEFF_INDEX];
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_UPPER,
        (coefficient_bits >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_LOWER,
        (coefficient_bits & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        0);
}

inline void poly_tti_replay_tile_log_square_factorized_odd() {
#if defined(TT_SELECTED_LOG_SQUARE_WH_DECODED_TERMINALS) && !defined(ARCH_WORMHOLE)
#error "decoded signed-nonfinite log-square terminals require Wormhole"
#endif
    static_assert(
        TT_SELECTED_LOG_SQUARE_TERMINAL_BOUND_BITS == 0x3f800000u,
        "log-square TTI lowering requires the schedule-carried unit bound");
#if !defined(TT_SELECTED_LOG_SQUARE_POST_ROUND_RAW_TERMINALS) || \
    !defined(TT_SELECTED_LOG_SQUARE_ENDPOINTS_SIGNED_INF) || \
    !defined(TT_SELECTED_LOG_SQUARE_SIGNED_DAZ_ZERO)
#error "log-square TTI lowering requires its complete typed terminal plan"
#endif
#if defined(TT_SELECTED_LOG_SQUARE_EXTERIOR_NAN) == \
    defined(TT_SELECTED_LOG_SQUARE_EXTERIOR_SIGNED_INF)
#error "log-square TTI lowering requires exactly one exterior action"
#endif
    constexpr uint32_t kLogC0Bits =
        TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[0];
    TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, kLogC0Bits);
    TT_REPLAY_RECORD(TT_SELECTED_LOG_SQUARE_BODY_SLOTS);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // x*x [MAD]
    TTI_SFPNOP;
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // -x*x [MAD]
    TTI_SFPNOP;
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        0);  // u = 1-x*x [MAD]
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x3F40);  // 0.75; MAD gap
    TTI_SFPIADD(0, 1, 0, 6);
    TTI_SFPSETMAN(0, 0, 0, 1);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPIADD(0, 2, 1, 6);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0xC080);
    TTI_SFPMOV(0, 0, 2, 2);
    TTI_SFPIADD(0, 3, 2, 6);
    TTI_SFPMULI(0xBE80, ckernel::p_sfpu::LREG2, 0);  // -0.25*s [MAD]
    TTI_SFPNOP;
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG2,
        0);  // 2^-k-1 [MAD]
    TTI_SFPNOP;
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        0);  // reduced r [MAD]
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG3,
        0);  // c2*r+c1 [MAD]
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG3,
        0);  // +c0 [MAD]
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // r*r [MAD]
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG1,
        0);  // log1p reduced result [MAD]
    TTI_SFPABS(0, 0, 2, 0);
    TTI_SFPCAST(2, 2, 0);
    TTI_SFPSETSGN(0, 2, 0, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        0);  // log1p(-x*x) in L0 [MAD]
    TTI_SFPNOP;  // close the certified 31-slot body before the SFPI suffix

    auto same_row_suffix = []() {
        // Bind the same-row coordinate to a fixed live register before the
        // SFPI suffix allocates any temporaries.  Loading into L0 after Horner
        // would be invisible to the SFPI allocator and could overwrite its
        // live ratio; L6 is dead after the recorded log body and remains live
        // as x_raw through the terminal finalizer.
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG6,
            0,
            TT_ADDRMOD_DST_HOLD,
            0);
        TTI_SFPNOP;
        TTI_SFPSETSGN(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            1);  // float |log1p(-x*x)|; immediate sign clear
        constexpr uint32_t top_bits =
            TT_SELECTED_LOG_SQUARE_OUTER_COEFF_BITS[
                TT_SELECTED_LOG_SQUARE_OUTER_DEGREE];
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            SFPLOADI_MOD0_UPPER,
            (top_bits >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            SFPLOADI_MOD0_LOWER,
            (top_bits & 0xffffu));
        static_assert(
            TT_SELECTED_LOG_SQUARE_OUTER_DEGREE == 3u ||
                TT_SELECTED_LOG_SQUARE_OUTER_DEGREE == 4u,
            "certified log-square suffix supports P3/P4 outer leaves");
        if constexpr (TT_SELECTED_LOG_SQUARE_OUTER_DEGREE == 4u) {
            poly_tti_log_square_outer_rung<3>();
        }
        poly_tti_log_square_outer_rung<2>();
        poly_tti_log_square_outer_rung<1>();
        poly_tti_log_square_outer_rung<0>();
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG2,
            0);  // 1*x + x = 2*x [MAD]
        // The non-recorded WH suffix has no automatic MAD dependency stall.
        // Without this gap the multiply can consume L2's prior exponent value.
        TT_SFP_GAP_WH_HAZARD();
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG1,
            0);
        TTI_SFPNOP;
        // The declared terminal scale is exactly 2^-1.  Keep the normal
        // exponent adjustment and its sole BF16 rounding cell in fixed TTI
        // registers: high-level vector conditionals after a recorded replay
        // can alias the physical same-row register carrier.  L0 preserves the
        // unscaled value, L2 classifies biased exponent one, and L3 compares
        // its normalized magnitude with the post-round MIN_NORMAL boundary.
        TTI_SFPMOV(
            0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG0,
            0);
        TTI_SFPEXEXP(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG2,
            sfpi::SFPEXEXP_MOD1_NODEBIAS);
        TTI_SFPIADD(
            0xFFF,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG2,
            sfpi::SFPIADD_MOD1_ARG_IMM |
                sfpi::SFPIADD_MOD1_CC_GTE0);  // biased exponent - 1 >= 0
        TTI_SFPSETSGN(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG3,
            1);  // |unscaled|
        TTI_SFPDIVP2(
            0x07E,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG3,
            1);  // normalize exponent-one magnitude
        TTI_SFPADDI(
            0xBFFF,
            ckernel::p_sfpu::LREG3,
            0);  // normalized magnitude - 1.9921875
        TTI_SFPDIVP2(
            0xFFF,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG1,
            1);  // exact normal divide by two; exponent-one FTZs to zero
        TTI_SFPENCC(0, 0, 0, 0);  // exponent zero must not wrap to max-finite
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG2,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG3,
            0,
            sfpi::SFPSETCC_MOD1_LREG_GTE0);
        TTI_SFPSETMAN(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            1);  // signed MIN_NORMAL for the one post-round boundary cell
        TTI_SFPENCC(0, 0, 0, 0);
#if defined(TT_SELECTED_LOG_SQUARE_WH_DECODED_TERMINALS)
        // The WH decoded transport preserves the sign of the nonfinite
        // quotient but DAZ maps both zero signs to +0. Reuse the same core
        // and sole narrowing; install terminals only after Horner is dead.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        // SETEXP on x would retain its mantissa: build infinity from zero,
        // then copy only the decoded sign from the same-row carrier L6.
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPSETEXP(255, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);
        // Register-form SETSGN retains the destination sign and replaces its
        // magnitude from the source (not the other way around).
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2,
                     sfpi::SFPEXEXP_MOD1_NODEBIAS);
        TTI_SFPIADD(0xF81, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
                    sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG0, 0, 0);  // exponent < 127
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPENCC(0, 0, 0, 0);
#else
        // The anonymous certificate owns the open symmetric unit interval,
        // signed infinities at its two closed endpoints, and NaN everywhere
        // else.  Finalize that shape in fixed TTI registers after all Horner
        // temporaries are dead; this avoids exposing the physical same-row
        // carriers to the high-level allocator.
        TTI_SFPSETSGN(
            0,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG2,
            1);  // |x|
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LREG2,
            0);  // |x|-1 [MAD]
        // Perform the sole narrowing before installing any exact terminal.
        // Otherwise FP32 qNaN narrows to +Inf on Blackhole and a source-level
        // NaN assignment silently violates the Torch terminal contract.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG1,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            0,
            0x7FC0);  // post-narrow BF16 qNaN candidate
#if defined(TT_SELECTED_LOG_SQUARE_EXTERIOR_SIGNED_INF)
        TTI_SFPSETEXP(
            255,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG3,
            1);  // signed-infinity exterior default
#endif
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, 0);  // |x| < 1
        TTI_SFPMOV(
            0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG3,
            0);  // retain scaled core
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG2,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);  // |x| == 1
        TTI_SFPSETEXP(
            255,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG3,
            1);  // signed endpoint infinity
        TTI_SFPENCC(0, 0, 0, 0);
        // Signed DAZ zero is part of the typed terminal plan. L6 still owns
        // the original decoded coordinate, including the sign of every DAZ
        // zero. Form copysign(+0, original x) directly from that live value.
        // This conditional result = signed zero is after narrowing, so egress
        // cannot canonicalize -0; exhaustive class validation remains a
        // required silicon check for this target-specific ingress behavior.
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG6,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            sfpi::SFPLOADI_MOD0_FLOATB,
            0x0000);
        TTI_SFPSETSGN(
            0,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG1,
            0);
        TTI_SFPMOV(
            0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG3,
            0);
        TTI_SFPENCC(0, 0, 0, 0);
#endif
        TTI_SFPSTORE(
            ckernel::p_sfpu::LREG3,
            0,
            TT_ADDRMOD_DST_ADV,
            0);
    };
    same_row_suffix();
    TT_REPLAY_REST_WITH_SAME_ROW_SUFFIX(
        TT_SELECTED_LOG_SQUARE_BODY_SLOTS, 31, same_row_suffix());
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(TT_SELECTED_CORE_TTI_SCALED_SQUARE_RECONSTRUCTION)
template <uint32_t J>
inline void poly_tti_scaled_square_rung() {
    // This replay shares a tile with the generic typed-tail overlay.  The
    // overlay owns the architectural LREG file between replay rows, so no
    // Horner coefficient may survive there as a cross-row pin.  Prgm0 is
    // immutable and owns the first rung; every later addend is streamed in
    // this row.  Alternating the dead coordinate carrier L0 with L3 gives the
    // immediately preceding MAD source its required lifetime.
    constexpr uint32_t streamed_reg =
        (J & 1u) ? ckernel::p_sfpu::LREG3 : ckernel::p_sfpu::LREG0;
    if constexpr (J == 0u) {
        TTI_SFPNOP;
    } else {
        TTI_SFPLOADI(
            streamed_reg,
            SFPLOADI_MOD0_UPPER,
            (poly_tti_addend_bits(J) >> 16));
        TTI_SFPLOADI(
            streamed_reg,
            SFPLOADI_MOD0_LOWER,
            (poly_tti_addend_bits(J) & 0xffffu));
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        (J == 0u) ? ckernel::p_sfpu::LREG12 : streamed_reg,
        ckernel::p_sfpu::LREG2,
        0);
}

inline void poly_tti_replay_tile_scaled_square() {
    TT_REPLAY_RECORD(TT_SELECTED_CORE_TTI_SCALED_SQUARE_BODY_SLOTS);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_UPPER,
        (TT_SELECTED_CORE_TTI_SCALED_SQUARE_SCALE_BITS >> 16));
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_LOWER,
        (TT_SELECTED_CORE_TTI_SCALED_SQUARE_SCALE_BITS & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPNOP;
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        1);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);
    poly_tti_scaled_square_rung<0>();
    poly_tti_scaled_square_rung<1>();
    poly_tti_scaled_square_rung<2>();
    poly_tti_scaled_square_rung<3>();
    poly_tti_scaled_square_rung<4>();
    poly_tti_scaled_square_rung<5>();
    poly_tti_scaled_square_rung<6>();
    poly_tti_scaled_square_rung<7>();
    auto same_row_suffix = []() {
        // The recorded core holds D-RWC on this row. Reload through that same
        // physical counter. S55 consumes each body-natural signed-NaN action;
        // if one typed sign refinement remains, reload its physical U16 word
        // only after the numeric body and mathematical finalizer are dead.
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG0,
            0,
            TT_ADDRMOD_DST_HOLD,
            0);
        TTI_SFPNOP;
        vFloat y = l_reg[LRegs::LReg2];
        vFloat x_raw = l_reg[LRegs::LReg0];
        finalize_raw_domain_actions(x_raw, y);
        l_reg[LRegs::LReg2] = y;
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        {
            vUInt raw_u16 = dst_reg[0].mode<DataLayout::U16>();
            vFloat typed_y = l_reg[LRegs::LReg2];
            finalize_encoded_raw_domain_actions(raw_u16, typed_y);
            l_reg[LRegs::LReg2] = typed_y;
        }
#endif
        y = l_reg[LRegs::LReg2];
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        l_reg[LRegs::LReg2] = y;
        TTI_SFPSTORE(
            ckernel::p_sfpu::LREG2,
            0,
            TT_ADDRMOD_DST_ADV,
            0);
    };
    same_row_suffix();
    TT_REPLAY_REST_WITH_SAME_ROW_SUFFIX(
        TT_SELECTED_CORE_TTI_SCALED_SQUARE_BODY_SLOTS, 31, same_row_suffix());
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(TT_INLINE_PROGRAM_TTI_ENDPOINT_CONTINUATION_OVERLAY)
constexpr uint32_t poly_tti_endpoint_scale_reg() {
    if constexpr (TT_PROGRAM_TTI_ENDPOINT_SCALE_BITS == 0x00000000u) {
        return ckernel::p_sfpu::LCONST_0;
    } else if constexpr (TT_PROGRAM_TTI_ENDPOINT_SCALE_BITS == 0x3f800000u) {
        return ckernel::p_sfpu::LCONST_1;
    } else if constexpr (TT_PROGRAM_TTI_ENDPOINT_SCALE_BITS == 0xbf800000u) {
        return ckernel::p_sfpu::LCONST_neg1;
    } else {
        return ckernel::p_sfpu::LREG6;
    }
}

inline void poly_tti_pin_endpoint_scale() {
    if constexpr (!kPolyTtiEndpointConstant && !kPolyTtiEndpointHardwareScale) {
        TT_REPLAY_PIN_FP32(
            ckernel::p_sfpu::LREG6,
            TT_PROGRAM_TTI_ENDPOINT_SCALE_BITS);
    }
}
#endif

// One interleaved Horner rung pair: accA/accB += next coefficient.
template <uint32_t J>
inline void poly_tti_plain_rung() {
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        poly_tti_addend_reg(J),
        ckernel::p_sfpu::LREG2,
        0);  // accA = accA*xA + c[D-2-J]
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        poly_tti_addend_reg(J),
        ckernel::p_sfpu::LREG3,
        0);  // accB
}

inline void poly_tti_replay_tile_plain() {
    poly_tti_pin_coeffs();
#if defined(TT_INLINE_PROGRAM_TTI_ENDPOINT_CONTINUATION_OVERLAY)
    poly_tti_pin_endpoint_scale();
    TT_REPLAY_RECORD(kPolyTtiEndpointBodySlots);  // record + execute elements 0,1
#else
    TT_REPLAY_RECORD(kPolyTtiPlainBodySlots);  // record + execute elements 0,1
#endif
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB
#if defined(TT_INLINE_PROGRAM_TTI_ENDPOINT_CONTINUATION_OVERLAY)
    if constexpr (!kPolyTtiEndpointConstant) {
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG4,
            0);  // rA = fma(-1, B=1, xA)
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG5,
            0);  // rB
        TTI_SFPSWAP(
            0,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG4,
            9);  // rA = max(rA, +0)
        TTI_SFPSWAP(
            0,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG5,
            9);  // rB
    }
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        1);  // xcA = min(xA, B=1)
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        1);  // xcB
#endif
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);  // accA = c[D]*xA + c[D-1]
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG3,
        0);  // accB
    poly_tti_plain_rung<0>();
    if constexpr (POLY_DEGREE >= 3) {
        poly_tti_plain_rung<1>();
    }
    if constexpr (POLY_DEGREE >= 4) {
        poly_tti_plain_rung<2>();
    }
    if constexpr (POLY_DEGREE >= 5) {
        poly_tti_plain_rung<3>();
    }
    if constexpr (POLY_DEGREE >= 6) {
        poly_tti_plain_rung<4>();
    }
    if constexpr (POLY_DEGREE >= 7) {
        poly_tti_plain_rung<5>();
    }
#if defined(TT_INLINE_PROGRAM_TTI_ENDPOINT_CONTINUATION_OVERLAY)
    if constexpr (!kPolyTtiEndpointConstant) {
        TTI_SFPMAD(
            poly_tti_endpoint_scale_reg(),
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG2,
            0);  // yA = fma(a, rA, P(xcA))
        TTI_SFPMAD(
            poly_tti_endpoint_scale_reg(),
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG3,
            0);  // yB
    }
#endif
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // yA fp32 -> bf16 RNE (producer 2 back)
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // yB
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // yA (counter unchanged)
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_ADV, 2);  // yB, dest += 4
#if defined(TT_INLINE_PROGRAM_TTI_ENDPOINT_CONTINUATION_OVERLAY)
    TT_REPLAY_REST(kPolyTtiEndpointBodySlots, 15);  // pairs 1..15
#else
    TT_REPLAY_REST(kPolyTtiPlainBodySlots, 15);  // pairs 1..15
#endif
    TT_REPLAY_TILE_EPILOGUE();
}
#endif  // POLY_TTI_SHAPE_PLAIN

#if POLY_TTI_SHAPE_SIGNED_ABS
// One single-element Horner rung: [gap] + MAD. The gap is a NOP except for the
// in-body-coefficient rung (L3), whose SFPLOADI pair IS the gap (2 slots).
template <uint32_t J>
inline void poly_tti_abs_rung() {
    if constexpr (poly_tti_addend_reg(J) == 3) {
#if defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
        if constexpr (J == 4u) {
            // c2 was loaded a half-word at a time in the J1/J2 gaps.
            TTI_SFPNOP;
        } else {
#endif
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, SFPLOADI_MOD0_UPPER, (poly_tti_addend_bits(J) >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, SFPLOADI_MOD0_LOWER, (poly_tti_addend_bits(J) & 0xFFFFu));
#if defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
        }
    } else if constexpr (J == 1u) {
        // Stream exact c2 into L3 through two otherwise mandatory Horner gaps.
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            SFPLOADI_MOD0_UPPER,
            (poly_tti_addend_bits(4u) >> 16));
    } else if constexpr (J == 2u) {
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            SFPLOADI_MOD0_LOWER,
            (poly_tti_addend_bits(4u) & 0xFFFFu));
#endif
    } else {
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
        // The raw negative-NaN predicate's bitwise, condition-code-free
        // prefix occupies dependency gaps that the unchanged Horner already
        // requires.  L4 is available because the certified c2 addend is the
        // exact hardware zero; L3 stopped carrying the core bound after min.
        if constexpr (J == 0u) {
            TTI_SFPXOR(
                0,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG4,
                0);  // L4 = raw ^ 0x80ff
        } else if constexpr (J == 1u) {
            TTI_SFPAND(
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG3,
                1);  // L3 = delta & 0x80ff
        } else {
            TTI_SFPNOP;
        }
#else
        TTI_SFPNOP;
#endif
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        poly_tti_addend_reg(J),
        ckernel::p_sfpu::LREG2,
        0);  // acc = acc*|x| + c[D-2-J]
}

#if defined(ARCH_BLACKHOLE) && \
    defined(TT_INLINE_PROGRAM_TTI_COMMON_SFPU_BITWISE_INGRESS_CLOSURE)
constexpr uint32_t kPolyTtiAbsRecordedBodySlots =
    TT_PROGRAM_TTI_COMMON_SFPU_BODY_SLOTS;
#elif defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
constexpr uint32_t kPolyTtiAbsRecordedBodySlots =
    kPolyTtiSignedAbsAffineBodySlots;
#elif defined(TT_INLINE_PROGRAM_TTI_SYMMETRIC_SATURATION_OVERLAY)
constexpr uint32_t kPolyTtiAbsRecordedBodySlots =
    kPolyTtiSymmetricOverlayBodySlots;
#elif defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
constexpr uint32_t kPolyTtiAbsRecordedBodySlots =
    kPolyTtiSignedAbsRecipBodySlots;
#else
constexpr uint32_t kPolyTtiAbsRecordedBodySlots = kPolyTtiAbsBodySlots;
#endif

inline void poly_tti_replay_tile_abs() {
    poly_tti_pin_coeffs();
    TT_REPLAY_RECORD(kPolyTtiAbsRecordedBodySlots);  // record + execute element 0
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        TT_ADDRMOD_DST_HOLD,
        0);  // raw BH DST word
#endif
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
    // Start the bound immediate while the floating load matures.  Its lower
    // half is loaded after abs, before the bound is consumed by min.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_UPPER,
        (TT_TYPED_TTI_ABS_AFFINE_CORE_BOUND_BITS >> 16));
#else
    TTI_SFPNOP;  // load -> consumer gap
#endif
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 1);  // |x| (imm sign 0)
#if defined(ARCH_BLACKHOLE) && \
    defined(TT_INLINE_PROGRAM_TTI_COMMON_SFPU_BITWISE_INGRESS_CLOSURE)
    // BH DST raw BF16 is sign15|mantissa14:8|exponent7:0.  The certified
    // equality/nonzero masks select exactly the 127 negative NaN payloads.
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        TT_ADDRMOD_DST_HOLD,
        0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_USHORT,
        TT_PROGRAM_TTI_RAW_EQUAL_VALUE);
    TTI_SFPXOR(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        0);
    TTI_SFPAND(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        1);
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG2,
        0,
        sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG3,
        0,
        sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        1);
    TTI_SFPENCC(0, 0, 0, 0);
#elif defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_LOWER,
        (TT_TYPED_TTI_ABS_AFFINE_CORE_BOUND_BITS & 0xFFFFu));
#else
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_UPPER,
        (TT_TYPED_TTI_ABS_AFFINE_CORE_BOUND_BITS >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_LOWER,
        (TT_TYPED_TTI_ABS_AFFINE_CORE_BOUND_BITS & 0xFFFFu));
#endif
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        1);  // |x| = min(|x|, certified B)
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOADI_MOD0_USHORT,
        TT_TYPED_TTI_ABS_AFFINE_RAW_EQUAL_VALUE);  // min -> head gap
#else
    TTI_SFPNOP;  // min producer -> Horner head consumer gap
#endif
#elif defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
    // Existing BH sfpu_reciprocal_iter<1>, transcribed instruction-for-
    // instruction.  Positive |x| makes the NaN guard relevant only at zero:
    // there it keeps the +inf seed, and min(|x|,+inf) preserves the core zero.
    TTI_SFPARECIP(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        0);  // L0 = 1-|x|; raw x is reloaded by the final LOADMACRO
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG2,
        2);  // t = |x|*r0 - 2.0
    TTI_SFPNOP;
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, 0);  // t < 0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG3,
        3);  // r1 = r0*-t - 0 (predicated)
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        1);  // u = min(|x|, reciprocal(|x|))
#elif defined(TT_INLINE_PROGRAM_TTI_SYMMETRIC_SATURATION_OVERLAY)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        SFPLOADI_MOD0_UPPER,
        (TT_PROGRAM_TTI_ABS_CORE_BOUND_BITS >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        SFPLOADI_MOD0_LOWER,
        (TT_PROGRAM_TTI_ABS_CORE_BOUND_BITS & 0xFFFFu));
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        1);  // |x| = min(|x|, certified core bound)
    TTI_SFPNOP;  // min producer -> Horner MAD consumer gap
#endif
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);  // acc = c[D]*|x| + c[D-1] (SETSGN is bit-unit: gap-0 safe)
    poly_tti_abs_rung<0>();
    if constexpr (POLY_DEGREE >= 3) {
        poly_tti_abs_rung<1>();
    }
    if constexpr (POLY_DEGREE >= 4) {
        poly_tti_abs_rung<2>();
    }
    if constexpr (POLY_DEGREE >= 5) {
        poly_tti_abs_rung<3>();
    }
    if constexpr (POLY_DEGREE >= 6) {
        poly_tti_abs_rung<4>();
    }
    if constexpr (POLY_DEGREE >= 7) {
        poly_tti_abs_rung<5>();
    }
    if constexpr (POLY_DEGREE >= 8) {
        poly_tti_abs_rung<6>();
    }
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
    // Preserve raw L0 through the sign restore.  L1's clamped coordinate is
    // dead after J6 and can be replaced by the original absolute magnitude
    // for the positive affine tail a*|x|+b.  Exact-zero c2/c1/c0 addends use
    // LCONST0 in the raw-ingress closure, leaving L4 for the raw predicate.
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 1);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG1,
        0);  // a*|x|+b, capacity-gated to |x|-1
#endif
    if constexpr (kPolyTtiClamp) {
        TTI_SFPNOP;
        // vec_min_max vs the hw 1.0: result = min(result, 1.0); the max lands
        // in the read-only constant lane (the exact word sfpi emits for the
        // BASIS_CLAMP_MAX == 1.0 clamp).
        TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);
    }
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
    TTI_SFPNOP;  // final Horner MAD -> continuation max consumer gap
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        9);  // magnitude=max(P(min(|x|,B)), |x|-1)
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
    // Restore the ordinary input sign first.  The complementary-mask raw
    // predicate then clears only the 127 negative-NaN result signs; raw -Inf
    // has delta==0 and remains negative.  CC is enabled for exactly one
    // simple-unit instruction and is disabled before LOADMACRO.
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        0);
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG3,
        0,
        sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(
        0,
        ckernel::p_sfpu::LREG4,
        0,
        sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        1);
    TTI_SFPENCC(0, 0, 0, 0);
    // Macro 0 rounds literal L0 into VD=L2 and stores it at issue+3.  The
    // macro's load value is intentionally dead; it exists only because
    // SFPLOADMACRO is the capacity-preserving RNE/store carrier.
    TTI_SFPLOADMACRO(2, 0, TT_ADDRMOD_DST_ADV, 0);
#else
    // The proven macro reloads raw x into dead L3, restores its sign onto L2,
    // performs BF16 RNE, and stores through ADDR_MOD_6.
    TTI_SFPLOADMACRO(3, 0, TT_ADDRMOD_DST_ADV, 0);
#endif
#elif defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL)
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG0, 0, 0);  // (1-|x|) < 0 => reciprocal tail
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG2,
        0);  // tail: y = complement - P(u)
    TTI_SFPENCC(3, 0, 0, 10);
    // Macro 0: LOAD current raw x -> VD=L3, SETSGN@+1 from L2/raw-sign,
    // RNE-to-staging@+3, STORE@+5.  L3's streamed c1 is dead after the final
    // Horner rung, and the next body does not rewrite L3 until reciprocal@+4.
    // This is essential: VD=L0 races SETSGN@+1 against the next body's
    // SFPLOAD L0, and silicon returns that next raw row (observed as
    // 0.5 -> 0.50390625). ROUND_SRCB_OVERRIDE is zero so the Round slot
    // overrides its architectural VC input with VD, exactly as the ISA's
    // FP16_B LOADMACRO example and the proven log tail do.  The next
    // iteration's reciprocal SETCC is the eighth subsequent SFPU issue, so
    // global CC is disabled throughout every in-flight macro instruction.
    // BH encodes (sequence_id << 2) | (VD & 3) in lreg_ind, so sequence 0
    // with VD=L3 is the first operand value 3; the second operand remains the
    // load InstrMod.  Keeping this explicit prevents the rejected 0,3 form,
    // whose objdump was `sfploadmacro 0,L0,0,3,6` (L0 plus InstrMod 3).
    TTI_SFPLOADMACRO(3, 0, TT_ADDRMOD_DST_ADV, 0);
#else
    TTI_SFPNOP;
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);  // sign of x restored
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // fp32 -> bf16 RNE (SETSGN gap-0 safe)
    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);  // dest += 2
#endif
    TT_REPLAY_REST(kPolyTtiAbsRecordedBodySlots, 31);  // elements 1..31
#if defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL) || \
    defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
    // STORE@LM+5 is five issued instructions past the final body slot (unlike
    // log LM, which has one ordinary body instruction after LOADMACRO).  Drain
    // all five before SETRWC/commit can observe the destination bank.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
    TT_REPLAY_TILE_EPILOGUE();
}
#endif  // POLY_TTI_SHAPE_SIGNED_ABS

#if POLY_TTI_SHAPE_PARITY_EVEN
// Parity-even rung addend for 0-based rung index J: coefficient c[TOP-2(J+2)]
// (stride-2 walk over the even coefficients; the odd-index LUT residue is
// skipped exactly like eval_polynomial_parity). prgm0 first, then the
// per-tile pinned L7..L5; rungs past those load their addend IN-BODY into L4
// (self-contained — see the kPolyTtiParityBodySlots comment).
constexpr uint32_t poly_tti_parity_addend_reg(uint32_t j) {
    switch (j) {
        case 0: return 12;  // prgm0
        case 1: return 7;
        case 2: return 6;
        case 3: return 5;
        default: return 4;  // in-body SFPLOADI scratch
    }
}
constexpr uint32_t poly_tti_parity_addend_bits(uint32_t j) {
    return __builtin_bit_cast(uint32_t, poly_tti_c(kPolyTtiParityTop - 2u * (j + 2u)));
}

// One interleaved x^2-Horner rung pair: accA/accB = acc*s + c[TOP-2(J+2)].
// In-body-coefficient rungs (L4) prepend their SFPLOADI pair (bit-unit,
// gap-0 readable; also extra spacing for the MAD chain).
template <uint32_t J>
inline void poly_tti_parity_rung() {
    if constexpr (poly_tti_parity_addend_reg(J) == 4) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_UPPER, (poly_tti_parity_addend_bits(J) >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_LOWER, (poly_tti_parity_addend_bits(J) & 0xFFFFu));
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        poly_tti_parity_addend_reg(J),
        ckernel::p_sfpu::LREG2,
        0);  // accA = accA*sA + c
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        poly_tti_parity_addend_reg(J),
        ckernel::p_sfpu::LREG3,
        0);  // accB
}

inline void poly_tti_replay_tile_parity_even() {
    // Per-tile pins: rung 1-3 addends (rung 0 rides prgm0; deeper rungs are
    // in-body). Register map: L0/L1 = xA/xB then sA/sB (squared in place —
    // even parity never needs x again), L2/L3 = accA/accB, L4 = in-body
    // coefficient scratch, L12/13/14 = prgm0/1/2.
    if constexpr (kPolyTtiParitySteps >= 3 && poly_tti_parity_addend_reg(1) == 7) {
        TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, poly_tti_parity_addend_bits(1));
    }
    if constexpr (kPolyTtiParitySteps >= 4 && poly_tti_parity_addend_reg(2) == 6) {
        TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG6, poly_tti_parity_addend_bits(2));
    }
    if constexpr (kPolyTtiParitySteps >= 5 && poly_tti_parity_addend_reg(3) == 5) {
        TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG5, poly_tti_parity_addend_bits(3));
    }
    TT_REPLAY_RECORD(kPolyTtiParityBodySlots);  // record + execute elements 0,1
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // sA = xA*xA (same single-rounded MUL as the sfpi x2)
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // sB
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);  // accA = c[TOP]*sA + c[TOP-2] (sA gap: sB MUL between)
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG3,
        0);  // accB
    if constexpr (kPolyTtiParitySteps >= 2) {
        poly_tti_parity_rung<0>();
    }
    if constexpr (kPolyTtiParitySteps >= 3) {
        poly_tti_parity_rung<1>();
    }
    if constexpr (kPolyTtiParitySteps >= 4) {
        poly_tti_parity_rung<2>();
    }
    if constexpr (kPolyTtiParitySteps >= 5) {
        poly_tti_parity_rung<3>();
    }
    if constexpr (kPolyTtiParitySteps >= 6) {
        poly_tti_parity_rung<4>();
    }
    if constexpr (kPolyTtiParitySteps >= 7) {
        poly_tti_parity_rung<5>();
    }
    if constexpr (kPolyTtiParitySteps >= 8) {
        poly_tti_parity_rung<6>();
    }
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // yA fp32 -> bf16 RNE (producer 2 back)
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // yB
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // yA (counter unchanged)
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_ADV, 2);  // yB, dest += 4
    TT_REPLAY_REST(kPolyTtiParityBodySlots, 15);  // pairs 1..15
    TT_REPLAY_TILE_EPILOGUE();
}
#endif  // POLY_TTI_SHAPE_PARITY_EVEN

inline void poly_tti_replay_tile() {
#if POLY_TTI_SHAPE_FACTORIZED_UNIT
    poly_tti_replay_tile_factorized_unit();
#elif POLY_TTI_SHAPE_LOG_SQUARE
    poly_tti_replay_tile_log_square_factorized_odd();
#elif defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION)
    poly_tti_replay_tile_square_affine();
#elif defined(TT_SELECTED_CORE_TTI_SCALED_SQUARE_RECONSTRUCTION)
    poly_tti_replay_tile_scaled_square();
#elif POLY_TTI_SHAPE_PLAIN
    poly_tti_replay_tile_plain();
#elif POLY_TTI_SHAPE_SIGNED_ABS
    poly_tti_replay_tile_abs();
#elif POLY_TTI_SHAPE_PARITY_EVEN
    poly_tti_replay_tile_parity_even();
#endif
}

#if defined(TT_SELECTED_TTI_RAW_SHADOW_LATE_SCHEDULE)
inline void poly_tti_preserve_raw_shadow_tile() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        dst_reg[TT_SELECTED_TTI_RAW_SHADOW_ROW_BASE + d]
            .mode<DataLayout::U16>() = raw_u16;
    }
}

inline void poly_tti_finalize_raw_shadow_tile() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat result = dst_reg[d];
        vUInt raw_u16 =
            dst_reg[TT_SELECTED_TTI_RAW_SHADOW_ROW_BASE + d]
                .mode<DataLayout::U16>();
        finalize_encoded_raw_domain_actions(raw_u16, result);
#ifdef USE_BF16
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
        dst_reg[d] = result;
    }
}
#endif
#else
#define POLY_TTI_CANDIDATE 0
#endif

// Include specialized implementations for common segment counts
// These use manual unrolling to work around Wormhole SFPU compiler bug
#if !EVAL_METHOD_IS_STANDALONE
#include "piecewise_generic_specialized.cpp"
#endif

// DST-resident coefficient store (opt-in pilot; see the header). Included
// inside namespace sfpi; compiles to nothing unless DST_COEFF_PROBE or
// DST_COEFF_STORE is defined by the codegen.
#include "dst_coeff_store.h"

#if defined(TT_DOMAIN_ACTION_PROGRAM) || defined(TT_INLINE_PROGRAM_OVERLAY)
inline void identity_domain_actions_eval() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = x;
        finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(AFFINE_COLLAPSE) && !defined(AFFINE_IDENTITY)
// Affine collapse: the whole fit is y = c0 + c1*x. One SFPMAD per element,
// bypassing the entire segment cascade / LUT machinery. c0/c1 are SFPMAD
// immediates (single hoistable load each), so the per-element body is one MAD.
inline void affine_collapse_eval() {
    constexpr float C0 = AFFINE_C0;
    constexpr float C1 = AFFINE_C1;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = C1 * x + C0;
        y = apply_output_postcompose(y);
        finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#ifdef FUSE_GRAD_MUL
// Fused unary-backward multiply: y = grad * f'(x), replacing a separate
// framework ttnn.multiply launch.
//
// On entry the evaluator has left f'(x) in DST tile 0 (rows 0..31) and the
// reader has streamed `grad` into DST tile 1 (rows 32..63) via
// copy_tile(cb_grad, 0, 1).
//
// WHY HERE, in the tile-loop epilogue, and not inside the segment cascade:
// by this point every cascade temporary is dead (the evaluator has returned),
// so the live set is exactly two vFloats against the 8-entry LREG file. An
// in-cascade multiply would instead keep `grad` live across the whole segment
// chain -- precisely the register-pressure class that already ICEs the SFPI
// allocator on the _dual path ("maximum number of generated reload insns per
// insn achieved (90)", piecewise_generic_specialized.cpp). One insertion here
// also covers EVERY eval kind, so no evaluator body needs to change.
//
// Accuracy: with grad == 1 the multiply is exact (IEEE-754 v*1.0), so a fused
// dump must be BYTE-IDENTICAL to the unfused f' dump -- that is the regression
// oracle for this path. For general grad the bf16 DST row means the result is
// bf16(bf16(f'(x)) * g), a double rounding that matches what TTNN already does
// (its f' intermediate is a bf16 DRAM tensor before the multiply).
inline void fuse_grad_mul() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT)
        // Verifier-owned backward composition: a zero derivative suppresses
        // every incoming gradient class to canonical +0; a nonzero derivative
        // copies the gradient and optionally applies the declared fixed scale.
        // This runs after the selected evaluator and overlay have retired, so
        // it preserves the common two-row fused transport and LREG budget.
        vFloat factor = dst_reg[d];
        vFloat y = 0.0f;
        v_if(factor != 0.0f) {
            y = dst_reg[32 + d];
#if defined(TT_INLINE_PROGRAM_GRADIENT_SCALE)
            y = y * TT_INLINE_PROGRAM_GRADIENT_SCALE;
#endif
        }
        v_endif;
#else
        vFloat y = dst_reg[d] * dst_reg[32 + d];
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

// Single-element gated-affine origin-half body with a typed joint terminal.
// S55 owns the U16 partition and closed issue budget.  The escape issues this
// exact body directly for every row; the selected path records it once and
// replays it, so DIRECT_ALGEBRAIC_TTI_DISABLE changes transport only.
#if defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    defined(TT_GATED_SINGLE_ROW_JOINT_TERMINAL) && \
    !defined(POSTCOMPOSE_AFFINE_Y) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    !defined(TT_INLINE_PROGRAM_OVERLAY)
#define GATED_SINGLE_ROW_JOINT_TTI_CANDIDATE 1
constexpr bool kGatedSingleRowJointTtiReplay =
    tti_replay_fits(TT_GATED_JOINT_BODY_SLOTS);

__attribute__((always_inline)) inline void gated_single_row_joint_tti_body() {
    static_assert(
        (float)(GATED_QUADRATIC_Q0) == 0.5f &&
            (float)(GATED_QUADRATIC_Q1) > 0.0f &&
            (float)(GATED_QUADRATIC_Q1) < 0.5f &&
            TT_GATED_JOINT_NEG_NAN_RESULT == 2u &&
            TT_GATED_JOINT_NEG_INF_RESULT == 3u &&
            TT_GATED_JOINT_EQUAL_MASK == 0x80ffu &&
            TT_GATED_JOINT_EQUAL_VALUE == 0x80ffu &&
            TT_GATED_JOINT_MANTISSA_MASK == 0x7f00u &&
            TT_GATED_JOINT_NEG_NAN_PATTERNS == 127u &&
            TT_GATED_JOINT_NEG_INF_PATTERNS == 1u &&
            TT_GATED_JOINT_NUMERIC_SLOTS == 11u &&
            TT_GATED_JOINT_TERMINAL_SLOTS == 14u &&
            TT_GATED_JOINT_BODY_SLOTS == 25u &&
            TT_GATED_JOINT_PREDECESSOR_SLOTS == 27u &&
            TT_GATED_JOINT_GAP_FILL_SLOTS == 2u &&
            TT_GATED_JOINT_REMAINING_NOPS == 1u &&
            TT_GATED_JOINT_BODY_SLOTS + TT_GATED_JOINT_GAP_FILL_SLOTS ==
                TT_GATED_JOINT_PREDECESSOR_SLOTS &&
            TT_GATED_JOINT_PEAK_LREGS <= 8u,
        "gated joint terminal replay is not resource/class closed");
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // 0: x
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG1,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        TT_ADDRMOD_DST_HOLD,
        0);  // 1: raw U16 / x-load gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        0);  // 2: gate
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOADI_MOD0_USHORT,
        TT_GATED_JOINT_EQUAL_VALUE);  // 3: gate gap / raw anchor
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);  // 4
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);  // 5
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);  // 6
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);  // 7
    TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);  // 8
    // Raw partitioning is independent of L4 and fills the required
    // DIVP2->ADDI gap.
    TTI_SFPAND(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        1);  // 9: negative exponent-FF partition delta
    TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0);  // 10
    // The independent x*gate product fills the ADDI->SETCC gap.  It is also
    // separated from its gate-clamp producer at slot 7 by three issues.  No
    // CC window is active across either filler.
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // 11: x*gate
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 12
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);  // 13
    TTI_SFPENCC(0, 0, 0, 0);  // 14
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 15
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);  // 16
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0xFF80);  // 17
    TTI_SFPENCC(0, 0, 0, 0);  // 18
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 19
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x0000);  // 20
    TTI_SFPENCC(0, 0, 0, 0);  // 21
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // 22
    TTI_SFPNOP;  // 23: RNE/store dependency
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_ADV, 0);  // 24
}

inline void gated_single_row_joint_tti_tile() {
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG6,
        __builtin_bit_cast(uint32_t, (float)(GATED_QUADRATIC_Q1)));
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG7,
        __builtin_bit_cast(uint32_t, (float)(GATED_QUADRATIC_Q0)));
#if defined(DIRECT_ALGEBRAIC_TTI_DISABLE)
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d) {
        gated_single_row_joint_tti_body();
    }
#else
    TT_REPLAY_RECORD(TT_GATED_JOINT_BODY_SLOTS);
    gated_single_row_joint_tti_body();
    TT_REPLAY_REST(TT_GATED_JOINT_BODY_SLOTS, 31);
#endif
    TT_REPLAY_TILE_EPILOGUE();
}
#else
#define GATED_SINGLE_ROW_JOINT_TTI_CANDIDATE 0
constexpr bool kGatedSingleRowJointTtiReplay = false;
#endif

// ---------------------------------------------------------------------------
// Direct-algebraic TTI replay lowering.
//
// The replayed shapes below are already complete evaluators selected by the
// ordinary declarative/codegen flow; this changes only how their fixed
// algebraic instruction stream is transported to Tensix.  Two elements are
// interleaved so every MAD-family producer has an independent instruction in
// its read-after-write gap.  The final B store advances D-RWC by four rows and
// the shared replay epilogue restores it before the next tile's datacopy.
//
// Eligibility is structural and fail-closed: BF16 Blackhole, no postcompose or
// typed domain overlay/action, and an exact supported evaluation shape.  The
// existing sfpi loops remain the A arm and DIRECT_ALGEBRAIC_TTI_DISABLE forces
// them for byte-identity and timing comparisons.  No activation identity is
// visible here and no selected coefficient is changed.
#if defined(USE_BF16) && defined(ARCH_BLACKHOLE) &&                  \
    !defined(DIRECT_ALGEBRAIC_TTI_DISABLE) &&                        \
    !defined(POSTCOMPOSE_AFFINE_Y) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    !defined(TT_INLINE_PROGRAM_OVERLAY) &&                           \
    /* The unit-high slope body preserves the BH common-copy +Inf ingress. */ \
    /* A declared negative-NaN return of +Inf is therefore already owned */  \
    /* by that body and needs neither a discriminator nor a second path. */  \
    (!defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) ||       \
     defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION) ||            \
     defined(TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_OWNS_NEG_NAN_POS_INF)) && \
    !defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) &&            \
    (!defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) ||       \
     defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)) &&           \
    !defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) &&         \
    (!defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) ||     \
     defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)) &&          \
    !defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR)
#define DIRECT_ALGEBRAIC_TTI_CANDIDATE 1

#if defined(CLAMPED_AFFINE_COLLAPSE) && defined(CLAMPED_AFFINE_HAS_MIN) && \
    defined(CLAMPED_AFFINE_HAS_MAX)
constexpr bool kDirectTtiClampedAffineUnitBounds2x =
    ((float)(CLAMPED_AFFINE_MIN) == 0.0f) &&
    ((float)(CLAMPED_AFFINE_MAX) == 1.0f) &&
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    tti_replay_fits(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS);
#else
    tti_replay_fits(12u);
#endif
constexpr bool kDirectTtiClampedAffineSymmetricIdentity2x =
    ((float)(CLAMPED_AFFINE_C1) == 1.0f) &&
    ((float)(CLAMPED_AFFINE_C0) == 0.0f) &&
    ((float)(CLAMPED_AFFINE_MIN) == -1.0f) &&
    ((float)(CLAMPED_AFFINE_MAX) == 1.0f) && tti_replay_fits(10u);
constexpr bool kDirectTtiClampedAffine2x =
    kDirectTtiClampedAffineUnitBounds2x ||
    kDirectTtiClampedAffineSymmetricIdentity2x;
#else
constexpr bool kDirectTtiClampedAffineUnitBounds2x = false;
constexpr bool kDirectTtiClampedAffineSymmetricIdentity2x = false;
constexpr bool kDirectTtiClampedAffine2x = false;
#endif

#if defined(THRESHOLD_IDENTITY_SELECT) && \
    defined(TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI)
constexpr bool kDirectTtiThresholdIdentity2x =
    TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_ROWS == 2u &&
    TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_BODY_SLOTS == 16u &&
    TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_PEAK_LREGS == 7u &&
    tti_replay_fits(TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_BODY_SLOTS);
#else
constexpr bool kDirectTtiThresholdIdentity2x = false;
#endif

#if defined(ABS_VALUE)
constexpr uint32_t kDirectTtiAbsBodySlots =
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS;
#else
    8u;
#endif
constexpr bool kDirectTtiAbs2x = tti_replay_fits(kDirectTtiAbsBodySlots);
#else
constexpr bool kDirectTtiAbs2x = false;
#endif

#if defined(GATED_AFFINE_PRODUCT) || defined(GATED_QUADRATIC_COLLAPSE)
constexpr bool kDirectTtiGatedAffineProduct2x = tti_replay_fits(
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS
#else
    14u
#endif
);
#else
constexpr bool kDirectTtiGatedAffineProduct2x = false;
#endif

#if defined(SLOPE_MAX) && defined(SLOPE_MAX_IS_MAX) && \
    defined(TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI)
constexpr bool kDirectTtiSlopeMax2x =
    ((float)(SLOPE_MAX_SLOPE_HI) == 1.0f) &&
    ((float)(SLOPE_MAX_SLOPE_LO) > 0.0f) &&
    ((float)(SLOPE_MAX_SLOPE_LO) < 1.0f) &&
    (TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_COUNT <= 1u) &&
    (TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BODY_SLOTS ==
     (TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_COUNT == 1u ? 22u : 12u)) &&
    (TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_ESCAPE_BODY_SLOTS ==
     (TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_COUNT == 1u ? 23u : 12u)) &&
    tti_replay_fits(TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BODY_SLOTS) &&
    tti_replay_fits(TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_ESCAPE_BODY_SLOTS);
#else
constexpr bool kDirectTtiSlopeMax2x = false;
#endif

constexpr bool kDirectAlgebraicTtiReplay =
    kDirectTtiClampedAffine2x || kDirectTtiThresholdIdentity2x ||
    kDirectTtiGatedAffineProduct2x || kDirectTtiSlopeMax2x ||
    kDirectTtiAbs2x;
#else
#define DIRECT_ALGEBRAIC_TTI_CANDIDATE 0
constexpr bool kDirectTtiClampedAffine2x = false;
constexpr bool kDirectTtiClampedAffineUnitBounds2x = false;
constexpr bool kDirectTtiClampedAffineSymmetricIdentity2x = false;
constexpr bool kDirectTtiThresholdIdentity2x = false;
constexpr bool kDirectTtiAbs2x = false;
constexpr bool kDirectTtiGatedAffineProduct2x = false;
constexpr bool kDirectTtiSlopeMax2x = false;
constexpr bool kDirectAlgebraicTtiReplay = false;
#endif

#if DIRECT_ALGEBRAIC_TTI_CANDIDATE
template <uint32_t A, uint32_t B>
inline void direct_tti_finish_pair() {
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        A,
        A,
        A,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        B,
        B,
        B,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(A, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPSTORE(B, 0, TT_ADDRMOD_DST_ADV, 2);  // pair complete, dest += 4
}

// Split form for a capacity-closed 32-slot replay plus one same-row store
// suffix.  The recorded body rounds both values and stores A without advancing;
// the suffix stores B and owns the pair's destination advance.
template <uint32_t A, uint32_t B>
inline void direct_tti_round_pair_store_first() {
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN, 0, A, A, A,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN, 0, B, B, B,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(A, 0, TT_ADDRMOD_DST_HOLD, 0);
}

template <uint32_t B>
inline void direct_tti_store_second() {
    TTI_SFPSTORE(B, 0, TT_ADDRMOD_DST_ADV, 2);
}

#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
template <
    uint32_t A,
    uint32_t B,
    uint32_t RAWA,
    uint32_t RAWB,
    uint32_t MASK,
    uint32_t SCRATCH>
inline void direct_tti_fused_raw_terminal_pair() {
    TTI_SFPLOAD(
        RAWA,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        TT_ADDRMOD_DST_HOLD,
        0);
    TTI_SFPLOAD(
        RAWB,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        TT_ADDRMOD_DST_HOLD,
        2);
    TTI_SFPLOADI(
        MASK,
        sfpi::SFPLOADI_MOD0_USHORT,
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE);
    TTI_SFPXOR(0, MASK, RAWA, 0);
    TTI_SFPXOR(0, MASK, RAWB, 0);
    if constexpr (TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_NONZERO_MASK == 0u) {
        TTI_SFPSETCC(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(
            A,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(
            B,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16);
        TTI_SFPENCC(0, 0, 0, 0);
    } else {
        TTI_SFPAND(MASK, RAWA, SCRATCH, 1);
        TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(
            A,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPAND(MASK, RAWB, SCRATCH, 1);
        TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(
            B,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

// Exact masked-equality plus masked-nonzero terminal.  NONZERO_MASK is pinned
// outside the recording.  Unlike the compact complementary-mask helper above,
// this spelling does not test the whole XOR delta, so a sign bit outside both
// declared masks cannot leak into the selected class.
template <
    uint32_t A,
    uint32_t B,
    uint32_t RAWA,
    uint32_t RAWB,
    uint32_t MASK,
    uint32_t SCRATCH,
    uint32_t NONZERO_MASK>
inline void direct_tti_fused_raw_terminal_pair_exact_nonzero() {
    TTI_SFPLOAD(
        RAWA, sfpi::SFPLOAD_MOD0_FMT_UINT16, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPLOAD(
        RAWB, sfpi::SFPLOAD_MOD0_FMT_UINT16, TT_ADDRMOD_DST_HOLD, 2);
    TTI_SFPLOADI(
        MASK, sfpi::SFPLOADI_MOD0_USHORT,
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE);
    TTI_SFPXOR(0, MASK, RAWA, 0);
    TTI_SFPXOR(0, MASK, RAWB, 0);
    TTI_SFPAND(MASK, RAWA, SCRATCH, 1);
    TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPAND(NONZERO_MASK, RAWA, RAWA, 1);
    TTI_SFPSETCC(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(
        A, sfpi::SFPLOADI_MOD0_FLOATB,
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPAND(MASK, RAWB, SCRATCH, 1);
    TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPAND(NONZERO_MASK, RAWB, RAWB, 1);
    TTI_SFPSETCC(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(
        B, sfpi::SFPLOADI_MOD0_FLOATB,
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16);
    TTI_SFPENCC(0, 0, 0, 0);
}
#endif

#if defined(ABS_VALUE)
inline void direct_tti_abs_tile() {
    constexpr uint32_t kBodySlots = kDirectTtiAbsBodySlots;
    static_assert(tti_replay_fits(kBodySlots));
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    static_assert(
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_MASK == 0x80ffu &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE == 0x80ffu &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_NONZERO_MASK == 0x7f00u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16 == 0xff80u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_TRUE_PATTERNS == 127u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS == 23u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_PEAK_LREGS <= 8u,
        "typed direct-algebraic terminal fusion is not resource/class closed");
#endif
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB; A load gap
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        1);  // |xA|; immediate sign zero
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        1);  // |xB|; A op is B load gap
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    direct_tti_fused_raw_terminal_pair<
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG7>();
#endif
    direct_tti_finish_pair<ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3>();
    TT_REPLAY_REST(kBodySlots, 15);
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(CLAMPED_AFFINE_COLLAPSE) && defined(CLAMPED_AFFINE_HAS_MIN) && \
    defined(CLAMPED_AFFINE_HAS_MAX)
inline void direct_tti_clamped_affine_tile() {
    constexpr uint32_t kBodySlots =
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS;
#else
        12u;
#endif
    static_assert(tti_replay_fits(kBodySlots));
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    static_assert(
        ((TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_MASK == 0x80ffu &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE == 0x80ffu &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_NONZERO_MASK == 0x7f00u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS == 27u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_PEAK_LREGS <= 8u) ||
         (TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_MASK == 0xffffu &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE == 0x8000u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_NONZERO_MASK == 0u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS == 21u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_PEAK_LREGS <= 7u)),
        "clamped typed terminal fusion is not resource/predicate closed");
#endif
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG6,
        __builtin_bit_cast(uint32_t, (float)(CLAMPED_AFFINE_C1)));
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG7,
        __builtin_bit_cast(uint32_t, (float)(CLAMPED_AFFINE_C0)));
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG3,
        0);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, 1);
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    direct_tti_fused_raw_terminal_pair<
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1>();
#endif
    direct_tti_finish_pair<ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3>();
    TT_REPLAY_REST(kBodySlots, 15);
    TT_REPLAY_TILE_EPILOGUE();
}

// Exact identity-affine clamp to [-1, +1].  SFPSWAP writes both operands, so
// both bounds must be read-only architectural constants; ordinary pinned LREGs
// are not persistent clamp operands.  Loading directly into the result LREGs
// also removes the identity MAD pair without changing the declarative shape.
inline void direct_tti_symmetric_identity_clamp_tile() {
    constexpr uint32_t kBodySlots =
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS;
#else
        10u;
#endif
    static_assert(tti_replay_fits(kBodySlots));
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    static_assert(
        ((TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_MASK == 0xffffu &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE == 0x8000u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_NONZERO_MASK == 0u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16 == 0u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_TRUE_PATTERNS == 1u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS == 21u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_PEAK_LREGS <= 7u) ||
         (TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_MASK == 0x80ffu &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE == 0x80ffu &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_NONZERO_MASK == 0x7f00u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS == 27u &&
          TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_PEAK_LREGS <= 8u)),
        "symmetric-clamp exact terminal is not resource/predicate closed");
#endif
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG2, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG3, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, 1);
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    direct_tti_fused_raw_terminal_pair<
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1>();
#endif
    direct_tti_finish_pair<ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3>();
    TT_REPLAY_REST(kBodySlots, 15);
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(GATED_AFFINE_PRODUCT) || defined(GATED_QUADRATIC_COLLAPSE)
inline void direct_tti_gated_affine_product_tile() {
    // For x*clamp(a*x+0.5, 0, 1), BH flushes the two signed FP32-subnormal
    // products for BF16 input 0x00ff before destination RNE can map them to
    // signed MIN_NORMAL.  Detect that exact typed input in twelve paired slots:
    // exact addexp(+126) moves the comparison to 1.9921875, where subtraction
    // between adjacent BF16 values remains normal rather than being erased by
    // FTZ.  Predicated SETMAN(0) retains x's sign and exponent while building
    // signed FP32 MIN_NORMAL.  No raw reload, operation identity, coefficient
    // rewrite, or alternate evaluator enters this schedule.
    constexpr bool kOriginHalfTie =
        ((float)(GATED_QUADRATIC_Q0) == 0.5f) &&
        ((float)(GATED_QUADRATIC_Q1) != 0.0f) &&
        ((float)(GATED_QUADRATIC_Q1) < 0.5f) &&
        ((float)(GATED_QUADRATIC_Q1) > -0.5f);
    constexpr uint32_t kBodySlots =
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
        TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS;
#else
        kOriginHalfTie ? 26u : 14u;
#endif
    static_assert(tti_replay_fits(kBodySlots));
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    static_assert(
        (float)(GATED_QUADRATIC_Q0) == 1.0f &&
            (float)(GATED_QUADRATIC_Q1) == 0.5f &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_MASK == 0xffffu &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_EQUAL_VALUE == 0x8000u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_NONZERO_MASK == 0u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_OUTPUT_BF16 == 0u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_TRUE_PATTERNS == 1u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_BODY_SLOTS == 25u &&
            TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_PEAK_LREGS <= 8u,
        "gated exact terminal is not resource/predicate closed");
#endif
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG6,
        __builtin_bit_cast(uint32_t, (float)(GATED_QUADRATIC_Q1)));
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG7,
        __builtin_bit_cast(uint32_t, (float)(GATED_QUADRATIC_Q0)));
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG3,
        0);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, 1);
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG3,
        0);
    if constexpr (kOriginHalfTie) {
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);  // |xA|
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5, 1);  // |xB|
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);  // *2^126
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 1);  // *2^126
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0);  // scaled |xA| - 1.9921875
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG5, 0);  // scaled |xB| - 1.9921875
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG4,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG5,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
        TTI_SFPENCC(0, 0, 0, 0);
    }
#if defined(TT_DIRECT_ALGEBRAIC_RAW_TERMINAL_FUSION)
    direct_tti_fused_raw_terminal_pair<
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1>();
#endif
    direct_tti_finish_pair<ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3>();
    TT_REPLAY_REST(kBodySlots, 15);
    TT_REPLAY_TILE_EPILOGUE();
}

#endif

#if defined(SLOPE_MAX) && defined(SLOPE_MAX_IS_MAX) && \
    defined(TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI)
inline void direct_tti_unit_high_slope_tile() {
#if defined(SLOPE_PRGM_PRELOAD_DISABLE)
    constexpr uint32_t kBodySlots =
        TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_ESCAPE_BODY_SLOTS;
#else
    constexpr uint32_t kBodySlots =
        TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BODY_SLOTS;
#endif
    static_assert(kDirectTtiSlopeMax2x);
    static_assert(tti_replay_fits(kBodySlots));
#if defined(SLOPE_PRGM_PRELOAD_DISABLE)
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG6,
        __builtin_bit_cast(uint32_t, (float)(SLOPE_MAX_SLOPE_LO)));
#endif
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG0, 0, 0);  // xA < 0
    TTI_SFPMUL(
#if defined(SLOPE_PRGM_PRELOAD_DISABLE)
        ckernel::p_sfpu::LREG6,
#else
        ckernel::p_sfpu::LREG12,
#endif
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // selected low-slope multiply, predicated
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);  // xB < 0
    TTI_SFPMUL(
#if defined(SLOPE_PRGM_PRELOAD_DISABLE)
        ckernel::p_sfpu::LREG6,
#else
        ckernel::p_sfpu::LREG12,
#endif
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // selected low-slope multiply, predicated
    TTI_SFPENCC(3, 0, 0, 10);
    if constexpr (TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_COUNT == 1u) {
        // Preserve the coefficient-derived BF16 RNE result which Blackhole's
        // FP32 multiply otherwise flushes.  The physical U16 predicate is
        // exact and anonymous; the two independent row deltas also provide
        // each raw load/XOR consumer gap inside the replay body.
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOAD_MOD0_FMT_UINT16,
            TT_ADDRMOD_DST_HOLD,
            0);
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG3,
            sfpi::SFPLOAD_MOD0_FMT_UINT16,
            TT_ADDRMOD_DST_HOLD,
            2);
#if defined(SLOPE_PRGM_PRELOAD_DISABLE)
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG4,
            sfpi::SFPLOADI_MOD0_USHORT,
            TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_RAW_U16);
#endif
        TTI_SFPXOR(
            0,
#if defined(SLOPE_PRGM_PRELOAD_DISABLE)
            ckernel::p_sfpu::LREG4,
#else
            ckernel::p_sfpu::LREG13,
#endif
            ckernel::p_sfpu::LREG2,
            0);
        TTI_SFPXOR(
            0,
#if defined(SLOPE_PRGM_PRELOAD_DISABLE)
            ckernel::p_sfpu::LREG4,
#else
            ckernel::p_sfpu::LREG13,
#endif
            ckernel::p_sfpu::LREG3,
            0);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG2,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG0,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_OUTPUT_BF16);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(
            0,
            ckernel::p_sfpu::LREG3,
            0,
            sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            sfpi::SFPLOADI_MOD0_FLOATB,
            TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_OUTPUT_BF16);
        TTI_SFPENCC(0, 0, 0, 0);
    }
    direct_tti_finish_pair<ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1>();
    TT_REPLAY_REST(kBodySlots, 15);
    TT_REPLAY_TILE_EPILOGUE();
}
#endif

#if defined(THRESHOLD_IDENTITY_SELECT) && \
    defined(TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI)
inline void direct_tti_threshold_identity_tile() {
    constexpr uint32_t kBodySlots =
        TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_BODY_SLOTS;
    static_assert(
        TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_ROWS == 2u &&
            kBodySlots == 16u &&
            TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_PEAK_LREGS == 7u &&
            TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_COMPARATOR_BF16 ==
                (((__builtin_bit_cast(
                       uint32_t, (float)(THRESHOLD_IDENTITY_LAMBDA)) >> 16) == 0u)
                     ? 0x0080u
                     : ((__builtin_bit_cast(
                             uint32_t, (float)(THRESHOLD_IDENTITY_LAMBDA)) >> 16) + 1u)) &&
            tti_replay_fits(kBodySlots),
        "threshold-identity replay is not resource/value closed");
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG7,
        __builtin_bit_cast(
            uint32_t,
            -(float)(
                __builtin_bit_cast(
                    float,
                    TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI_COMPARATOR_BF16
                        << 16))));
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 2);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, 0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    direct_tti_finish_pair<
        ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1>();
    TT_REPLAY_REST(kBodySlots, 15);
    TT_REPLAY_TILE_EPILOGUE();
}
#endif
#endif  // DIRECT_ALGEBRAIC_TTI_CANDIDATE

#if defined(CLAMPED_AFFINE_COLLAPSE)
// Clamped affine collapse: y = min(max(c0 + c1*x, low), high). This is the
// exact algebraic form of piecewise-linear CSVs with flat outer regions, but is
// driven only by coefficient pattern detection in run_csv.sh.
inline void clamped_affine_collapse_eval() {
    constexpr float C0 = CLAMPED_AFFINE_C0;
    constexpr float C1 = CLAMPED_AFFINE_C1;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = C1 * x + C0;
#ifdef CLAMPED_AFFINE_HAS_MIN
        vFloat lo = CLAMPED_AFFINE_MIN;
        ordered_min_max(lo, y);  // y = max(lo, y)
#endif
#ifdef CLAMPED_AFFINE_HAS_MAX
        vFloat hi = CLAMPED_AFFINE_MAX;
        ordered_min_max(y, hi);  // y = min(y, hi)
#endif
        y = apply_output_postcompose(y);
        finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(ABS_VALUE)
// y=abs(x). Exact whole-function lowering for two affine pieces (-x, +x).
inline void abs_value_eval() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = setsgn(x, 0);
        y = apply_output_postcompose(y);
        finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR)
        finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(THRESHOLD_IDENTITY_SELECT)
// y=x for |x|>lambda, else zero. Equality belongs to zero.
inline void threshold_identity_select_eval() {
    constexpr float LAMBDA = THRESHOLD_IDENTITY_LAMBDA;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = x;
        vFloat ax = setsgn(x, 0);
        v_if(ax <= LAMBDA) { y = 0.0f; }
        v_endif;
        y = apply_output_postcompose(y);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
#if defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_post_round_raw_domain_actions(x_raw, raw_u16, y);
#else
        finalize_post_round_raw_domain_actions(x_raw, y);
#endif
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(THRESHOLD_SOFTSHIFT_SELECT)
// y=sign(x)*(abs(x)-lambda) for |x|>lambda, else zero. Equality belongs to zero.
inline void threshold_softshift_select_eval() {
    constexpr float LAMBDA = THRESHOLD_SOFTSHIFT_LAMBDA;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat ax = setsgn(x, 0);
        vFloat y = 0.0f;
        v_if(ax > LAMBDA) {
            y = ax - LAMBDA;
            v_if(x < 0.0f) { y = -y; }
            v_endif;
        }
        v_endif;
        y = apply_output_postcompose(y);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
#if defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_post_round_raw_domain_actions(x_raw, raw_u16, y);
#else
        finalize_post_round_raw_domain_actions(x_raw, y);
#endif
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(GATED_AFFINE_PRODUCT) || defined(GATED_QUADRATIC_COLLAPSE)
// y = x * clamp(q0 + q1*x, 0, 1).
inline void gated_affine_product_eval() {
    constexpr float Q0 = GATED_QUADRATIC_Q0;
    constexpr float Q1 = GATED_QUADRATIC_Q1;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat gate = Q1 * x + Q0;
        vFloat zero = 0.0f;
        vFloat one = 1.0f;
        ordered_min_max(zero, gate);  // gate = max(0, gate)
        ordered_min_max(gate, one);   // gate = min(gate, 1)
        vFloat y = x * gate;
        y = apply_output_postcompose(y);
        finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(SLOPE_MAX)
// y = combine(s_hi*x, s_lo*x): exact two-slope piecewise-linear kink at x=0
// (leaky_relu, prelu). Compute both slope products, then one vec_min_max picks
// the branch -- for x>=0 the larger slope wins under max; for x<0 the sign flip
// makes the smaller slope win. One SFPMUL (s_lo*x; s_hi*x folds to x when s_hi=1)
// plus one vec_min_max, versus a segmented Horner cascade.
inline void slope_max_eval() {
    constexpr float SLO = SLOPE_MAX_SLOPE_LO;
    constexpr float SHI = SLOPE_MAX_SLOPE_HI;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat hi = SHI * x;
        vFloat lo = SLO * x;
        ordered_min_max(lo, hi);  // lo = min(lo, hi), hi = max(lo, hi)
#ifdef SLOPE_MAX_IS_MAX
        vFloat y = hi;
#else
        vFloat y = lo;
#endif
#if defined(ARCH_BLACKHOLE) && defined(USE_BF16)
        // SFPMUL flushes an FP32 subnormal before the BF16 destination RNE.
        // For an exact power-of-two linear arm, the fitter derives the sole
        // BF16 input (if any) whose exact result is in the top half-subnormal
        // bin and must therefore round to MIN_NORMAL before output FTZ.  The
        // typed pair is coefficient/target-derived; no activation identity or
        // extra evaluator participates.  Compare the exact physical BF16 DST
        // word: SFPI floating immediate equality coarsens this binade and can
        // alias many adjacent inputs.
#ifdef SLOPE_MAX_BF16_FTZ_RNE_NEGATIVE
        vUInt raw_delta_negative = dst_reg[d].mode<DataLayout::U16>();
        raw_delta_negative = raw_delta_negative ^
            vUInt(SLOPE_MAX_BF16_FTZ_RNE_RAW_U16_NEGATIVE);
        v_if(raw_delta_negative == 0u) {
            y = SLOPE_MAX_BF16_FTZ_RNE_OUTPUT_NEGATIVE;
        }
        v_endif;
#endif
#ifdef SLOPE_MAX_BF16_FTZ_RNE_POSITIVE
        vUInt raw_delta_positive = dst_reg[d].mode<DataLayout::U16>();
        raw_delta_positive = raw_delta_positive ^
            vUInt(SLOPE_MAX_BF16_FTZ_RNE_RAW_U16_POSITIVE);
        v_if(raw_delta_positive == 0u) {
            y = SLOPE_MAX_BF16_FTZ_RNE_OUTPUT_POSITIVE;
        }
        v_endif;
#endif
#endif
        y = apply_output_postcompose(y);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

}  // namespace sfpi
#endif
