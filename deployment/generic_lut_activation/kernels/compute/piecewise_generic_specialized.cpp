// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Template-recursive unrolling for piecewise polynomial evaluation.
 *
 * Compile-time unrolls the v_if / eval_polynomial / v_endif chain for ANY
 * segment count using C++ template recursion. The SFPU compiler (GCC 15.1.0)
 * handles if constexpr recursion, and __attribute__((always_inline)) ensures
 * all instantiations merge into the caller before LTO — preventing the
 * constprop.isra register spilling that plagued the old hand-written
 * specializations at high degrees.
 *
 * ADAPTIVE DEGREE: When HAS_SEGMENT_DEGREES is defined (by run_csv.sh or
 * generate_embedded_kernels.py), SEGMENT_DEGREES[SEG] provides the effective
 * degree per segment as a constexpr array. Since both the array and SEG are
 * compile-time constants, SEGMENT_DEGREES[SEG] is a valid template argument
 * for eval_polynomial<>. This skips wasted FMA ops on zero high-order coeffs.
 *
 * PARITY x²-HORNER: When POLY_PARITY_ODD or POLY_PARITY_EVEN is defined,
 * evaluation uses x² basis with stride-2 coefficient access, halving FMA count.
 * x² is precomputed once per DST register and threaded through the unroller.
 *
 * CPS (coeffs per segment) is always POLY_DEGREE+1 because LUT storage uses
 * max degree for indexing regardless of per-segment effective degree.
 */

#pragma once

#include "sfpi_min_max.h"

// The CSV codegen declares whether the asymptotic correction is active on the
// left tail, right tail, or whole domain.  Keep that decision compile-time so a
// right-tail factor does not materialize the old 1e38 upper-bound comparison
// (and, more importantly, does not run on the non-tail segments).
#if (                                                                                    \
    defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X) ||           \
    defined(ASYMPTOTIC_FACTOR_QUADRATIC)) &&                                             \
    ((defined(ASYMPTOTIC_REGION_LEFT) + defined(ASYMPTOTIC_REGION_RIGHT) + defined(ASYMPTOTIC_REGION_ALL)) != 1)
#error "asymptotic factor requires exactly one LEFT/RIGHT/ALL region"
#endif
#if defined(ASYMPTOTIC_REGION_LEFT)
#define TT_ASYMPTOTIC_IF(x) v_if((x) < ASYMPTOTIC_BOUND)
#define TT_ASYMPTOTIC_END_IF v_endif
#elif defined(ASYMPTOTIC_REGION_RIGHT)
#define TT_ASYMPTOTIC_IF(x) v_if((x) >= ASYMPTOTIC_BOUND)
#define TT_ASYMPTOTIC_END_IF v_endif
#elif defined(ASYMPTOTIC_REGION_ALL)
#define TT_ASYMPTOTIC_IF(x) if constexpr (true)
#define TT_ASYMPTOTIC_END_IF
#endif

#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X) || defined(ASYMPTOTIC_FACTOR_QUADRATIC)
inline vFloat apply_asymptotic_scale(vFloat value) {
    value = value * ASYMPTOTIC_SCALE;
#ifdef ASYMPTOTIC_NEGATE_OUTPUT
    // Do not spell this as multiplication by -1 or unary minus: SFPI GCC
    // 15.1's rvtt_combine pass ICEs in try_combine_negated_result for the
    // right-tail logsigmoid chain.  XOR is exact IEEE sign negation and is an
    // established SFPI idiom (also used by the trig/cbrt reconstruction paths).
    return as<vFloat>(as<vInt>(value) ^ (vInt)0x80000000);
#else
    return value;
#endif
}
#endif

// ============================================================================
// Hardware-exponent-ALU standalone evaluators (exp2 / log2 / pow)
//
// These bypass the piecewise segment cascade entirely: the decompose + Horner +
// recombine in exp_hw_eval / log_hw_eval / pow_hw_eval (defined in
// piecewise_generic.cpp) IS the full approximation. We loop over all 32 dst
// registers, preloading nothing per-iteration that GCC can already hoist (the
// folded constants are constexpr and become SFPMAD immediates).
// ============================================================================

#if EVAL_METHOD_IS_STANDALONE
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_hw_reduce(const std::array<float, LUT_SIZE>& /*lut*/) {
#if defined(EVAL_METHOD_TRIG_RESIDUAL)
#pragma GCC unroll 16
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = trig_residual_odd_eval<TRIG_RESIDUAL_DEGREE>(x);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
    return;
#elif defined(EVAL_METHOD_TAN_STANDALONE)
#pragma GCC unroll 16
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = tan_standalone_eval<TAN_STANDALONE_DEGREE>(x);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
    return;
#elif defined(EVAL_METHOD_NEWTON_ROOT)
    // Newton-Raphson magic-seed root: constants preloaded in kernel_main, no
    // per-loop hoist needed. sqrt/rsqrt mirror native (~15-18 SFPU instrs); cbrt
    // is division-free (inverse-cube-root multiply-only Newton). All fit the
    // register budget unrolled, but cbrt's larger body unrolls more modestly.
#if (NEWTON_ROOT_N == 3)
#pragma GCC unroll 2
#else
#pragma GCC unroll 8
#endif
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = newton_root_eval<POLY_DEGREE>(x);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
    return;
#else  // !EVAL_METHOD_NEWTON_ROOT
#if defined(HW_PRELOAD)
    // GENERIC constant-pool preload path (exp2 / log2 / pow, any degree). The 3
    // hottest constants live in vConstFloatPrgm0/1/2 (programmed ONCE in
    // kernel_main). A few more loop-invariant scalars (clamp 255.0, the round
    // magic, the sigmoid/sigmoid-product MULT) are hoisted into pre-loop LREGs here; the remaining
    // coefficients are read from their constexpr global inside *_hw_eval_preloaded
    // (compiler hoists what fits, literals the rest — the codegen-logged spill).
#if defined(EXPONENT_ALU_EXP2)
    vFloat thr_hoist = 255.0f;
    // Hoist the 127.0 bias into a loop-invariant LREG so x*MULT+127 fuses into a
    // single SFPMAD per element (no separate SFPMUL+SFPADDI for the literal).
    vFloat c127_hoist = 127.0f;
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    vFloat mult_hoist = EXP_HW_MULT;  // prgm0 reserved for sfpu_reciprocal
#endif
    // FIX A: hoist the below-prgm coeffs (c[DEG-2..0]) into pre-loop LREGs ONCE so
    // the per-element Horner reloads none of them (drives in-body SFPLOADI -> 0).
    // [EXP_HW_DEGREE>=2 ? EXP_HW_DEGREE-1 : 1] keeps the array non-zero-sized.
    float exp_cvspill[(EXP_HW_DEGREE >= 2) ? (EXP_HW_DEGREE - 1) : 1];
    if constexpr (EXP_HW_DEGREE >= 2) {
#pragma GCC unroll 16
        for (int k = 0; k < (int)EXP_HW_DEGREE - 1; k++) {
            const int power = (int)EXP_HW_DEGREE - 2 - k;  // coeff polynomial power
#if defined(EXP_HW_FUSED)
            // FUSED: pre-scale by 2^-23*power so Horner runs over the
            // unnormalized exman fraction (no per-element `* 0x1p-23f`).
            exp_cvspill[k] = EXP_HW_COEFFS[power] * exp_hw_fused_scale(power);
#else
            exp_cvspill[k] = EXP_HW_COEFFS[power];  // c[DEG-2]..c[0]
#endif
        }
    }
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = exp_hw_eval_preloaded<EXP_HW_DEGREE>(
            x,
            thr_hoist,
            c127_hoist,
            exp_cvspill
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
            ,
            mult_hoist
#endif
        );
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
    return;
#elif defined(EXPONENT_ALU_LOG2)
    // Below-prgm coeffs c[DEG-2..DEG-5] pinned in LREGs (hw_preload_pin — plain
    // hoisted locals get constant-folded back into per-element SFPLOADI pairs);
    // deeper tail spills to cvspill (see the generic caller in
    // piecewise_generic.cpp for the measured mechanism).
    constexpr float log_pin_a = (LOG_HW_DEGREE >= 2) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 2] : 0.0f;
    constexpr float log_pin_b = (LOG_HW_DEGREE >= 3) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 3] : 0.0f;
    constexpr float log_pin_c = (LOG_HW_DEGREE >= 4) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 4] : 0.0f;
    constexpr float log_pin_d = (LOG_HW_DEGREE >= 5) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 5] : 0.0f;
    vFloat log_cva = 0.0f;
    vFloat log_cvb = 0.0f;
    vFloat log_cvc = 0.0f;
    vFloat log_cvd = 0.0f;
    if constexpr (LOG_HW_DEGREE >= 2) {
        log_cva = hw_preload_pin<LRegs::LReg7>(log_pin_a);
    }
    if constexpr (LOG_HW_DEGREE >= 3) {
        log_cvb = hw_preload_pin<LRegs::LReg6>(log_pin_b);
    }
    if constexpr (LOG_HW_DEGREE >= 4) {
        log_cvc = hw_preload_pin<LRegs::LReg5>(log_pin_c);
    }
    if constexpr (LOG_HW_DEGREE >= 5) {
        log_cvd = hw_preload_pin<LRegs::LReg4>(log_pin_d);
    }
    float log_cvspill[(LOG_HW_DEGREE >= 6) ? (LOG_HW_DEGREE - 5) : 1];
    if constexpr (LOG_HW_DEGREE >= 6) {
#pragma GCC unroll 16
        for (int k = 0; k < (int)LOG_HW_DEGREE - 5; k++) {
            log_cvspill[k] = LOG_HW_COEFFS[(int)LOG_HW_DEGREE - 6 - k];
        }
    }
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = log_hw_eval_preloaded<LOG_HW_DEGREE>(x, log_cva, log_cvb, log_cvc, log_cvd, log_cvspill);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
    return;
#elif defined(EXPONENT_ALU_POW)
    vFloat magic_hoist = ckernel::sfpu::Converter::as_float(0x4B400000U);
    // FIX A: hoist below-prgm coeffs c[DEG-2..0] into pre-loop LREGs (see exp).
    float pow_cvspill[(POW_HW_DEGREE >= 2) ? (POW_HW_DEGREE - 1) : 1];
    if constexpr (POW_HW_DEGREE >= 2) {
#pragma GCC unroll 16
        for (int k = 0; k < (int)POW_HW_DEGREE - 1; k++) {
            pow_cvspill[k] = POW_HW_COEFFS[(int)POW_HW_DEGREE - 2 - k];
        }
    }
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat y = pow_hw_eval_preloaded<POW_HW_DEGREE>(x, magic_hoist, pow_cvspill);
        finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
    return;
#endif
#endif  // HW_PRELOAD
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
#if defined(EXPONENT_ALU_EXP2)
        vFloat y = exp_hw_eval<EXP_HW_DEGREE>(x);
#elif defined(EXPONENT_ALU_LOG2)
        vFloat y = log_hw_eval<LOG_HW_DEGREE>(x);
#else
        vFloat y = pow_hw_eval<POW_HW_DEGREE>(x);
#endif
        finalize_raw_domain_actions(x_raw, y);
        // bf16 dst: round-to-nearest before SFPSTORE truncates (matches TTNN).
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
#endif  // !EVAL_METHOD_NEWTON_ROOT
}
#endif

// ============================================================================
// Single-eval recursive unroller
// ============================================================================

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void unroll_segment(
    const std::array<float, LUT_SIZE>& lut,
    vFloat x_clamped,
    vFloat x,
    vFloat& result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    ,
    vFloat x2
#elif defined(BASIS_INPUT_ABS_X)
    ,
    vFloat x_eval
#endif
) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
#ifdef HAS_SEGMENT_DEGREES
        constexpr uint32_t DEG = SEGMENT_DEGREES[SEG];
#else
        constexpr uint32_t DEG = POLY_DEGREE;
#endif
        v_if(x_clamped >= lut[SEG]) {
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            result = eval_polynomial_parity<DEG>(&lut[CO + SEG * CPS], x, x2);
#elif defined(BASIS_INPUT_ABS_X)
            result = eval_polynomial<DEG>(&lut[CO + SEG * CPS], x_eval);
#else
            result = eval_polynomial<DEG>(&lut[CO + SEG * CPS], x);
#endif
        }
        v_endif;
        unroll_segment<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x_clamped,
            x,
            result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            ,
            x2
#elif defined(BASIS_INPUT_ABS_X)
            ,
            x_eval
#endif
        );
    }
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_N(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    for (int d = 0; d < 32; d++) {
#if defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) || defined(TT_SELECTED_CORE_TOTAL_FORM)
        // The source row remains untouched until the final store.  Keep only
        // the prepared coordinate live across the mature evaluator and reload
        // raw x for the certified terminal overlay.  This is the single-lane
        // counterpart of the dual evaluator's existing same-row schedule.
        vFloat x_orig = prepare_raw_domain_input(dst_reg[d]);
#if defined(TT_INLINE_PROGRAM_SELECTED_FACTOR_INTRINSIC_CLAMP)
        // The paired reconstruction certificate proves over all 65,536 BF16
        // encodings that both exterior coordinates may enter the unchanged
        // selected core at its owned endpoints and remain intrinsic-total.
        // Use unpredicated SFPSWAP lowering here: a high-level predicate would
        // inherit the ambient SFPU CC state before the mature body runs.
        vFloat selected_core_lo = TT_PROGRAM_OVERLAY_CORE_LOWER;
        vFloat selected_core_hi = TT_PROGRAM_OVERLAY_CORE_UPPER;
        ordered_min_max(selected_core_lo, x_orig);
        ordered_min_max(x_orig, selected_core_hi);
#endif
#else
        vFloat x_raw = dst_reg[d];
        vFloat x_orig = prepare_raw_domain_input(x_raw);
#endif

#if defined(RANGE_REDUCTION_EXP)
        // Exp range reduction + compose arms, ported verbatim from the generic
        // single-tile piecewise_generic_lut (piecewise_generic.cpp). The
        // specialized evaluator previously ran BARE plain-exp here (exp_reduce +
        // plain 2^k expand), silently dropping the cosh/sinh/sigmoid/silu/exp2
        // reconstructions -- exactly the same disease the REDUCE_LOG arm had.
        // Overflow is decided per-arm; exp_expand's reconstructed-exponent guard
        // (new_exp >= 255 -> +inf) is the precise saturator. Each EXP_OVERFLOW*
        // constant is declared only in the arm(s) that read it (-Wunused).
#if defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD)
        // cosh/sinh stay finite past exp(x)'s overflow: th=0.5*exp(|x|) does not
        // hit FLT_MAX until |x|=ln(2*FLT_MAX)=89.41598... (mpmath-verified).
        constexpr float EXP_OVERFLOW_HYPERBOLIC = 89.416f;
#elif defined(REDUCE_EXP_BASE2)
        // exp2's shoulders are owned by the pre-evaluation return_class terminals
        // (x<=-126.5 -> +0, x>=128 -> +inf) and exp_expand's new_exp guard.
#elif defined(REDUCE_EXP_COMPOSE_ELU) || defined(REDUCE_EXP_COMPOSE_SELU)
        // elu/celu/selu own both shoulders (x>=0 identity/affine; -1 / -lambda*
        // alpha saturation tail), so no natural-exp EXP_OVERFLOW constant.
#else
        constexpr float EXP_OVERFLOW = 256.0f;
#endif
        vFloat x;
        vInt k_int;
#if defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD)
        // cosh/sinh reduce on the ABS coordinate (one reduction serves both
        // signed exponentials); x_abs is reused in the expand block for the
        // overflow test, sinh's small-|x| crossover, and the final copysign.
        vFloat x_abs = setsgn(x_orig, 0);
        exp_reduce(x_abs, x, k_int);
#elif defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT) || defined(REDUCE_EXP_COMPOSE_SIGMOID)
        // silu/swish and plain sigmoid reduce on -x directly (sign carried by
        // which side of zero -x lands on); the two arms diverge only at expand.
        vFloat x_neg = -x_orig;
        exp_reduce(x_neg, x, k_int);
#elif defined(REDUCE_EXP_BASE2)
        // exp2(x) = 2^k * 2^r: exact integer reduction (k=round(x), r=x-k).
        exp2_reduce(x_orig, x, k_int);
#else
        exp_reduce(x_orig, x, k_int);
#endif
#elif defined(RANGE_REDUCTION_TRIG)
        vFloat x;
        vInt q_int;
        trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        vFloat x;
        vInt j_int;
        tan_reduce(x_orig, x, j_int);
#elif defined(RANGE_REDUCTION_LOG)
        vFloat x;
        vInt e_int;
        log_reduce(x_orig, x, e_int);
#if defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED) || defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT)
        // sqrt(2)-centered mantissa split (see the identical arm in
        // piecewise_generic.cpp::piecewise_generic_lut). log_reduce gives m in
        // [1,2); fold the [sqrt2,2) half down by one more exponent-FIELD step
        // (setexp, not an fp subtract) so the residual becomes symmetric about
        // 1 in [sqrt2/2, sqrt2). Then x = m - 1.0f is the residual coordinate
        // f the single-segment minimax was fit in (log(1+f), c0 pinned to 0).
        // Sterbenz guarantees m-1.0f is exact. log_expand below is unchanged --
        // it only ever sees the final poly VALUE (== log(m)), not the
        // coordinate. WITHOUT this arm the specialized evaluator ran the
        // residual-fit polynomial on the raw mantissa m in [1,2): silicon MAE
        // ~5.6, ~4.7M negative outputs, pure_ulp_max=inf.
        constexpr float LOG_SQRT2 = 1.4142135623730951f;
        v_if(x >= LOG_SQRT2) {
            e_int = e_int + 1;
            x = setexp(x, 126);
        }
        v_endif;
        x = x - 1.0f;
#endif
#elif defined(RANGE_REDUCTION_CBRT)
        // Extract BIASED exponent (always non-negative) and sign BEFORE poly eval.
        // Use exexp_nodebias to avoid SFPU's sign-magnitude format for negative debiased values.
        vInt cbrt_biased_e = exexp(setsgn(x_orig, 0), ExponentMode::Biased);
        vInt cbrt_sign = as<vInt>(x_orig) & 0x80000000;
        vFloat x = cbrt_reduce_m(x_orig);
#elif defined(RANGE_REDUCTION_RECIP_COMPLEMENT)
        // Signed-abs reciprocal/complement reduction (atan): reduced =
        // min(|x|, 1/|x|) in [0,1], poly coordinate s = reduced^2. Mirrors the
        // canonical single-lane cascade arm in piecewise_generic.cpp.
        vFloat recip_reduced;
        vFloat x;
        {
            vFloat abs_x = setsgn(x_orig, 0);
            vFloat inv_abs = ckernel::sfpu::sfpu_reciprocal_iter<2>(abs_x);
            recip_reduced = abs_x;
            v_if(abs_x > 1.0f) { recip_reduced = inv_abs; }
            v_endif;
            x = recip_reduced * recip_reduced;
        }
#else
        vFloat x = x_orig;
#endif

#if defined(BASIS_INPUT_ABS_X)
        vFloat x_eval = setsgn(x_orig, 0);
        vFloat& x_clamped = x_eval;
#else
        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x_clamped = x;
#endif

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
        vFloat x2 = x * x;
#endif

        // Segment 0 (no v_if needed)
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
#ifdef HAS_SEGMENT_DEGREES
        vFloat result = eval_polynomial_parity<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x, x2);
#else
        vFloat result = eval_polynomial_parity<POLY_DEGREE>(&lut[COEFF_OFFSET], x, x2);
#endif
#elif defined(BASIS_INPUT_ABS_X)
#ifdef HAS_SEGMENT_DEGREES
        vFloat result = eval_polynomial<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x_eval);
#else
        vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x_eval);
#endif
#elif defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT)
        // Linear-term-SPLIT reconstruction (see eval_polynomial_log_split in
        // piecewise_generic.cpp): the dominant linear term c1*f is injected
        // EXACTLY outside the Horner chain on the sqrt2-centered residual f
        // (x above). Single-segment route only -- the sqrt2-centered log family
        // is fit as one interval.
        static_assert(NUM_SEGMENTS == 1, "REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT requires a single segment");
        vFloat result = eval_polynomial_log_split<POLY_DEGREE>(&lut[COEFF_OFFSET], x);
#else
#ifdef HAS_SEGMENT_DEGREES
        vFloat result = eval_polynomial<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x);
#else
        vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);
#endif
#endif

        // Segments 1..NUM_SEGMENTS-1 via recursive unrolling
        unroll_segment<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x_clamped,
            x,
            result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            ,
            x2
#elif defined(BASIS_INPUT_ABS_X)
            ,
            x_eval
#endif
        );

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        result = result * x_eval;
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
        result = result * basis_sqrt(vFloat(1.0f) - x_eval);
#endif

#if defined(BASIS_AFFINE_EVEN)
        result = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig + (BASIS_AFFINE_EVEN_SCALE * x_eval) * result;
#endif

#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
        v_if(x_eval > vFloat(BASIS_RIGHT_TAIL_ABS_THRESHOLD)) {
            result = vFloat(BASIS_RIGHT_TAIL_ABS_A) * x_eval + vFloat(BASIS_RIGHT_TAIL_ABS_B);
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

#if defined(RANGE_REDUCTION_EXP)
        // Exp expand + compose reconstruction, ported verbatim from the generic
        // single-tile piecewise_generic_lut (piecewise_generic.cpp). See the
        // reduce block above for the mirror.
#if defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN)
        // cosh(x): fold the 0.5 scale into the exponent field of BOTH terms --
        // th = P*2^(k-1) = 0.5*exp(|x|), rh = (1/P)*2^(-k-1) = 0.5*exp(-|x|).
        // rh underflows to 0 gracefully via exp_expand's new_exp<=0 guard; th is
        // only reachable once the EXP_OVERFLOW_HYPERBOLIC shoulder routed +inf.
        v_if(x_abs > EXP_OVERFLOW_HYPERBOLIC) { result = std::numeric_limits<float>::infinity(); }
        v_else {
            vFloat recip_poly = ckernel::sfpu::sfpu_reciprocal_iter<2>(result);
            vFloat th = exp_expand(result, k_int - 1);
            vFloat rh = exp_expand(recip_poly, -k_int - 1);
            result = th + rh;
        }
        v_endif;
#elif defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD)
        // sinh(x) = sign(x)*(0.5*exp(|x|) - 0.5*exp(-|x|)). th-rh catastrophically
        // cancels for small |x| (k_int==0, the entire reduced segment), so that
        // region uses a from-scratch odd minimax fit of sinh(x) itself
        // (SINH_SMALL_X_COEFFS, evaluated directly in the already-signed x_orig).
        v_if(x_abs > EXP_OVERFLOW_HYPERBOLIC) {
            result = copysgn(vFloat(std::numeric_limits<float>::infinity()), x_orig);
        }
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
        // silu/swish = x/(1+exp(-x)). Guard both tails: x -> -inf gives the true
        // limit 0 (naive x*recip(inf) = NaN); x > EXP_OVERFLOW gives x*1 = x, which
        // also matches the reduction-valid band [EXP_OVERFLOW, 2^24) bit-for-bit
        // (past 2^24 the magic-number round in exp_reduce loses its round).
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
        // sigmoid(x) = 1/(1+exp(-x)). Same tail guards as silu without the final
        // ``x *`` factor: x -> -inf gives 0, x > EXP_OVERFLOW gives 1.0f (both
        // exact and reduction-band consistent).
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
        // exp2(x) = 2^k * 2^r: reconstructed with the same exact 2^k exponent
        // fold as the plain arm. return_class terminals + exp_expand's new_exp<=0
        // guard own the +0/+inf/underflow shoulders, so no branch is needed here.
        result = exp_expand(result, k_int);
#elif defined(REDUCE_EXP_COMPOSE_ELU) || defined(REDUCE_EXP_COMPOSE_SELU)
        // elu/celu/selu: cancellation-free expm1 reconstruction from the
        // expm1(r)/r reduced leaf (result == P(r) ~= expm1(r)/r):
        //     expm1(x) = (2^k - 1) + 2^k * (r * P(r))
        // (2^k - 1) is EXACT for k<=0 (the whole reconstruction band), so the
        // leading -1 never cancels -- the direct e^x-1 poly floors at ~1500 fp32
        // ULP, this reaches ~1 ULP. `x` is r (exp_reduce output); k_int the
        // Cody-Waite integer. Middle-band value is discarded for x_orig>=0 /
        // x_orig<=tail, so 2^k overflow there is harmless. Mirrors the canonical
        // single-lane cascade arm in piecewise_generic.cpp.
        {
            vFloat reduced_expm1 = result * x;                  // r * P(r) = expm1(r)
            vFloat two_to_k = exp_expand(vFloat(1.0f), k_int);  // 2^k (exact power of two)
            vFloat base = two_to_k - 1.0f;                      // 2^k - 1 (exact for k<=0)
            vFloat expm1_x = base + two_to_k * reduced_expm1;   // expm1(x)
#if defined(REDUCE_EXP_COMPOSE_SELU)
            constexpr float SELU_LAMBDA = 1.0507010221481323f;
            constexpr float SELU_LAMBDA_ALPHA = 1.7580993175506592f;
            constexpr float SELU_SAT_BOUND = -16.869848251342773f;
            result = SELU_LAMBDA_ALPHA * expm1_x;
            v_if(x_orig >= 0.0f) { result = SELU_LAMBDA * x_orig; }
            v_elseif(x_orig <= SELU_SAT_BOUND) { result = -SELU_LAMBDA_ALPHA; }
            v_endif;
#else
            constexpr float ELU_SAT_BOUND = -17.32868003845215f;
            result = expm1_x;
            v_if(x_orig >= 0.0f) { result = x_orig; }
            v_elseif(x_orig <= ELU_SAT_BOUND) { result = -1.0f; }
            v_endif;
#endif
            // Sign-concordance (see piecewise_generic.cpp): elu/celu/selu have
            // sign(f(x))==sign(x); force the output sign from x_orig so -0 and the
            // sign-preserving-DAZ subnormals recover their sign on device (the
            // expm1(-0) reconstruction yields +0 and the >=0 select misses -0).
            result = copysgn(result, x_orig);
        }
#else
        // Plain exp: EXP_UNDERFLOW is local to this arm (the compose arms never
        // underflow to a hard 0.0f the same way).
        constexpr float EXP_UNDERFLOW = -88.5f;
        v_if(x_orig > EXP_OVERFLOW) { result = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig < EXP_UNDERFLOW) { result = 0.0f; }
        v_else { result = exp_expand(result, k_int); }
        v_endif;
#endif
#elif defined(RANGE_REDUCTION_TRIG)
        result = trig_expand(result, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        result = tan_expand(result, j_int);
#elif defined(RANGE_REDUCTION_LOG)
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
        // atan reconstruction: |x|<=1 -> |x|*P; |x|>1 -> pi/2 - (1/|x|)*P; the
        // tail folds to pi/2 EXACTLY as |x|->inf (reduction owns its shoulders).
        // Mirrors the canonical single-lane cascade arm.
        {
            constexpr float RECIP_COMPLEMENT_PI_HALF = 1.5707963267948966f;
            vFloat core_value = recip_reduced * result;
            vFloat magnitude = core_value;
            v_if(setsgn(x_orig, 0) > 1.0f) { magnitude = RECIP_COMPLEMENT_PI_HALF - core_value; }
            v_endif;
            result = copysgn(magnitude, x_orig);
        }
#endif

        // ================================================================
        // Asymptotic factoring: multiply correction polynomial by dominant
        // factor for segments in the tail region. Applied post-evaluation
        // since dominant(x) is independent of the piecewise segment.
        // Mutually exclusive with range reduction (never combined).
        // ================================================================
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        // f(x) = exp(ASYMPTOTIC_EXP_ARG_SCALE * x²) * ASYMPTOTIC_SCALE * correction(x)
        // e.g. GELU tail: exp(-x²/2) * (-1/√(2π)) * poly(x)
        TT_ASYMPTOTIC_IF(x_orig) {
            vFloat t = x_orig * x_orig * ASYMPTOTIC_EXP_ARG_SCALE;
            result = apply_asymptotic_scale(result * asymptotic_exp(t));
        }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_EXP_LINEAR)
        // f(x) = exp(ASYMPTOTIC_EXP_ARG_SCALE * x) * ASYMPTOTIC_SCALE * correction(x)
        // e.g. sigmoid left tail: exp(x) * poly(x)
        TT_ASYMPTOTIC_IF(x_orig) {
            vFloat t = x_orig * ASYMPTOTIC_EXP_ARG_SCALE;
            result = apply_asymptotic_scale(result * asymptotic_exp(t));
        }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
        // f(x) = x * exp(ASYMPTOTIC_EXP_ARG_SCALE * x) * ASYMPTOTIC_SCALE * correction(x)
        // e.g. silu/mish left tail: x * exp(x) * poly(x)
        TT_ASYMPTOTIC_IF(x_orig) {
            vFloat t = x_orig * ASYMPTOTIC_EXP_ARG_SCALE;
            result = apply_asymptotic_scale(result * x_orig * asymptotic_exp(t));
        }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_X)
        // f(x) = x * ASYMPTOTIC_SCALE * correction(x)
        // Trivial factoring (e.g., GELU right tail: x * poly(x))
        TT_ASYMPTOTIC_IF(x_orig) { result = apply_asymptotic_scale(result * x_orig); }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_QUADRATIC)
        // f(x) = (x-A)(x-B) * ASYMPTOTIC_SCALE * P(x). Interior-root factoring (lgamma roots 1,2).
        TT_ASYMPTOTIC_IF(x_orig) {
            result =
                apply_asymptotic_scale(result * (x_orig - ASYMPTOTIC_QUAD_ROOT_A) * (x_orig - ASYMPTOTIC_QUAD_ROOT_B));
        }
        TT_ASYMPTOTIC_END_IF;
#endif

#ifdef HAS_CRITICAL_POINT
        v_if(x == lut[CRITICAL_IDX]) { result = CRITICAL_VALUE; }
        v_endif;
#endif

#if defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) || defined(TT_SELECTED_CORE_TOTAL_FORM)
        finalize_raw_domain_actions(dst_reg[d], result);
#else
        finalize_raw_domain_actions(x_raw, result);
#endif
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        // The source DST row is still untouched here.  Recover raw-class
        // distinctions after the numeric evaluator without extending their
        // live range through Horner evaluation.
        finalize_encoded_raw_domain_actions(dst_reg[d].mode<DataLayout::U16>(), result);
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

// ============================================================================
// Dual-eval recursive unroller (processes 2 DST rows for ILP)
// ============================================================================

#ifdef USE_DUAL_EVAL

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void unroll_segment_dual(
    const std::array<float, LUT_SIZE>& lut,
    vFloat x1c,
    vFloat x2c,
    vFloat x1,
    vFloat x2,
    vFloat& r1,
    vFloat& r2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    ,
    vFloat x1_sq,
    vFloat x2_sq
#endif
) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
#ifdef HAS_SEGMENT_DEGREES
        constexpr uint32_t DEG = SEGMENT_DEGREES[SEG];
#else
        constexpr uint32_t DEG = POLY_DEGREE;
#endif
        {
            vFloat tmp1, tmp2;
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            eval_polynomial_dual_parity<DEG>(&lut[CO + SEG * CPS], x1, x2, x1_sq, x2_sq, tmp1, tmp2);
#else
            eval_polynomial_dual<DEG>(&lut[CO + SEG * CPS], x1, x2, tmp1, tmp2);
#endif
            vFloat b = lut[SEG];
            v_if(x1c >= b) { r1 = tmp1; }
            v_endif;
            v_if(x2c >= b) { r2 = tmp2; }
            v_endif;
        }
        unroll_segment_dual<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x1c,
            x2c,
            x1,
            x2,
            r1,
            r2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            ,
            x1_sq,
            x2_sq
#endif
        );
    }
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_N_dual(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    for (int d = 0; d < 32; d += 2) {
#if defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) || defined(TT_SELECTED_CORE_TOTAL_FORM)
        // The raw inputs already reside in DST until the final stores.  Load
        // only the prepared core coordinates here, then reload raw x at the
        // finalizer.  This severs raw-input liveness across the mature dual
        // evaluator without a scratch store, a second loop, or BF16 rounding.
        vFloat x_orig1 = prepare_raw_domain_input(dst_reg[d]);
        vFloat x_orig2 = prepare_raw_domain_input(dst_reg[d + 1]);
#else
        vFloat x_raw1 = dst_reg[d];
        vFloat x_raw2 = dst_reg[d + 1];
        vFloat x_orig1 = prepare_raw_domain_input(x_raw1);
        vFloat x_orig2 = prepare_raw_domain_input(x_raw2);
#endif

#if defined(RANGE_REDUCTION_EXP)
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
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

        // Dual-eval: skip clamping to avoid SFPU register spill from vec_min_max temporaries.
        // Segment boundaries still gate correctly — out-of-range x just picks the edge segment.
        vFloat& x1_clamped = x1;
        vFloat& x2_clamped = x2;

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
        vFloat x1_sq = x1 * x1;
        vFloat x2_sq = x2 * x2;
#endif

        // Segment 0: dual evaluation for ILP
        vFloat result1, result2;
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
#ifdef HAS_SEGMENT_DEGREES
        eval_polynomial_dual_parity<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x1, x2, x1_sq, x2_sq, result1, result2);
#else
        eval_polynomial_dual_parity<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, x1_sq, x2_sq, result1, result2);
#endif
#else
#ifdef HAS_SEGMENT_DEGREES
        eval_polynomial_dual<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x1, x2, result1, result2);
#else
        eval_polynomial_dual<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);
#endif
#endif

        // Segments 1..NUM_SEGMENTS-1 via recursive unrolling
        unroll_segment_dual<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x1_clamped,
            x2_clamped,
            x1,
            x2,
            result1,
            result2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            ,
            x1_sq,
            x2_sq
#endif
        );

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        result1 = result1 * x1;
        result2 = result2 * x2;
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
        result1 = result1 * basis_sqrt(vFloat(1.0f) - x1);
        result2 = result2 * basis_sqrt(vFloat(1.0f) - x2);
#endif

#if defined(BASIS_AFFINE_EVEN)
        result1 = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig1 + (BASIS_AFFINE_EVEN_SCALE * x1) * result1;
        result2 = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig2 + (BASIS_AFFINE_EVEN_SCALE * x2) * result2;
#endif

#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
        v_if(x1 > vFloat(BASIS_RIGHT_TAIL_ABS_THRESHOLD)) {
            result1 = vFloat(BASIS_RIGHT_TAIL_ABS_A) * x1 + vFloat(BASIS_RIGHT_TAIL_ABS_B);
        }
        v_endif;
        v_if(x2 > vFloat(BASIS_RIGHT_TAIL_ABS_THRESHOLD)) {
            result2 = vFloat(BASIS_RIGHT_TAIL_ABS_A) * x2 + vFloat(BASIS_RIGHT_TAIL_ABS_B);
        }
        v_endif;
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
        v_if(x_orig1 < 0.0f) { result1 = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig1 == 0.0f) { result1 = -std::numeric_limits<float>::infinity(); }
        v_else { result1 = log_expand(result1, e_int1); }
        v_endif;
        v_if(x_orig2 < 0.0f) { result2 = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig2 == 0.0f) { result2 = -std::numeric_limits<float>::infinity(); }
        v_else { result2 = log_expand(result2, e_int2); }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Note: dual-eval is disabled for cbrt (range reduction → single-eval fallback),
        // but this must compile. Uses same biased-exponent pattern as single-eval.
        {
            vFloat tmp1 = dst_reg[d];
            vInt be1 = exexp(setsgn(tmp1, 0), ExponentMode::Biased);
            vInt s1 = as<vInt>(tmp1) & 0x80000000;
            constexpr float OT = 0.3333333333333333f;
            const vFloat mag = ckernel::sfpu::Converter::as_float(0x4B400000U);
            vFloat ef1 = convert<vFloat>(as<vSMag>(be1), RoundMode::Nearest) - 127.0f;
            vFloat qr1 = ef1 * OT + mag;
            vInt q1 = as<vInt>(qr1) - as<vInt>(mag);
            vFloat qb1 = qr1 - mag;
            vInt r1 = as<vInt>(ef1 - (qb1 + qb1 + qb1) + mag) - as<vInt>(mag);
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
            v_if(be1 < 1) { result1 = 0.0f; }
            v_else { result1 = cbrt_expand(result1, q1, r1, s1); }
            v_endif;
        }
        {
            vFloat tmp2 = dst_reg[d + 1];
            vInt be2 = exexp(setsgn(tmp2, 0), ExponentMode::Biased);
            vInt s2 = as<vInt>(tmp2) & 0x80000000;
            constexpr float OT = 0.3333333333333333f;
            const vFloat mag = ckernel::sfpu::Converter::as_float(0x4B400000U);
            vFloat ef2 = convert<vFloat>(as<vSMag>(be2), RoundMode::Nearest) - 127.0f;
            vFloat qr2 = ef2 * OT + mag;
            vInt q2 = as<vInt>(qr2) - as<vInt>(mag);
            vFloat qb2 = qr2 - mag;
            vInt r2 = as<vInt>(ef2 - (qb2 + qb2 + qb2) + mag) - as<vInt>(mag);
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
            v_if(be2 < 1) { result2 = 0.0f; }
            v_else { result2 = cbrt_expand(result2, q2, r2, s2); }
            v_endif;
        }
#endif

        // Asymptotic factoring (dual-eval): apply dominant factor to both lanes
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        TT_ASYMPTOTIC_IF(x_orig1) {
            vFloat t1 = x_orig1 * x_orig1 * ASYMPTOTIC_EXP_ARG_SCALE;
            result1 = apply_asymptotic_scale(result1 * asymptotic_exp(t1));
        }
        TT_ASYMPTOTIC_END_IF;
        TT_ASYMPTOTIC_IF(x_orig2) {
            vFloat t2 = x_orig2 * x_orig2 * ASYMPTOTIC_EXP_ARG_SCALE;
            result2 = apply_asymptotic_scale(result2 * asymptotic_exp(t2));
        }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_EXP_LINEAR)
        TT_ASYMPTOTIC_IF(x_orig1) {
            vFloat t1 = x_orig1 * ASYMPTOTIC_EXP_ARG_SCALE;
            result1 = apply_asymptotic_scale(result1 * asymptotic_exp(t1));
        }
        TT_ASYMPTOTIC_END_IF;
        TT_ASYMPTOTIC_IF(x_orig2) {
            vFloat t2 = x_orig2 * ASYMPTOTIC_EXP_ARG_SCALE;
            result2 = apply_asymptotic_scale(result2 * asymptotic_exp(t2));
        }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
        TT_ASYMPTOTIC_IF(x_orig1) {
            vFloat t1 = x_orig1 * ASYMPTOTIC_EXP_ARG_SCALE;
            result1 = apply_asymptotic_scale(result1 * x_orig1 * asymptotic_exp(t1));
        }
        TT_ASYMPTOTIC_END_IF;
        TT_ASYMPTOTIC_IF(x_orig2) {
            vFloat t2 = x_orig2 * ASYMPTOTIC_EXP_ARG_SCALE;
            result2 = apply_asymptotic_scale(result2 * x_orig2 * asymptotic_exp(t2));
        }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_X)
        TT_ASYMPTOTIC_IF(x_orig1) { result1 = apply_asymptotic_scale(result1 * x_orig1); }
        TT_ASYMPTOTIC_END_IF;
        TT_ASYMPTOTIC_IF(x_orig2) { result2 = apply_asymptotic_scale(result2 * x_orig2); }
        TT_ASYMPTOTIC_END_IF;
#elif defined(ASYMPTOTIC_FACTOR_QUADRATIC)
        TT_ASYMPTOTIC_IF(x_orig1) {
            result1 = apply_asymptotic_scale(
                result1 * (x_orig1 - ASYMPTOTIC_QUAD_ROOT_A) * (x_orig1 - ASYMPTOTIC_QUAD_ROOT_B));
        }
        TT_ASYMPTOTIC_END_IF;
        TT_ASYMPTOTIC_IF(x_orig2) {
            result2 = apply_asymptotic_scale(
                result2 * (x_orig2 - ASYMPTOTIC_QUAD_ROOT_A) * (x_orig2 - ASYMPTOTIC_QUAD_ROOT_B));
        }
        TT_ASYMPTOTIC_END_IF;
#endif

#ifdef HAS_CRITICAL_POINT
        v_if(x1 == lut[CRITICAL_IDX]) { result1 = CRITICAL_VALUE; }
        v_endif;
        v_if(x2 == lut[CRITICAL_IDX]) { result2 = CRITICAL_VALUE; }
        v_endif;
#endif

#if defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) || defined(TT_SELECTED_CORE_TOTAL_FORM)
        finalize_raw_domain_actions(dst_reg[d], result1);
        finalize_raw_domain_actions(dst_reg[d + 1], result2);
#else
        finalize_raw_domain_actions(x_raw1, result1);
        finalize_raw_domain_actions(x_raw2, result2);
#endif
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_encoded_raw_domain_actions(dst_reg[d].mode<DataLayout::U16>(), result1);
        finalize_encoded_raw_domain_actions(dst_reg[d + 1].mode<DataLayout::U16>(), result2);
#endif

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

// ============================================================================
// Coefficient-blend single-eval (parity only)
// ============================================================================
//
// Instead of evaluating a full x²-Horner inside each segment's v_if (the
// O(segments x degree) cascade in piecewise_generic_lut_specialized_N), the
// blend cascade only OVERWRITES the parity coefficient registers with cheap
// predicated moves per segment, then runs ONE x²-Horner on the blended
// coefficients after the cascade. This collapses the dominant term:
//   NUM_SEGMENTS x (full Horner)  ->  NUM_SEGMENTS x (few moves) + 1 x (Horner)
//
// BIT-EQUIVALENCE: for an input in segment s, the cascade selects exactly the
// coefficient set of the highest boundary <= x (segment 0 is the initial set,
// each later v_if that fires overwrites it). The final x²-Horner is evaluated
// at a FIXED top index = parity-top of POLY_DEGREE (the max degree). Per-segment
// adaptive degree leaves the unused high-order coefficient slots zero in the
// LUT, and a leading `0*x2 + c` step is exact in float — so evaluating at the
// max top with zeroed high coeffs is bit-identical to the shorter per-segment
// x²-Horner used by the cascade path. ODD parity multiplies the final
// accumulator by x; EVEN parity does not (matching eval_polynomial_parity).
//
// Single-eval only — blend + dual would keep two coefficient sets live and
// overflow the SFPU register file.
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)

// Parity layout: ODD uses coefficient indices 1,3,5,... ; EVEN uses 0,2,4,...
// TOP = highest in-parity index <= POLY_DEGREE; NSLOTS parity coefficients,
// where slot j holds coefficient index (PARITY_BASE + 2*j).
template <uint32_t POLY_DEGREE>
struct blend_parity_traits {
#if defined(POLY_PARITY_ODD)
    static constexpr int PARITY_BASE = 1;
    static constexpr int TOP = (POLY_DEGREE % 2 == 1) ? (int)POLY_DEGREE : (int)POLY_DEGREE - 1;
#else
    static constexpr int PARITY_BASE = 0;
    static constexpr int TOP = (POLY_DEGREE % 2 == 0) ? (int)POLY_DEGREE : (int)POLY_DEGREE - 1;
#endif
    static constexpr uint32_t NSLOTS = (uint32_t)((TOP - PARITY_BASE) / 2) + 1;
};

// Recursively overwrite the NSLOTS blended parity coefficients for segment SEG
// when x >= boundary[SEG]. Slot j <- lut[base + PARITY_BASE + 2*j].
template <uint32_t SLOT, uint32_t NSLOTS, int PARITY_BASE, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void blend_assign(
    const std::array<float, LUT_SIZE>& lut, uint32_t base, vFloat* coeffs) {
    if constexpr (SLOT < NSLOTS) {
        coeffs[SLOT] = lut[base + PARITY_BASE + 2 * SLOT];
        blend_assign<SLOT + 1, NSLOTS, PARITY_BASE, LUT_SIZE>(lut, base, coeffs);
    }
}

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void blend_cascade(
    const std::array<float, LUT_SIZE>& lut, vFloat x_clamped, vFloat* coeffs) {
    if constexpr (SEG < NUM_SEGMENTS) {
        using T = blend_parity_traits<POLY_DEGREE>;
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
        constexpr uint32_t base = CO + SEG * CPS;
        v_if(x_clamped >= lut[SEG]) { blend_assign<0, T::NSLOTS, T::PARITY_BASE, LUT_SIZE>(lut, base, coeffs); }
        v_endif;
        blend_cascade<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x_clamped, coeffs);
    }
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_N_blend(const std::array<float, LUT_SIZE>& lut) {
    using T = blend_parity_traits<POLY_DEGREE>;
    constexpr uint32_t CPS = POLY_DEGREE + 1;
    constexpr uint32_t CO = NUM_SEGMENTS + 1;
    constexpr uint32_t NSLOTS = T::NSLOTS;

    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x_orig = prepare_raw_domain_input(x_raw);
#if defined(BASIS_INPUT_ABS_X)
        vFloat x = setsgn(x_orig, 0);
#else
        vFloat x = x_orig;
#endif

        // Blended parity coefficients, initialized to segment 0's coefficients.
        vFloat coeffs[NSLOTS];
        blend_assign<0, NSLOTS, T::PARITY_BASE, LUT_SIZE>(lut, CO, coeffs);

        // Cascade: overwrite coefficient registers for the matching segment.
        blend_cascade<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x, coeffs);

        // ONE x²-Horner on the blended coefficients (fixed top = parity-top of
        // POLY_DEGREE). Highest slot first, matching eval_polynomial_parity.
        vFloat x2 = x * x;
        vFloat acc = coeffs[NSLOTS - 1];
#pragma GCC unroll 16
        for (int k = (int)NSLOTS - 2; k >= 0; k--) {
            acc = acc * x2 + coeffs[k];
        }
#if defined(POLY_PARITY_ODD)
        acc = acc * x;  // final *x for odd parity
#endif

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        acc = acc * x;
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
        acc = acc * basis_sqrt(vFloat(1.0f) - x);
#endif

#if defined(BASIS_AFFINE_EVEN)
        acc = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig + (BASIS_AFFINE_EVEN_SCALE * x) * acc;
#endif

#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
        v_if(x > vFloat(BASIS_RIGHT_TAIL_ABS_THRESHOLD)) {
            acc = vFloat(BASIS_RIGHT_TAIL_ABS_A) * x + vFloat(BASIS_RIGHT_TAIL_ABS_B);
        }
        v_endif;
#endif

#if defined(BASIS_CLAMP_MAX)
        vFloat basis_clamp_max_value = BASIS_CLAMP_MAX_VALUE;
        ordered_min_max(acc, basis_clamp_max_value);
#endif

#if defined(BASIS_POST_SIGN_X)
        acc = copysgn(acc, x_orig);
#endif

#if defined(BASIS_POST_REFLECT_PI)
        v_if(x_orig < 0.0f) { acc = BASIS_PI_VALUE - acc; }
        v_endif;
#endif

#if defined(BASIS_LEFT_TAIL_ZERO)
        v_if(x_orig < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { acc = 0.0f; }
        v_endif;
#endif

#if defined(BASIS_RIGHT_TAIL_IDENTITY)
        v_if(x_orig > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { acc = x_orig; }
        v_endif;
#endif

#ifdef HAS_CRITICAL_POINT
        v_if(x == lut[CRITICAL_IDX]) { acc = CRITICAL_VALUE; }
        v_endif;
#endif

        finalize_raw_domain_actions(x_raw, acc);

        // bf16 dst: RNE-round before the store (see single-eval note above).
#ifdef USE_BF16
        acc = convert<vFloat16b>(acc, RoundMode::Nearest);
#endif
        dst_reg[d] = acc;
    }
}

#endif  // POLY_PARITY_ODD || POLY_PARITY_EVEN

// ============================================================================
// Coefficient-blend single-eval (NON-parity)
// ============================================================================
//
// Same select-then-eval technique as the parity blend above, but for general
// (non-parity) per-segment polynomials — which is what real activation fits
// produce (only the global function is odd/even; the local segment polynomials
// are dense). The v_if cascade overwrites ALL (POLY_DEGREE+1) coefficient
// registers with cheap predicated moves, then ONE dense Horner runs after the
// cascade on the blended coefficients:
//   NUM_SEGMENTS x (full Horner)  ->  NUM_SEGMENTS x (degree+1 moves) + 1 x Horner
//
// BIT-EQUIVALENCE: for an input in segment s, the cascade selects exactly the
// coefficient set of the highest boundary <= x (segment 0 is the initial set;
// each later v_if that fires overwrites it). The final dense Horner is always
// evaluated at the FIXED top index POLY_DEGREE. With adaptive per-segment
// degree (HAS_SEGMENT_DEGREES), low-degree segments leave their high-order
// coefficient slots zero in the LUT; a leading `0*x + c` Horner step is exact
// in float, so evaluating at the max top with zeroed high coeffs is bit-
// identical to the shorter per-segment Horner used by the cascade path. This
// matches eval_polynomial<DEG>, which Horners coefficient index DEG..0 from
// &lut[base] (index i at lut[base+i]).
//
// Single-eval only — blend + dual would keep two coefficient sets live and
// overflow the SFPU register file. Gated to POLY_DEGREE <= 5 in the dispatcher
// (degree >= 6 ICEs GCC reload with POLY_DEGREE+1 live coefficient registers).
#if !defined(POLY_PARITY_ODD) && !defined(POLY_PARITY_EVEN)

// Recursively overwrite the (POLY_DEGREE+1) blended coefficient registers for
// segment SEG when x >= boundary[SEG]. Slot i <- lut[base + i].
template <uint32_t SLOT, uint32_t NSLOTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void blend_assign_dense(
    const std::array<float, LUT_SIZE>& lut, uint32_t base, vFloat* coeffs) {
    if constexpr (SLOT < NSLOTS) {
        coeffs[SLOT] = lut[base + SLOT];
        blend_assign_dense<SLOT + 1, NSLOTS, LUT_SIZE>(lut, base, coeffs);
    }
}

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void blend_cascade_dense(
    const std::array<float, LUT_SIZE>& lut, vFloat x_clamped, vFloat* coeffs) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
        constexpr uint32_t base = CO + SEG * CPS;
        v_if(x_clamped >= lut[SEG]) { blend_assign_dense<0, POLY_DEGREE + 1, LUT_SIZE>(lut, base, coeffs); }
        v_endif;
        blend_cascade_dense<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x_clamped, coeffs);
    }
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_N_blend_dense(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t CPS = POLY_DEGREE + 1;
    constexpr uint32_t CO = NUM_SEGMENTS + 1;
    constexpr uint32_t NSLOTS = POLY_DEGREE + 1;

    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x_orig = prepare_raw_domain_input(x_raw);
#if defined(BASIS_INPUT_ABS_X)
        vFloat x = setsgn(x_orig, 0);
#else
        vFloat x = x_orig;
#endif

        // Blended coefficients, initialized to segment 0's coefficients.
        vFloat coeffs[NSLOTS];
        blend_assign_dense<0, NSLOTS, LUT_SIZE>(lut, CO, coeffs);

        // Cascade: overwrite coefficient registers for the matching segment.
        blend_cascade_dense<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x, coeffs);

        // ONE dense Horner on the blended coefficients (fixed top = POLY_DEGREE).
        // Highest index first, matching eval_polynomial<DEG>.
        vFloat acc = coeffs[NSLOTS - 1];
#pragma GCC unroll 16
        for (int k = (int)NSLOTS - 2; k >= 0; k--) {
            acc = acc * x + coeffs[k];
        }

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        acc = acc * x;
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
        acc = acc * basis_sqrt(vFloat(1.0f) - x);
#endif

#if defined(BASIS_AFFINE_EVEN)
        acc = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig + (BASIS_AFFINE_EVEN_SCALE * x) * acc;
#endif

#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
        v_if(x > vFloat(BASIS_RIGHT_TAIL_ABS_THRESHOLD)) {
            acc = vFloat(BASIS_RIGHT_TAIL_ABS_A) * x + vFloat(BASIS_RIGHT_TAIL_ABS_B);
        }
        v_endif;
#endif

#if defined(BASIS_CLAMP_MAX)
        vFloat basis_clamp_max_value = BASIS_CLAMP_MAX_VALUE;
        ordered_min_max(acc, basis_clamp_max_value);
#endif

#if defined(BASIS_POST_SIGN_X)
        acc = copysgn(acc, x_orig);
#endif

#if defined(BASIS_POST_REFLECT_PI)
        v_if(x_orig < 0.0f) { acc = BASIS_PI_VALUE - acc; }
        v_endif;
#endif

#if defined(BASIS_LEFT_TAIL_ZERO)
        v_if(x_orig < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { acc = 0.0f; }
        v_endif;
#endif

#if defined(BASIS_RIGHT_TAIL_IDENTITY)
        v_if(x_orig > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { acc = x_orig; }
        v_endif;
#endif

#ifdef HAS_CRITICAL_POINT
        v_if(x == lut[CRITICAL_IDX]) { acc = CRITICAL_VALUE; }
        v_endif;
#endif

        finalize_raw_domain_actions(x_raw, acc);

        // bf16 dst: RNE-round before the store (see single-eval note above).
#ifdef USE_BF16
        acc = convert<vFloat16b>(acc, RoundMode::Nearest);
#endif
        dst_reg[d] = acc;
    }
}

#endif  // !POLY_PARITY_ODD && !POLY_PARITY_EVEN

// ============================================================================
// Dispatcher
// ============================================================================

// Blend gate (shared preconditions): multi-segment is where the cascade->blend
// win materializes, and range-reduction / asymptotic factoring hold extra live
// registers (vInt exponents, x_orig, etc.) that would push blend over the
// register frontier. These hold for both parity and non-parity blends.
#if !defined(EVAL_METHOD_REDUCED_POLY) && !EVAL_METHOD_IS_STANDALONE && !defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) && \
    !defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) && !defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) &&                            \
    !defined(ASYMPTOTIC_FACTOR_X) && !defined(ASYMPTOTIC_FACTOR_QUADRATIC)
#define BLEND_GATE_NO_REDUCTION 1
#else
#define BLEND_GATE_NO_REDUCTION 0
#endif

// Per-variant blend eligibility + register-fit bound:
//   parity:     POLY_DEGREE <= 12  (x²-Horner halves live coeffs; >=16 ICEs GCC reload)
//   non-parity: POLY_DEGREE <= 5   (POLY_DEGREE+1 live coeff regs; degree >= 6 ICEs reload)
#if BLEND_GATE_NO_REDUCTION && (defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN))
#define BLEND_GATE_ELIGIBLE 1
#define BLEND_DEGREE_LIMIT 12
#elif BLEND_GATE_NO_REDUCTION
#define BLEND_GATE_ELIGIBLE 1
#define BLEND_DEGREE_LIMIT 5
#else
#define BLEND_GATE_ELIGIBLE 0
#define BLEND_DEGREE_LIMIT 0
#endif

// Compile-time cascade work estimate (FMA-equivalent count for the in-v_if
// evaluation): with adaptive degree it is the sum of per-segment effective
// degrees; otherwise NUM_SEGMENTS * POLY_DEGREE. Both paths pay the same
// NUM_SEGMENTS predicate cascade, so this captures the part that differs.
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS>
constexpr uint32_t blend_cascade_eval_cost() {
#ifdef HAS_SEGMENT_DEGREES
    uint32_t s = 0;
    for (uint32_t i = 0; i < NUM_SEGMENTS; i++) {
        s += SEGMENT_DEGREES[i];
    }
    return s;
#else
    return NUM_SEGMENTS * POLY_DEGREE;
#endif
}

// Auto-select predictor: should the (eligible) blend path actually be used?
// "Best design wins" — engage blend only when it is the faster design.
//
//   PARITY blend is the established win: parity x²-Horner halves the live
//   coefficient count, so the per-segment predicated moves (~NSLOTS = DEGREE/2)
//   are cheaper than the cascade's per-segment x²-Horner. Engages whenever
//   eligible.
//
//   NON-PARITY dense blend overwrites POLY_DEGREE+1 coefficient registers per
//   segment, then runs one dense Horner. The cascade it replaces evaluates
//   eval_polynomial<SEGMENT_DEGREES[s]> under predication for every segment, so
//   the work it removes is blend_cascade_eval_cost() FMAs; the work it adds is
//   NUM_SEGMENTS*(POLY_DEGREE+1) predicated coefficient moves + one trailing
//   POLY_DEGREE Horner. Because the embedded LUT is constexpr, GCC folds every
//   coefficient into an FMA immediate, so a cascade FMA and a blend coefficient
//   move cost about the same — and since blend_cascade_eval_cost() <=
//   NUM_SEGMENTS*POLY_DEGREE < NUM_SEGMENTS*(POLY_DEGREE+1), the cascade is
//   always the faster design here. Measured on Blackhole (256 tiles, bf16):
//   the non-parity blend is bit-identical to the cascade but never faster
//   (tanh p5_s32 24.1 vs 16.9; sinh p4_s32 25.9 vs 22.5; sigmoid p4_s16
//   dense ratio 0.98 14.5 vs 13.5). So this predicate is FALSE for every
//   non-parity config and routes them to the cascade. The non-parity blend
//   path remains available and correct for harnesses where coefficients are
//   not constexpr-folded (e.g. a runtime LUT), where the load/move asymmetry
//   flips the trade — there the predicate fires once the cascade work exceeds
//   the blend's move cost.
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS>
constexpr bool blend_predicted_faster() {
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    return true;
#else
    return blend_cascade_eval_cost<POLY_DEGREE, NUM_SEGMENTS>() > NUM_SEGMENTS * (POLY_DEGREE + 1) + POLY_DEGREE;
#endif
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
#if defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD) && defined(POLY_TTI_DISABLE)
    (void)lut;
    poly_log_square_factorized_odd_sfpi_tile();
    return;
#endif
#if EVAL_METHOD_IS_STANDALONE
    // EVAL_METHOD_EXPONENT_ALU / EVAL_METHOD_NEWTON_ROOT are standalone evaluators
    // — each owns the entire approximation and ignores the piecewise segment
    // cascade. Route directly to the standalone evaluator and return.
    piecewise_generic_lut_hw_reduce<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    return;
#endif
#if POLY_TTI_CANDIDATE
    // TTI replay lowering for the supported single-segment shapes (see the
    // kPolyTtiReplay block in piecewise_generic.cpp). Reads the constexpr
    // embedded LUT_DATA directly; when the gate is false the sfpi dispatch
    // below compiles unchanged.
    if constexpr (kPolyTtiReplay) {
        (void)lut;
#if defined(TT_SELECTED_TTI_RAW_SHADOW_LATE_SCHEDULE)
        poly_tti_preserve_raw_shadow_tile();
#endif
        poly_tti_replay_tile();
#if defined(TT_SELECTED_TTI_RAW_SHADOW_LATE_SCHEDULE)
        poly_tti_finalize_raw_shadow_tile();
#endif
        return;
    }
#endif
#if BLEND_GATE_ELIGIBLE
    if constexpr (
        NUM_SEGMENTS > 1 && POLY_DEGREE <= BLEND_DEGREE_LIMIT && blend_predicted_faster<POLY_DEGREE, NUM_SEGMENTS>()) {
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
        piecewise_generic_lut_specialized_N_blend<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#else
        piecewise_generic_lut_specialized_N_blend_dense<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#endif
        return;
    }
#endif
#ifdef USE_DUAL_EVAL
// Range reduction keeps extra vInt registers live across the polynomial evaluation,
// pushing total SFPU register pressure beyond the hardware limit.
// Fall back to single-eval when reduce-then-poly (REDUCED_POLY) is active.
#if defined(TT_SELECTED_CORE_TOTAL_INPLACE_FACTORED_OVERLAY_V1)
    // This lowering contract admits exactly the symmetric P8/S2 factored
    // selected core proved by S55.  Keep the mature dual polynomial body and
    // reload raw x only in its same-row finalizer; the direct-log tail and
    // typed raw terminals therefore add no replay/second row loop.
    static_assert(POLY_DEGREE == 8u && NUM_SEGMENTS == 2u);
#if !defined(ASYMPTOTIC_FACTOR_X)
#error "in-place factored selected-core overlay requires factor-x reconstruction"
#endif
    piecewise_generic_lut_specialized_N_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#elif defined(TT_SELECTED_CORE_TOTAL_COMPACT_LOOP_ROW)
    // This typed selected-core form carries a direct-log tail whose predicate
    // tree remains over the SFPI compiler's practical capacity when combined
    // with the recursively unrolled multi-segment evaluator.  The mature
    // loop-row evaluator preserves the CSV arithmetic and terminal overlay
    // while keeping only one segment body live in the compiler IR.
    piecewise_generic_lut<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#elif defined(TT_SELECTED_CORE_TOTAL_SINGLE_ROW)
    piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#elif defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) && defined(DST_COEFF_DISABLE)
    // This verifier-emitted descriptor binds a terminal overlay whose
    // cross-evaluator liveness exceeds the physical SFPU register file.  It
    // describes the Program schedule, not the coefficient transport.  When
    // the existing byte-equivalence control explicitly disables that
    // transport, retain the descriptor's single-row schedule instead of
    // reintroducing the over-capacity ordinary dual overlay.  The mature
    // single-lane evaluator keeps the selected CSV and per-lane operation
    // order unchanged while retiring one lane before the terminal overlay.
    piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#elif defined(TT_INLINE_PROGRAM_SELECTED_FACTOR_ACTION_OVERLAY)
    // The certified selected-factor terminal needs an endpoint-clamped core
    // coordinate plus the same-row raw reload.  Keep that schedule single-lane:
    // dual evaluation cannot carry both core results, both endpoint clamps,
    // and the terminal action immediate in the physical SFPU register file.
    piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#elif defined(EVAL_METHOD_REDUCED_POLY)
    piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#elif defined(BASIS_INPUT_ABS_X)
    // signed_abs/odd_factored/affine-even basis keeps abs(x), original-sign
    // postprocess, and optional clamp/mul temporaries live. The compact
    // one-segment basis kernels fit dual-eval and recover the ILP win; larger
    // cascades stay on single-eval because they can trip SFPI reload spills.
    if constexpr (NUM_SEGMENTS == 1 && POLY_DEGREE <= 10) {
        piecewise_generic_lut_specialized_N_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    } else {
        piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    }
#elif defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    // Parity x²-Horner threads x² through BOTH dual lanes; at higher degree the combined
    // live-register count overflows the SFPU register file and crashes GCC's reload pass
    // (ICE: "maximum number of generated reload insns"). Fall back to single-eval for
    // high-degree parity; low-degree parity (<=4) still fits and keeps the dual-eval win.
    if constexpr (POLY_DEGREE > 4) {
        piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    } else {
        piecewise_generic_lut_specialized_N_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    }
#else
    piecewise_generic_lut_specialized_N_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#endif
#else
    piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#endif
}
