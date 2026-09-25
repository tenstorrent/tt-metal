// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
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
#if defined(TT_SELECTED_CORE_TOTAL_INPLACE_FACTORED_OVERLAY_V1)
        static_assert(NUM_SEGMENTS == 2 && POLY_DEGREE == 8);
        symmetric_factored_core_dual<POLY_DEGREE,
#if defined(TT_WH_FACTORED_CORE_MIRROR_FOLD)
            true
#else
            false
#endif
        >(&lut[COEFF_OFFSET], &lut[COEFF_OFFSET + COEFFS_PER_SEGMENT], lut[1],
          x1, x2, result1, result2);
#else
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

#endif

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        result1 = result1 * x1;
        result2 = result2 * x2;
#endif

#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
        result1 = sqrt_factored_product(result1, x1);
        result2 = sqrt_factored_product(result2, x2);
#endif

#if defined(BASIS_AFFINE_EVEN)
        result1 = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig1 + (BASIS_AFFINE_EVEN_SCALE * x1) * result1;
        result2 = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig2 + (BASIS_AFFINE_EVEN_SCALE * x2) * result2;
#endif

#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
        signed_abs_affine_tail<CanonicalSignedAbsAffineTail>(x1, result1);
        signed_abs_affine_tail<CanonicalSignedAbsAffineTail>(x2, result2);
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
        result1 = sqrt_factored_reflect(x_orig1, result1);
        result2 = sqrt_factored_reflect(x_orig2, result2);
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
#ifdef TT_LINEAR_BF16_UNDERFLOW_COUNT
        restore_bf16_linear_underflow(dst_reg[d].mode<DataLayout::U16>(), result1);
#endif
        result1 = convert<vFloat16b>(result1, RoundMode::Nearest);
#ifdef TT_LINEAR_BF16_UNDERFLOW_COUNT
        restore_bf16_linear_underflow(dst_reg[d + 1].mode<DataLayout::U16>(), result2);
#endif
        result2 = convert<vFloat16b>(result2, RoundMode::Nearest);
#endif
#if defined(TT_SELECTED_ZONE_GRADIENT)
        result1 = finalize_zone_gradient(dst_reg[d], dst_reg[d].mode<DataLayout::U16>(), result1, dst_reg[32 + d]);
        result1 = convert<vFloat16b>(result1, RoundMode::Nearest);
        result2 = finalize_zone_gradient(dst_reg[d + 1], dst_reg[d + 1].mode<DataLayout::U16>(), result2, dst_reg[33 + d]);
        result2 = convert<vFloat16b>(result2, RoundMode::Nearest);
#endif
        dst_reg[d] = result1;
        dst_reg[d + 1] = result2;
    }
}

#endif  // USE_DUAL_EVAL
