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
#if defined(TT_WH_SIGNED_ABS_COEFFICIENT_STORE)
    static_assert(NUM_SEGMENTS == 1 && POLY_DEGREE == 8);
    static_assert(__builtin_bit_cast(uint32_t, LUT_DATA[COEFF_OFFSET]) == 0u);
    static_assert(TT_SIGNED_ABS_COEFF_BASE == 64u && TT_SIGNED_ABS_COEFF_COUNT == 8u);
    static_assert(TT_SIGNED_ABS_COEFF_BASE + TT_SIGNED_ABS_COEFF_COUNT <= TT_SIGNED_ABS_COEFF_LIMIT);
#if !defined(ARCH_WORMHOLE) || !defined(USE_BF16) || defined(FUSE_GRAD_MUL)
#error "signed-abs coefficient store requires its unary WH BF16 owner"
#endif
#pragma GCC unroll 16
    for (uint32_t i = 1; i <= POLY_DEGREE; ++i) {
        dst_reg[TT_SIGNED_ABS_COEFF_BASE - 1 + i].mode<DataLayout::F32>() = vFloat(lut[COEFF_OFFSET + i]);
    }
#endif
#if defined(TT_WH_SIGNED_ABS_UNROLL32) && defined(TT_WH_SIGNED_ABS_COEFFICIENT_STORE)
#pragma GCC unroll 32
#endif
#if defined(TT_WH_REDUCED_POLY_UNROLL32)
#if !defined(ARCH_WORMHOLE) || !defined(USE_BF16) || !defined(REDUCE_RECIP_COMPLEMENT) || !defined(TT_WH_RAW_NAN_UNION) || defined(FUSE_GRAD_MUL)
#error "reduced-polynomial row scheduling requires its unary WH BF16 owner"
#endif
static_assert(POLY_DEGREE == 3 && NUM_SEGMENTS == 1);
#pragma GCC unroll 32
#endif
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
        vFloat recip_reduced;
        vFloat x = reciprocal_complement_reduce(x_orig, recip_reduced,
            [](vFloat value) { return ckernel::sfpu::sfpu_reciprocal_iter<2>(value); });
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
#if defined(TT_WH_SIGNED_ABS_COEFFICIENT_STORE)
        vFloat result = dst_reg[TT_SIGNED_ABS_COEFF_BASE - 1 + POLY_DEGREE].mode<DataLayout::F32>();
#pragma GCC unroll 16
        for (int i = POLY_DEGREE - 1; i >= 1; --i) {
            vFloat c = dst_reg[TT_SIGNED_ABS_COEFF_BASE - 1 + i].mode<DataLayout::F32>();
            result = sfpu_mad(result, x_eval, c);
        }
        result = sfpu_mad(result, x_eval, 0.0f);
#elif defined(HAS_SEGMENT_DEGREES)
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
        result = sqrt_factored_product(result, x_eval);
#endif

#if defined(BASIS_AFFINE_EVEN)
        result = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig + (BASIS_AFFINE_EVEN_SCALE * x_eval) * result;
#endif

#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
        signed_abs_affine_tail<CanonicalSignedAbsAffineTail>(x_eval, result);
#endif

#if defined(BASIS_CLAMP_MAX)
        vFloat basis_clamp_max_value = BASIS_CLAMP_MAX_VALUE;
        ordered_min_max(result, basis_clamp_max_value);
#endif

#if defined(BASIS_POST_SIGN_X)
        result = copysgn(result, x_orig);
#endif

#if defined(BASIS_POST_REFLECT_PI)
        result = sqrt_factored_reflect(x_orig, result);
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
        result = reciprocal_complement_expand(x_orig, recip_reduced, result);
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
#ifdef TT_LINEAR_BF16_UNDERFLOW_COUNT
        restore_bf16_linear_underflow(dst_reg[d].mode<DataLayout::U16>(), result);
#endif
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
#if defined(ARCH_WORMHOLE) && defined(TT_WH_CORE_SAME_ROW_GRAD_FINALIZE)
#if !defined(TT_SELECTED_CORE_TOTAL_SINGLE_ROW) || !defined(FUSE_GRAD_MUL) || !defined(USE_BF16)
#error "WH same-row gradient requires the selected BF16 single-row derivative"
#endif
        // Retain the original derivative rounding before consuming the
        // incoming BF16 gradient. Only the separate destination walk goes away.
        result = result * dst_reg[32 + d];
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
#if defined(TT_SELECTED_ZONE_GRADIENT)
        result = finalize_zone_gradient(dst_reg[d], dst_reg[d].mode<DataLayout::U16>(), result, dst_reg[32 + d]);
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
        dst_reg[d] = result;
    }
}
