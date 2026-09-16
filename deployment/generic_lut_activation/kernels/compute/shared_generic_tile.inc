// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared whole-tile evaluation: include after copying input into DST tile 0.
// Caller owns LUT context, gradient placement and final pack/release.

// All degree parameters are constexpr — use them directly as template args.
// No dispatch table needed; works for ANY poly_degree automatically.
#ifdef TRISC_MATH
#if defined(TT_TARGET_BUILTIN_PERIODIC_SINE)
        (void)p_lut;
        ckernel::sin_tile(0);
#elif defined(TT_TARGET_BUILTIN_PERIODIC_COSINE)
        (void)p_lut;
        ckernel::cos_tile(0);
#elif defined(TT_TARGET_BUILTIN_PERIODIC_TANGENT)
        (void)p_lut;
        // TTNN's unary chain invokes the target initializer immediately before
        // each tangent call. Tangent mutates programmable state, so a one-time
        // kernel initializer is not semantically equivalent across 64 tiles.
        ckernel::tan_tile_init();
        ckernel::tan_tile(0);
#elif defined(TT_FACTORED_CW_REPLAY)
        (void)p_lut;
        sfpi::integer_exponent_residual_replay_tile<
            poly_degree, num_segments, lut_size,
            __builtin_bit_cast(uint32_t,
                               LUT_DATA[sfpi::TT_FACTORED_CW_REPLAY_C2_INDEX]),
            __builtin_bit_cast(uint32_t,
                               LUT_DATA[sfpi::TT_FACTORED_CW_REPLAY_C3_INDEX]),
            __builtin_bit_cast(uint32_t,
                               LUT_DATA[sfpi::TT_FACTORED_CW_REPLAY_C4_INDEX]),
            sfpi::TT_FACTORED_CW_REPLAY_BODY_SLOTS,
            sfpi::TT_FACTORED_CW_REPLAY_SUFFIX_SLOTS,
            sfpi::TT_FACTORED_CW_REPLAY_ROWS>();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_IDENTITY || defined(AFFINE_IDENTITY) || defined(PACK_RELU_MODE)
        // Identity collapse (y = x): copy_tile already placed x in dst; the SFPU
        // eval is a pure no-op, so skip it entirely (no Horner, no LUT, no cascade).
        (void)p_lut;
#if defined(TT_DOMAIN_ACTION_PROGRAM) || defined(TT_INLINE_PROGRAM_OVERLAY)
        sfpi::identity_domain_actions_eval();
#endif
#if defined(TT_TARGET_BH_BF16_PACK_RELU_RAW_NEG_NAN_REPAIR)
#pragma GCC unroll 8
        for (int d = 0; d < 32; d++) {
            sfpi::vUInt raw_u16 =
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
            sfpi::vFloat repaired = sfpi::dst_reg[d];
            sfpi::apply_pack_relu_raw_neg_nan_repair(raw_u16, repaired);
            sfpi::dst_reg[d] = repaired;
        }
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_AFFINE || defined(AFFINE_COLLAPSE)
        // Affine collapse (y = c0 + c1*x): one SFPMAD per element, no cascade.
        (void)p_lut;
        sfpi::affine_collapse_eval();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_CLAMPED_AFFINE || defined(CLAMPED_AFFINE_COLLAPSE)
        // Clamped affine collapse: one SFPMAD plus optional min/max clamps.
        (void)p_lut;
#if defined(USE_BF16) && defined(CLAMPED_AFFINE_BF16_NATIVE_RELU) && \
    defined(TT_INLINE_PROGRAM_OVERLAY)
#error "typed core overlay requires route-aware native ReLU lowering; refusing evaluator demotion"
#elif defined(USE_BF16) && defined(CLAMPED_AFFINE_BF16_NATIVE_RELU) && \
    !defined(TT_DOMAIN_ACTION_PROGRAM)
        ckernel::relu_tile(0);
#elif DIRECT_ALGEBRAIC_TTI_CANDIDATE
        if constexpr (sfpi::kDirectTtiClampedAffineSymmetricIdentity2x) {
            sfpi::direct_tti_symmetric_identity_clamp_tile();
        } else if constexpr (sfpi::kDirectTtiClampedAffineUnitBounds2x) {
            sfpi::direct_tti_clamped_affine_tile();
        } else {
            sfpi::clamped_affine_collapse_eval();
        }
#else
        sfpi::clamped_affine_collapse_eval();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_VALUE || defined(ABS_VALUE)
        (void)p_lut;
#if DIRECT_ALGEBRAIC_TTI_CANDIDATE
        if constexpr (sfpi::kDirectTtiAbs2x) {
            sfpi::direct_tti_abs_tile();
        } else {
            sfpi::abs_value_eval();
        }
#else
        sfpi::abs_value_eval();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_THRESHOLD_IDENTITY || defined(THRESHOLD_IDENTITY_SELECT)
        (void)p_lut;
#if DIRECT_ALGEBRAIC_TTI_CANDIDATE
        if constexpr (sfpi::kDirectTtiThresholdIdentity2x) {
            sfpi::direct_tti_threshold_identity_tile();
        } else {
            sfpi::threshold_identity_select_eval();
        }
#elif defined(USE_BF16) && \
    defined(TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_EXACT_STORE)
        ckernel::hardshrink_tile(
            0, __builtin_bit_cast(
                   uint32_t, (float)(THRESHOLD_IDENTITY_LAMBDA)));
#else
        sfpi::threshold_identity_select_eval();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_THRESHOLD_SOFTSHIFT || defined(THRESHOLD_SOFTSHIFT_SELECT)
        (void)p_lut;
#if defined(USE_BF16) && \
    defined(TT_TARGET_BH_BF16_THRESHOLD_SOFTSHIFT_NATIVE)
        ckernel::softshrink_tile(
            0, __builtin_bit_cast(
                   uint32_t, (float)(THRESHOLD_SOFTSHIFT_LAMBDA)));
#else
        sfpi::threshold_softshift_select_eval();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_GATED_AFFINE_PRODUCT || defined(GATED_AFFINE_PRODUCT) || defined(GATED_QUADRATIC_COLLAPSE)
        (void)p_lut;
#if GATED_SINGLE_ROW_JOINT_TTI_CANDIDATE
        if constexpr (sfpi::kGatedSingleRowJointTtiReplay) {
            sfpi::gated_single_row_joint_tti_tile();
        } else {
            sfpi::gated_affine_product_eval();
        }
#elif DIRECT_ALGEBRAIC_TTI_CANDIDATE
        if constexpr (sfpi::kDirectTtiGatedAffineProduct2x) {
            sfpi::direct_tti_gated_affine_product_tile();
        } else {
            sfpi::gated_affine_product_eval();
        }
#else
        sfpi::gated_affine_product_eval();
#endif
#elif (TT_ACT_EVAL_KIND == TT_ACT_EVAL_SLOPE_MAX || defined(SLOPE_MAX)) && SLOPE_MAX_ELIGIBLE
        (void)p_lut;
#if DIRECT_ALGEBRAIC_TTI_CANDIDATE
#if defined(SLOPE_MAX) && defined(SLOPE_MAX_IS_MAX) && \
    defined(TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI)
        if constexpr (sfpi::kDirectTtiSlopeMax2x) {
            sfpi::direct_tti_unit_high_slope_tile();
        } else {
            sfpi::slope_max_eval();
        }
#else
        // The direct helper is declared only for a certified unit-high-slope
        // schedule.  `if constexpr (false)` still performs non-dependent name
        // lookup, so keep the call behind the identical declaration guard.
        sfpi::slope_max_eval();
#endif
#else
        sfpi::slope_max_eval();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ZONE_FACTORED_EXP_ROOT_CORRECTION_2 || \
    defined(EVAL_METHOD_ZONE_FACTORED_EXP_ROOT_CORRECTION_2)
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y =
                sfpi::zone_factored_exp_root_correction_2_eval(x);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#if defined(TT_EXP_ROOT_LATE_RAW_NEG_NAN_POS_INF)
            // This typed terminal plan is deliberately late: ordinary SFPU
            // ingress aliases both raw NaN signs with +Inf, while the TTNN
            // contract distinguishes the 127 negative payloads.  Inspect the
            // untouched encoded source after the exp/root body has retired.
            sfpi::vUInt raw_u16 =
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt exponent_delta =
                (raw_u16 ^ sfpi::vUInt(0x807fu)) & sfpi::vUInt(0x807fu);
            v_if(exponent_delta == 0u) {
                sfpi::vUInt mantissa = raw_u16 & sfpi::vUInt(0x7f00u);
                sfpi::vUInt sign = raw_u16 & sfpi::vUInt(0x0080u);
                v_if((mantissa != 0u) && (sign != 0u)) {
                    y = std::numeric_limits<float>::infinity();
                }
                v_endif;
            }
            v_endif;
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR)
            // Reload after the multileaf arithmetic has retired: retaining an
            // encoded classifier across exp/root/Horner exceeds the common
            // eight-LREG budget.  DST still owns the untouched source word.
            sfpi::vUInt raw_u16 =
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
            sfpi::finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ZONE_FACTORED_AFFINE_EVEN_DECAY_2 || \
    defined(EVAL_METHOD_ZONE_FACTORED_AFFINE_EVEN_DECAY_2)
        (void)p_lut;
#pragma GCC unroll 8
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::zone_factored_affine_even_decay_2_eval(x);
#if defined(TT_ZONE_ORIGIN_HALF_TARGET_REPAIR)
            sfpi::apply_zone_factored_origin_half_target_repair(
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>(), x, y);
#endif
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_RESIDUAL_AFFINE_1 || \
    defined(EVAL_METHOD_ABS_RESIDUAL_AFFINE_1)
        (void)p_lut;
#if ABS_RESIDUAL_TTI_CANDIDATE
        if constexpr (sfpi::kAbsResidualTtiReplay) {
            sfpi::abs_residual_tti_replay_tile();
        } else
#endif
        {
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::abs_residual_affine_1_eval<
                ABS_RESIDUAL_NUM_DEGREE,
                ABS_RESIDUAL_DEN_DEGREE>(x);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER)
            sfpi::finalize_encoded_raw_domain_actions(
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>(), y);
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_SQUARE_DECAY_1 || \
    defined(EVAL_METHOD_ABS_SQUARE_DECAY_1)
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::abs_square_decay_1_eval<
                SQUARE_DECAY_NUM_DEGREE,
                SQUARE_DECAY_DEN_DEGREE>(x);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT)
            // DST still owns the untouched source word.  Reload the encoded
            // class after the numeric body so the classifier consumes no
            // long-lived SFPU register.
            sfpi::finalize_encoded_raw_domain_actions(
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>(), y);
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_EXPONENT_BUCKET_LOG_DERIVATIVE_1 || \
    defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1)
        (void)p_lut;
        static_assert(
            sfpi::TT_EXPONENT_BUCKET_PEAK_LIVE <= 8u,
            "exponent-bucket log derivative exceeds SFPU register capacity");
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::exponent_bucket_log_derivative_1_eval(
                x, d + 32);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
#if defined(TT_TARGET_SELECTED_GUARDED_CLASS_TERMINAL)
            sfpi::apply_target_selected_guarded_class_terminal(x_raw, y);
#endif
#if defined(TT_TARGET_SELECTED_TTI_CLASS_REPAIR)
            // The selected core's row d+32 handoff is dead after this call.
            // Reuse it for selected BF16 egress while rows d retain raw input.
            sfpi::dst_reg[d + 32] = y;
#else
            sfpi::dst_reg[d] = y;
#endif
        }
#if defined(TT_TARGET_SELECTED_TTI_CLASS_REPAIR)
        sfpi::exponent_bucket_target_class_repair_tti_tile();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ROOT_NATIVE_LOG_RECIPROCAL_1 || \
    defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1)
        (void)p_lut;
        static_assert(
            sfpi::TT_ROOT_NATIVE_LOG_COMMON_SFPI_CLOSES == 1u &&
                sfpi::TT_ROOT_NATIVE_LOG_COMPILER_EXACT_REQUIRED == 1u &&
                sfpi::TT_ROOT_NATIVE_LOG_PEAK_LIVE <=
                    sfpi::TT_ROOT_NATIVE_LOG_LREG_CAPACITY,
            "root/native-log form exceeds certified resources");
#if defined(TT_ROOT_NATIVE_LOG_RAW_SHADOW_SCHEDULE)
        static_assert(
            sfpi::TT_ROOT_RAW_SHADOW_ROWS == 32u &&
                sfpi::TT_ROOT_RAW_SHADOW_ROW_BASE >= 64u &&
                sfpi::TT_ROOT_RAW_SHADOW_ROW_BASE +
                        sfpi::TT_ROOT_RAW_SHADOW_ROWS <=
                    sfpi::TT_ROOT_RAW_SHADOW_DST_CAPACITY,
            "root/native-log raw shadow overlaps its FP32 handoff or DST capacity");
        static_assert(
            sfpi::TT_ROOT_RAW_SHADOW_PRESERVE_PEAK_LIVE <= 8u &&
                sfpi::TT_ROOT_RAW_SHADOW_SUFFIX_PEAK_LIVE <= 8u,
            "root/native-log raw terminal exceeds SFPU liveness capacity");
#endif
        for (int d = 0; d < 32; d++) {
#if defined(TT_ROOT_NATIVE_LOG_RAW_SHADOW_SCHEDULE)
            {
                sfpi::vUInt encoded_raw =
                    sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
                sfpi::dst_reg[sfpi::TT_ROOT_RAW_SHADOW_ROW_BASE + d]
                    .mode<sfpi::DataLayout::U16>() = encoded_raw;
            }
#endif
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y =
                sfpi::root_native_log_reciprocal_1_eval<
                    sfpi::TT_ROOT_NATIVE_LOG_DEGREE>(
                    x,
#if defined(TT_ROOT_NATIVE_LOG_TOTAL_DOMAIN)
                    d + 32
#else
                    d
#endif
                );
            sfpi::finalize_raw_domain_actions(x_raw, y);
#if defined(TT_ROOT_NATIVE_LOG_RAW_SHADOW_SCHEDULE)
            sfpi::vUInt terminal_raw =
                sfpi::dst_reg[sfpi::TT_ROOT_RAW_SHADOW_ROW_BASE + d]
                    .mode<sfpi::DataLayout::U16>();
            sfpi::finalize_encoded_raw_domain_actions(terminal_raw, y);
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1 || \
    defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1)
        (void)p_lut;
        static_assert(
            TT_SHARED_INVERSE_SQUARE_BODY_SLOTS <= 32u &&
                TT_SHARED_INVERSE_SQUARE_CORE_PEAK_LIVE <= 8u &&
            SHARED_INVERSE_SQUARE_SUFFIX_PEAK_LIVE <= 8u,
            "shared inverse-square recurrence exceeds certified replay resources");
#if defined(ARCH_BLACKHOLE) && defined(USE_BF16)
        if constexpr (sfpi::kSharedInverseSquareTtiReplay) {
            sfpi::shared_inverse_square_recurrence_p0_p2_1_tti_replay_tile();
        } else
#endif
        {
        for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
            sfpi::vUInt raw_u16 =
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
#endif
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::shared_inverse_square_recurrence_p0_p2_1_eval(x);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
        }
#elif defined(BASIS_NORMALIZED_LOG_ODDS_SEPARABLE)
        sfpi::normalized_log_odds_separable_eval<lut_size>(*p_lut);
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ROOT_LOG1P_HANDOFF_1 || \
    defined(EVAL_METHOD_ROOT_LOG1P_HANDOFF_1)
        (void)p_lut;
        static_assert(LOG1P_HW_DEGREE == 1u);
        static_assert(sfpi::tti_replay_fits(sfpi::TT_ROOT_LOG1P_PRODUCER_SLOTS));
        static_assert(sfpi::tti_replay_fits(sfpi::TT_ROOT_LOG1P_CONSUMER_SLOTS));

        // Phase one: the form-declared fixed one-correction root producer.
        // It stores argument=(x-1)+sqrt((x-1)(x+1)) below the certified bound,
        // and x-1 above it, into the same-row FP32 tile-1 handoff.
        sfpi::vConstIntPrgm0 = NEWTON_ROOT_MAGIC;
        sfpi::vConstFloatPrgm1 = NEWTON_ROOT_C1;
        sfpi::vConstFloatPrgm2 = NEWTON_ROOT_C2;
        TT_REPLAY_RECORD(sfpi::TT_ROOT_LOG1P_PRODUCER_SLOTS);
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG2,
            0,
            TT_ADDRMOD_DST_HOLD,
            0);  // decoded FLOAT16_B ingress x
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LREG3,
            0);  // argument=x-1 [MAD]
        TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0x4d80);  // 2^28; MAD gap
        TTI_SFPGT(
            0,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG6,
            1);  // x < bound; preserve replay-pinned bound
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG4,
            0);  // x+1 [MAD]
        TTI_SFPNOP;
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG4,
            0);  // radicand [MAD]
        TTI_SFPSHFT(0xFFF, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, 5);
        TTI_SFPIADD(0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 6);
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG1,
            0);  // xy [MAD]
        TTI_SFPNOP;
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG1,
            1);  // c=-(inverse*xy) [MAD]
        TTI_SFPNOP;
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG14,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG5,
            0);  // C2+c [MAD]
        TTI_SFPNOP;
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG13,
            ckernel::p_sfpu::LREG1,
            0);  // c*(C2+c)+C1 [MAD]
        TTI_SFPNOP;
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG0,
            0);  // corrected inverse [MAD]
        TTI_SFPNOP;
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG3,
            0);  // root+argument [MAD]
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        TTI_SFPSTORE(
            ckernel::p_sfpu::LREG3,
            ckernel::InstrModLoadStore::FP32,
            TT_ADDRMOD_DST_ADV,
            sfpi::TT_ROOT_LOG1P_SCRATCH_OFFSET);
        TTI_SFPNOP;
        // The first argument is the recorded BODY length, not the number of
        // rows already produced.  Recording executes row zero, so replay the
        // producer for every remaining declared row.  Using `23` here
        // accidentally confused the 24-slot body length with the 32-row tile
        // and left scratch rows 24..31 unwritten for the consumer.
        TT_REPLAY_REST(
            sfpi::TT_ROOT_LOG1P_PRODUCER_SLOTS,
            sfpi::TT_ROOT_LOG1P_ROWS - 1u);
        TT_REPLAY_TILE_EPILOGUE();

        // Phase two: unchanged P1 Juffa LOG1P arithmetic. Prgm0/1/2 own
        // ln2*2^-23, c1, and exact ln2. c0 must be materialized inside every
        // record execution because P1 writes its accumulator; a one-time LREG
        // pin would feed each replay the preceding row's polynomial value.
        sfpi::vConstFloatPrgm0 =
            (float)(sfpi::TT_ROOT_LOG1P_LN2 * 0x1p-23f);
        sfpi::vConstFloatPrgm1 = LOG1P_HW_COEFFS[1];
        sfpi::vConstFloatPrgm2 = sfpi::TT_ROOT_LOG1P_LN2;
        constexpr uint32_t kRootLog1pC0Bits =
            __builtin_bit_cast(uint32_t, LOG1P_HW_COEFFS[0]);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0x4d80);  // 2^28
        TT_REPLAY_RECORD(sfpi::TT_ROOT_LOG1P_CONSUMER_SLOTS);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_UPPER,
            kRootLog1pC0Bits >> 16);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_LOWER,
            kRootLog1pC0Bits & 0xffffu);
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG2,
            ckernel::InstrModLoadStore::FP32,
            TT_ADDRMOD_DST_HOLD,
            sfpi::TT_ROOT_LOG1P_SCRATCH_OFFSET);  // argument
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG1,
            0);  // u=argument+1 [MAD]
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x3f40);  // 0.75; MAD gap
        TTI_SFPIADD(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 6);
        TTI_SFPSETMAN(0, 0, 0, 1);
        TTI_SFPMOV(0, 0, 1, 2);
        TTI_SFPIADD(0, 2, 1, 6);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0xc080);
        TTI_SFPMOV(0, 0, 2, 2);
        TTI_SFPIADD(0, 3, 2, 6);
        TTI_SFPMULI(0xbe80, ckernel::p_sfpu::LREG2, 0);  // *-0.25 [MAD]
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG4,
            0,
            TT_ADDRMOD_DST_HOLD,
            0);  // original decoded FLOAT16_B x; MAD gap
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LREG2,
            0);  // t [MAD]
        TTI_SFPEXEXP(
            0,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG3,
            sfpi::SFPEXEXP_MOD1_NODEBIAS);  // decoded biased exponent; MAD gap
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG1,
            0);  // r [MAD]
        TTI_SFPIADD(
            (-((int)sfpi::TT_ROOT_LOG1P_SPECIAL_BIASED_EXPONENT)) & 0xfff,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG3,
            sfpi::SFPIADD_MOD1_ARG_IMM |
                sfpi::SFPIADD_MOD1_CC_NONE);  // biased exponent - 255; MAD gap
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG13,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG7,
            0);  // P1 [MAD]
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG2,
            0);  // r*r [MAD]
        TTI_SFPNOP;  // decoded x remains live in L4; MAD gap
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG1,
            0);  // r+r*r*P [MAD]
        TTI_SFPABS(0, 0, 2, 0);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPSETSGN(0, 2, 0, 0);
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG12,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG0,
            0);  // k*ln2 + reduced result [MAD]
        TTI_SFPLE(
            0,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG6,
            1);  // bound <= x; preserve replay-pinned bound; MAD gap
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG14,
            ckernel::p_sfpu::LREG0,
            0);  // exact +ln2 [MAD]
        TTI_SFPENCC(3, 0, 0, 10);

        auto root_log1p_suffix = []() {
            // BH FLOAT16_B ingress has already collapsed every NaN payload to
            // signed infinity. First map decoded exponent-FF to +Inf, then let
            // x<1 restore qNaN for every negative special. The target-derived
            // certificate proves this observable class quotient exhaustively;
            // no raw DST payload is transported through the FP32 handoff.
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG3,
                0,
                sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x7f80);  // +Inf
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPGT(
                0,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LCONST_1,
                1);  // x < 1
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x7fc0);  // qNaN
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
                0);  // ordinary BF16 store; dest += one FP32 row
        };
        root_log1p_suffix();
#pragma GCC unroll 32
        for (int d = 1; d < (int)sfpi::TT_ROOT_LOG1P_ROWS; d++) {
            TTI_REPLAY(0, sfpi::TT_ROOT_LOG1P_CONSUMER_SLOTS, 0, 0);
            root_log1p_suffix();
        }
        TT_REPLAY_TILE_EPILOGUE();

#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_EXPONENT_ALU || defined(EVAL_METHOD_EXPONENT_ALU)
        (void)p_lut;
#if defined(HW_PRELOAD)
#if defined(EXPONENT_ALU_EXP2)
#if EXP_HW_BOUNDED_TWO_SIDED_TTI_CANDIDATE
        static_assert(
            sfpi::kBoundedTwoSidedTtiReplay,
            "bounded-exp two-sided form requires its certified replay");
        if constexpr (sfpi::kBoundedTwoSidedTtiReplay) {
            sfpi::bounded_exp_two_sided_tti_replay_tile();
        } else
#elif EXP_HW_TTI_CANDIDATE
        if constexpr (sfpi::kExpHwTtiReplayBare) {
            // TTI replay fast path (see the kExpHwTtiReplay comment block for
            // the full schedule). Per tile: 3 SFPLOADI pins + record the 13-op
            // per-element body once (executing element 0), then 31 single-issue
            // replays. The per-tile SETRWC issued by the copy_tile scaffolding
            // has already zeroed the D-RWC counter; ADDR_MOD_6 advances it +2
            // per SFPSTORE.
            // Two elements per body (A in L0/L1/L2, B in L3/L4/L6), interleaved
            // so every SFPMAD consumer sits >= 1 instruction after its producer.
            // At the inline-issue rate of the sfpi path the RISC-V frontend
            // provides that gap for free; in a replay stream the instructions
            // execute back-to-back and a gap-0 MAD consumer reads a STALE
            // register (measured: a handful of nondeterministic inf outputs per
            // 256 tiles from exexp of a stale xlog2). TTNN's exp_21f body obeys
            // the same discipline: every one of its MAD results is consumed
            // exactly one instruction later (SFPLOADI/SFPGT/SFPAND fillers).
            // 26 Tensix instructions per 2 elements = 13 per element (DEG 3);
            // 24 / 12 at DEG 2 (one fewer Horner MAD pair).
            // L7 carries the last register-resident Horner addend: c1f at
            // DEG 3 (final addend is then the hardware 1.0, gated c0 == 1.0);
            // at DEG 2 c1f lives in prgm2, freeing L7 for an ARBITRARY c0
            // (exp2's selected fit has c0 = 1.0017... != 1.0).
            constexpr float kExpTtiL7f = (EXP_HW_DEGREE == 3)
                                             ? EXP_HW_COEFFS[1] * sfpi::exp_hw_fused_scale(1)
                                             : EXP_HW_COEFFS[0];
            constexpr uint32_t kExpTtiL7Bits = __builtin_bit_cast(uint32_t, kExpTtiL7f);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);  // 127.0
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, kExpTtiL7Bits);
#if defined(TT_EXPONENT_ALU_EXP2_TERMINAL_CLAMP)
            static_assert(
                TT_EXPONENT_ALU_EXP2_REPLAY_PRELUDE_SLOTS == 2u,
                "EXP route replay prelude slot count changed");
            // SFPLOADI is not replayed by the hardware replay buffer. L2/L6
            // are free until their later Horner writes, so repin both to 255
            // before the record execution and before every replay issue.
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG2,
                sfpi::SFPLOADI_MOD0_FLOATB,
                0x437f);
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG6,
                sfpi::SFPLOADI_MOD0_FLOATB,
                0x437f);
#endif
            TT_REPLAY_RECORD(sfpi::kExpHwTtiBodySlots);  // record + execute elements 0,1
            TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // xA
            TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_HOLD, 2);  // xB
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG12,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG0,
                0);  // xlog2A = xA*mult + 127
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG12,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG3,
                0);  // xlog2B
#if defined(TT_EXPONENT_ALU_EXP2_TERMINAL_CLAMP)
            static_assert(
                TT_EXPONENT_ALU_EXP2_REPLAY_BODY_SLOTS == sfpi::kExpHwTtiBodySlots,
                "typed exp2 total-form replay slot count changed");
            static_assert(
                TT_EXPONENT_ALU_EXP2_TRANSFORMED_LOWER_CLAMP_BITS == 0x00000000u,
                "EXP terminal replay currently requires the exact +0 clamp");
            static_assert(
                TT_EXPONENT_ALU_EXP2_TRANSFORMED_UPPER_CLAMP_BITS == 0x437f0000u,
                "EXP terminal replay currently requires the exact 255 clamp");
            // The per-issue prelude pinned independent 255 vectors. Each swap
            // writes the clamped lane and may destroy its pin; Horner then
            // overwrites L2/L6, and the next prelude restores both.
            TTI_SFPSWAP(
                0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG0,
                9);  // xlog2A = max(+0, xlog2A)
            TTI_SFPSWAP(
                0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG3,
                9);  // xlog2B = max(+0, xlog2B)
            TTI_SFPSWAP(
                0,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG0,
                1);  // xlog2A = min(xlog2A, 255)
            TTI_SFPSWAP(
                0,
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG3,
                1);  // xlog2B = min(xlog2B, 255)
#endif
            TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 0);  // eA (debiased)
            TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 0);  // eB
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // mA (implicit one)
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 0);  // mB
            TTI_SFPSHFT(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);   // zA = mA << eA
            TTI_SFPSHFT(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 0);   // zB
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);  // fracA
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, sfpi::SFPEXMAN_MOD1_PAD9);  // fracB
            TTI_SFPCAST(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);  // fA
            TTI_SFPCAST(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 0);  // fB
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG2,
                0);  // pA = c3f*fA + c2f
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG6,
                0);  // pB
            // Second Horner rung: at DEG 3 the addend is c1f (L7) and the
            // result stays in the accumulator (L2/L6) for the final rung; at
            // DEG 2 this IS the final rung — addend c0 (L7), result straight
            // into L1/L4 where the SETEXP recombine below reads it.
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG7,
                (EXP_HW_DEGREE == 3) ? ckernel::p_sfpu::LREG2 : ckernel::p_sfpu::LREG1,
                0);  // pA*fA + (DEG3: c1f | DEG2: c0)
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG7,
                (EXP_HW_DEGREE == 3) ? ckernel::p_sfpu::LREG6 : ckernel::p_sfpu::LREG4,
                0);  // pB*fB + (DEG3: c1f | DEG2: c0)
            if constexpr (EXP_HW_DEGREE == 3) {
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG2,
                    ckernel::p_sfpu::LREG1,
                    ckernel::p_sfpu::LCONST_1,
                    ckernel::p_sfpu::LREG1,
                    0);  // pA*fA + 1.0 (hw const, lreg 10; gate proved c0 == 1.0)
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG6,
                    ckernel::p_sfpu::LREG4,
                    ckernel::p_sfpu::LCONST_1,
                    ckernel::p_sfpu::LREG4,
                    0);  // pB*fB + 1.0
            }
            TTI_SFPSETEXP(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 2);  // yA bare recombine
            TTI_SFPSETEXP(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 2);  // yB
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // yA fp32 -> bf16 RNE
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG3,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // yB
            TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // yA (counter unchanged)
            TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_ADV, 2);  // yB, dest += 4
#if defined(TT_EXPONENT_ALU_EXP2_TERMINAL_CLAMP)
            _Pragma("GCC unroll 32") for (int _r = 0; _r < 15; _r++) {
                TTI_SFPLOADI(
                    ckernel::p_sfpu::LREG2,
                    sfpi::SFPLOADI_MOD0_FLOATB,
                    0x437f);
                TTI_SFPLOADI(
                    ckernel::p_sfpu::LREG6,
                    sfpi::SFPLOADI_MOD0_FLOATB,
                    0x437f);
                TTI_REPLAY(0, sfpi::kExpHwTtiBodySlots, 0, 0);
            }
#else
            TT_REPLAY_REST(sfpi::kExpHwTtiBodySlots, 15);  // pairs 1..15
#endif
            // The ADDR_MOD_6 walk leaves the D-RWC at 64. The NEXT tile's A2D
            // datacopy MOP shares this counter and assumes it is 0 at tile
            // start (the sfpi path's absolute-offset loads never moved it, so
            // the assumption held silently). Not resetting it here shifts every
            // subsequent tile's input copy by 64 dst rows — measured as whole
            // tiles returning garbage in a double-buffer-periodic pattern.
            TT_REPLAY_TILE_EPILOGUE();
        } else if constexpr (sfpi::kExpHwTtiReplayPe) {
            // PE-CORRECTED (non-bare, c0 < 1.0) TTI replay fast path — see
            // the kExpHwTtiReplay comment block. Per tile: 3-5 SFPLOADI pins
            // (127.0 half; c0 — and c1f at DEG 3 — fp32 halves), record the
            // single-element body once (executing element 0), 31 replays,
            // SETRWC SET_D.
            //
            // Body = 1:1 transcription of the sfpi/GCC BH non-bare loop
            // (exp_hw_eval_preloaded, FUSED + SKIP_CLAMP, no compose): same
            // MAD order and the NoDebias 2^i recombine `p * (ep << 23)`. The
            // EXEXP(NoDebias) + SHFT-imm-23 + MUL encodings are lifted from
            // the silicon-proven kSigmoidTtiReplayD3 body (minus its +1.0
            // compose fold and reciprocal). Registers: L0 work
            // (x/xlog2/m/z/frac/f/y), L1 e then ep then 2^i, L2 Horner acc,
            // L5 = 127.0 pin, L7 = c1f pin at DEG 3 / c0 pin at DEG 2
            // (mirrors the bare DEG-2 convention), L4 = c0 pin (DEG 3 only),
            // L12/13/14 = MULT/ctopf/csecf (prgm0/1/2), L9 = hw 0.0. Gap
            // discipline: every MAD-family producer gets >= 1 slot before
            // its consumer (SFPNOP or an independent bit-unit op); bit-unit
            // results are gap-0 safe.
            constexpr float kExpTtiPeL7f = (EXP_HW_DEGREE == 3)
                                               ? EXP_HW_COEFFS[1] * sfpi::exp_hw_fused_scale(1)
                                               : EXP_HW_COEFFS[0];
            constexpr uint32_t kExpTtiPeL7Bits = __builtin_bit_cast(uint32_t, kExpTtiPeL7f);
            constexpr uint32_t kExpTtiPeC0Bits = __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);  // 127.0
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, kExpTtiPeL7Bits);
            if constexpr (EXP_HW_DEGREE == 3) {
                TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, kExpTtiPeC0Bits);
            }
            TT_REPLAY_RECORD(sfpi::kExpHwTtiPeBodySlots);  // record + execute element 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG12,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG0,
                0);  // xlog2 = x*mult + 127                                  [MAD]
            TTI_SFPNOP;
            TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 0);  // e (debiased)
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // m (implicit one)
            TTI_SFPSHFT(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);   // z = m << e
            TTI_SFPEXEXP(
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG1,
                sfpi::SFPEXEXP_MOD1_NODEBIAS);  // ep (biased, from z)
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9);  // frac
            TTI_SFPCAST(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // f (fp32)
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG2,
                0);  // p = ctopf*f + csecf                                   [MAD]
            TTI_SFPNOP;
            if constexpr (EXP_HW_DEGREE == 3) {
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG2,
                    ckernel::p_sfpu::LREG0,
                    ckernel::p_sfpu::LREG7,
                    ckernel::p_sfpu::LREG2,
                    0);  // p = p*f + c1f                                     [MAD]
                TTI_SFPNOP;
            }
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG0,
                (EXP_HW_DEGREE == 3) ? ckernel::p_sfpu::LREG4 : ckernel::p_sfpu::LREG7,
                ckernel::p_sfpu::LREG2,
                0);  // p = p*f + c0                                          [MAD]
            TTI_SFPSHFT(0x017, 1, 1, 7);  // 2^i bits = ep << 23 (imm; final-rung gap filler)
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG0,
                0);  // y = p * 2^i (sfpi twin: p * as<vFloat>(ep << 23))     [MAD]
            TTI_SFPNOP;
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // y fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);  // store, dest += 2
            TT_REPLAY_REST(sfpi::kExpHwTtiPeBodySlots, 31);  // elements 1..31
            // Re-zero the shared D-RWC counter (see the bare body above).
            TT_REPLAY_TILE_EPILOGUE();
        } else
#endif  // EXP_HW_TTI_CANDIDATE
#if EXP_HW_COMPOSE_TTI_CANDIDATE
#if defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
        if constexpr (sfpi::kSymmetricSigmoidProductTtiReplayD2Bare) {
            // Anonymous whole-core sigmoid product. The ordinary frontier P2
            // exp2 leaf is evaluated on |x| and remains doubled through an
            // output-lattice transport certified from structural metadata.
            constexpr uint32_t kSymMultBits =
                __builtin_bit_cast(uint32_t, (float)EXP_HW_MULT);
            constexpr uint32_t kSymC0Bits =
                __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
            constexpr int kSymRecordedBodyLenD2 = 23;
            static_assert(sfpi::tti_replay_fits(kSymRecordedBodyLenD2));
            static_assert(
                TT_SYMMETRIC_SIGMOID_PRODUCT_BODY_SLOTS == kSymRecordedBodyLenD2 &&
                    TT_SYMMETRIC_SIGMOID_PRODUCT_SUFFIX_SLOTS == 11 &&
                    TT_SYMMETRIC_SIGMOID_PRODUCT_PEAK_LREGS == 8 &&
                    TT_SYMMETRIC_SIGMOID_PRODUCT_TTNN_PARITY == 1,
                "certified symmetric sigmoid-product resources changed");
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG3, kSymMultBits);
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, kSymC0Bits);
            TT_REPLAY_RECORD(kSymRecordedBodyLenD2);
            TTI_SFPLOAD(
                ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG5,
                sfpi::SFPLOADI_MOD0_FLOATB,
                EXP_HW_EXPONENT_BIAS_BF16);  // exact bias; load hazard filler
            TTI_SFPSETSGN(
                0,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG2,
                1);  // |x|; preserved through the replay suffix
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG7,
                0);  // fused biased xlog2
            TTI_SFPNOP;
            TTI_SFPEXEXP(
                0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, 0);
            TTI_SFPEXMAN(
                0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 0);
            TTI_SFPSHFT(0, 5, 0, 0);
            TTI_SFPEXMAN(
                0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 1);
            TTI_SFPCAST(
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG5,
                0);
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG6,
                0);
            TTI_SFPGT(
                0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG7,
                8);
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG5,
                0);
            TTI_SFPAND(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG7,
                ckernel::p_sfpu::LREG0,
                1);
            TTI_SFPSETEXP(
                0,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG0,
                2);  // biased exp polynomial y
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG7,
                0);  // x+|x|; L1 keeps raw x through the suffix
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG6,
                0);  // |x|*y
            TTI_SFPMULI(0x3780, ckernel::p_sfpu::LREG0, 0);  // y*2^-16
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG0,
                0);  // 1+y*2^-16
            TTI_SFPARECIP(0, 0, 5, 0);
            TTI_SFPMULI(0x3800, ckernel::p_sfpu::LREG6, 0);  // |x|*y*2^-15
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG12,
                ckernel::p_sfpu::LREG0,
                2);
            TTI_SFPNOP;
#pragma GCC unroll 32
            for (int d = 0; d < 32; d++) {
                if (d != 0) {
                    TTI_REPLAY(0, kSymRecordedBodyLenD2, 0, 0);
                }
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG0,
                    ckernel::p_sfpu::LREG5,
                    ckernel::p_sfpu::LCONST_0,
                    ckernel::p_sfpu::LREG5,
                    3);
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG6,
                    ckernel::p_sfpu::LREG5,
                    ckernel::p_sfpu::LREG7,
                    ckernel::p_sfpu::LREG0,
                    1);  // doubled signed result
                // Fill the result-MAD -> exact-half dependency window
                // with the independent typed origin-half discriminator.
                // Scaling |x| by 2^126 makes the top first-binade magnitude
                // ordinary 1.9921875, so the test survives FP32 FTZ arithmetic.
                TTI_SFPDIVP2(
                    0x07E,
                    ckernel::p_sfpu::LREG2,
                    ckernel::p_sfpu::LREG5,
                    1);
                // Clamp the doubled result to raw x before halving.  Besides
                // the ordinary lower bound, this owns the positive 0x0100
                // origin word that cannot survive a direct FP32 half under FTZ.
                TTI_SFPSWAP(
                    0,
                    ckernel::p_sfpu::LREG0,
                    ckernel::p_sfpu::LREG1,
                    sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
                TTI_SFPMULI(0x3F00, ckernel::p_sfpu::LREG0, 0);  // exact half
                TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG5, 0);
                TTI_SFPSETCC(
                    0,
                    ckernel::p_sfpu::LREG5,
                    0,
                    sfpi::SFPSETCC_MOD1_LREG_EQ0);
                TTI_SFPSETMAN(
                    0,
                    ckernel::p_sfpu::LREG1,
                    ckernel::p_sfpu::LREG0,
                    1);  // raw-signed MIN_NORMAL for the exact half tie
                TTI_SFPENCC(0, 0, 0, 0);
                TTI_SFPSWAP(
                    0,
                    ckernel::p_sfpu::LREG2,
                    ckernel::p_sfpu::LREG0,
                    1);  // positive identity cap; negative values pass through
                TTI_SFPLOADMACRO(2, 0, TT_ADDRMOD_DST_ADV, 0);
            }
            // The final macro's fixed store fires at issue+3.  Other elements
            // drain under the next replay head; only the last needs padding.
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
            TT_REPLAY_TILE_EPILOGUE();
        } else if constexpr (sfpi::kSigmoidTtiReplayD3) {
#else
        // The resource certificate macros above exist only for the symmetric
        // product form.  Ordinary sigmoid/silu candidates must start at their
        // own replay branch instead of parsing an inactive symmetric body.
        if constexpr (sfpi::kSigmoidTtiReplayD3) {
#endif
            // DEG-3 NON-bare sigmoid / silu compose TTI replay fast path
            // (shape (b) in the kSigmoidTtiReplay comment block). Per tile:
            // 6 SFPLOADI pins (MULT, c1f, c0 — fp32 halves), record the
            // single-element body once (executing element 0), 31 replays,
            // SETRWC SET_D.
            //
            // Body = 1:1 transcription of the sfpi/GCC BH silu_p3 loop
            // (objdump: deg-3 FUSED Horner + NoDebias-exponent 2^i synthesis
            // + GCC-fused `u = p*2^i + 1.0` MAD + ARECIP one-Newton
            // reciprocal), with explicit gap slots after every MAD-family
            // producer. Registers mirror the objdump: L1 = x (live to the
            // end for the silu product), L3 = MULT pin, L2 = c1f pin,
            // L4 = c0 pin, L0 work (xlog2/m/z/frac/p/u/t), L5 e/ep/2^i/r0,
            // L6 Horner acc, L12 = 2.0 (reciprocal, prgm0), L13/L14 =
            // c3f/c2f (prgm1/2), L10 = hw 1.0, L9 = hw 0.0.
            constexpr float kSigMultF = EXP_HW_MULT;
            constexpr uint32_t kSigMultBits = __builtin_bit_cast(uint32_t, kSigMultF);
            constexpr uint32_t kSigC0Bits = __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
            // c1 pre-scaled by 2^-23 (FUSED Horner over the raw exman
            // fraction) — the exact constant the sfpi twin pins (exp_pin_a =
            // EXP_HW_COEFFS[DEG-2] * exp_hw_fused_scale(DEG-2), DEG==3).
            constexpr uint32_t kSigC1fBits =
                __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[1] * sfpi::exp_hw_fused_scale(1));
            constexpr int kSigBodyLenD3 =
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
                30;
#else
                28;
#endif
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kSigMultBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kSigMultBits & 0xFFFFu));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, (kSigC1fBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, (kSigC1fBits & 0xFFFFu));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, (kSigC0Bits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, (kSigC0Bits & 0xFFFFu));
            TT_REPLAY_RECORD(kSigBodyLenD3);  // record + execute element 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 0);            // x
            TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0); // xlog2 = x*MULT   [MAD]
            TTI_SFPNOP;
            TTI_SFPADDI(0x42FE, ckernel::p_sfpu::LREG0, 0);                   // xlog2 += 127     [MAD]
            TTI_SFPNOP;
            TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);  // e (debiased)
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // m (implicit 1)
            TTI_SFPSHFT(0, 5, 0, 0);                                          // z = m << e (L0)
            TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5,
                         sfpi::SFPEXEXP_MOD1_NODEBIAS);                       // ep (biased, from z)
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);  // frac (PAD9) over z
            TTI_SFPCAST(0, 0, 0);                                             // f (fp32)
            TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG6, 0);   // p = c3f*f + c2f  [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG6, 0);    // p = p*f + c1f    [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, 0);    // p = p*f + c0     [MAD]
            TTI_SFPSHFT(0x017, 5, 5, 7);                                      // 2^i bits = ep << 23 (imm; p-gap filler)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5,
                       ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0, 0); // u = p*2^i + 1.0  [MAD, GCC-fused]
            TTI_SFPNOP;
            TTI_SFPARECIP(0, 0, 5, 0);                                        // r0 = arecip(u) -> L5
            TTI_SFPNOP;                                                       // (ARECIP window)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5,
                       ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 2);   // t = u*r0 - 2.0   [MAD]
            TTI_SFPNOP;
            TTI_SFPSETCC(0, 0, 0, 0);                                         // CC: t < 0
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5, 3); // y = r0*-t - 0 (pred) [MAD]
            TTI_SFPENCC(3, 0, 0, 10);                                         // all lanes on
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
            TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0); // silu: x * y      [MAD]
            TTI_SFPNOP;
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0,
                              ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1,
                              sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);          // fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_ADV, 0);           // store, dest += 2
#else
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0,
                              ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5,
                              sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);          // fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, TT_ADDRMOD_DST_ADV, 0);           // store, dest += 2
#endif
#pragma GCC unroll 32
            for (int d = 1; d < 32; d++) {
                TTI_REPLAY(0, kSigBodyLenD3, 0, 0);
            }
            // Re-zero the shared D-RWC counter (see kExpHwTtiReplay).
            TT_REPLAY_TILE_EPILOGUE();
        } else if constexpr (sfpi::kSigmoidTtiReplay) {
            // Sigmoid / silu compose TTI replay fast path (see the
            // kSigmoidTtiReplay comment block). Per tile: 4 SFPLOADI pins
            // (MULT and c0, fp32 halves), record the single-element body once
            // (executing element 0), 31 replays, SETRWC SET_D.
            //
            // Body = 1:1 transcription of the sfpi/GCC BH loop (deg-2 FUSED
            // Horner + BARE recombine + ARECIP one-Newton reciprocal), with
            // the exp-proven SETEXP(ARG_EXPONENT) substitution and explicit
            // gap slots after every MAD-family producer. Registers: L1 = x
            // (live to the end for the silu product), L3 = MULT pin, L2 = c0
            // pin, L0/L4/L5 scratch, L12 = 2.0 (reciprocal, prgm0),
            // L13/L14 = c2f/c1f (prgm1/2).
            constexpr float kSigMultF = EXP_HW_MULT;
            constexpr uint32_t kSigMultBits = __builtin_bit_cast(uint32_t, kSigMultF);
            constexpr uint32_t kSigC0Bits = __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
            constexpr int kSigBodyLen =
#if defined(TT_EXPONENT_ALU_EXP2_RECIPROCAL_TERMINAL_CLAMP)
                28;
#elif defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
                28;
#else
                26;
#endif
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kSigMultBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kSigMultBits & 0xFFFFu));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, (kSigC0Bits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, (kSigC0Bits & 0xFFFFu));
#if defined(TT_EXPONENT_ALU_EXP2_RECIPROCAL_TERMINAL_CLAMP)
            static_assert(
                TT_EXPONENT_ALU_EXP2_REPLAY_BODY_SLOTS == 28u,
                "typed reciprocal replay slot count changed");
            static_assert(
                TT_EXPONENT_ALU_EXP2_REPLAY_PRELUDE_SLOTS == 0u,
                "typed reciprocal replay prelude changed");
            static_assert(
                TT_EXPONENT_ALU_EXP2_FINITE_RAW_PATTERN_COUNT == 65280u,
                "reciprocal terminal clamp is finite-only authority");
            static_assert(
                TT_EXPONENT_ALU_EXP2_TRANSFORMED_LOWER_CLAMP_BITS == 0x00000000u &&
                    TT_EXPONENT_ALU_EXP2_TRANSFORMED_UPPER_CLAMP_BITS == 0x437f0000u,
                "reciprocal terminal replay requires the exact closed [0,255] clamp");
            // L6 is dead throughout the selected D2 reciprocal body.  Its
            // destructive min-swap pin is loaded inside the recorded body by
            // filling an existing producer-to-ADDI hazard gap below.
#endif
            TT_REPLAY_RECORD(kSigBodyLen);  // record + execute element 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 0);            // x
            TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0); // xlog2 = x*MULT   [MAD]
#if defined(TT_EXPONENT_ALU_EXP2_RECIPROCAL_TERMINAL_CLAMP)
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG6,
                sfpi::SFPLOADI_MOD0_FLOATB,
                0x437f);  // independent load fills the x*MULT -> ADDI gap
#else
            TTI_SFPNOP;
#endif
            TTI_SFPADDI(0x42FE, ckernel::p_sfpu::LREG0, 0);                   // xlog2 += 127     [MAD]
            TTI_SFPNOP;
#if defined(TT_EXPONENT_ALU_EXP2_RECIPROCAL_TERMINAL_CLAMP)
            TTI_SFPSWAP(
                0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG0,
                9);  // xlog2 = max(+0, xlog2)
            TTI_SFPSWAP(
                0,
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG0,
                1);  // xlog2 = min(xlog2, 255)
#endif
            TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 0);  // e (debiased)
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // m (implicit 1)
            TTI_SFPSHFT(0, 4, 0, 0);                                          // z = m << e (L0)
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 1);  // frac (PAD9) -> L5
            TTI_SFPCAST(5, 5, 0);                                             // f (fp32)
            TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG5,
                       ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG4, 0);   // p = c2f*f + c1f  [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG5,
                       ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG4, 0);    // p = p*f + c0     [MAD]
            TTI_SFPNOP;
            TTI_SFPSETEXP(0, 4, 0, 2);                                        // y = 2^i*2^f (ARG_EXPONENT, z in L0)
            TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0, 0); // u = y + 1.0      [MAD]
            TTI_SFPNOP;
            TTI_SFPARECIP(0, 0, 4, 0);                                        // r0 = arecip(u) -> L4
            TTI_SFPNOP;                                                       // (ARECIP window)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4,
                       ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 2);   // t = u*r0 - 2.0   [MAD]
            TTI_SFPNOP;
            TTI_SFPSETCC(0, 0, 0, 0);                                         // CC: t < 0
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4, 3); // y = r0*-t - 0 (pred) [MAD]
            TTI_SFPENCC(3, 0, 0, 10);                                         // all lanes on
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
            TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG4,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0); // silu: x * y      [MAD]
            TTI_SFPNOP;
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0,
                              ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1,
                              sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);          // fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_ADV, 0);           // store, dest += 2
#else
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0,
                              ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4,
                              sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);          // fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, TT_ADDRMOD_DST_ADV, 0);           // store, dest += 2
#endif
#pragma GCC unroll 32
            for (int d = 1; d < 32; d++) {
                TTI_REPLAY(0, kSigBodyLen, 0, 0);
            }
            // Re-zero the shared D-RWC counter (see kExpHwTtiReplay).
            TT_REPLAY_TILE_EPILOGUE();
        } else
#endif  // EXP_HW_COMPOSE_TTI_CANDIDATE
#if EXP_HW_HYPERBOLIC_TTI_CANDIDATE
#if defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR)
        if constexpr (sfpi::kHyperbolicOddFactorTtiReplay) {
            sfpi::exp_hw_hyperbolic_odd_tti_tile();
        } else
#endif
        if constexpr (sfpi::kHyperbolicEvenTtiReplay) {
            constexpr uint32_t kMultBits =
                __builtin_bit_cast(uint32_t, (float)EXP_HW_MULT);
            constexpr uint32_t kC2Bits = __builtin_bit_cast(
                uint32_t, EXP_HW_COEFFS[2] * sfpi::exp_hw_fused_scale(2));
            constexpr uint32_t kC1Bits = __builtin_bit_cast(
                uint32_t, EXP_HW_COEFFS[1] * sfpi::exp_hw_fused_scale(1));
            constexpr uint32_t kC0Bits =
                __builtin_bit_cast(uint32_t, EXP_HW_COEFFS[0]);
            constexpr uint32_t kComposeScaleBits =
                __builtin_bit_cast(uint32_t, (float)EXP_HW_COMPOSE_SCALE);
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG1, kMultBits);
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG2, kC2Bits);
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG3, kC1Bits);
            TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, kC0Bits);
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG7,
                sfpi::SFPLOADI_MOD0_FLOATB,
                0x42fc);  // exact 126.0 exponent bias

            // LREG11 is a programmable architectural constant outside the
            // eight general LREGs. Program the descriptor-owned fp32 scale
            // uniformly, then restore the architectural -1.0 default below.
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_UPPER,
                (kComposeScaleBits >> 16));
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_LOWER,
                (kComposeScaleBits & 0xffffu));
            TTI_SFPCONFIG(0x5555, 11, 8);

            TT_REPLAY_RECORD(sfpi::kHyperbolicEvenTtiBodySlots);
            TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);
            TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG7,
                ckernel::p_sfpu::LREG0,
                0);
            // L6 is dead until the Horner head. Materialize the generic
            // exponent-ALU upper bound in the MAD hazard window, then consume
            // it once. This closes every overflow/special ingress lane without
            // a branch, raw shadow, or extra replay beyond the 32-slot limit.
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG6,
                sfpi::SFPLOADI_MOD0_FLOATB,
                0x437f);  // exact 255.0
            TTI_SFPSWAP(
                0,
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG0,
                1);  // xlog2 = min(xlog2, 255)
            TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);
            TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
            TTI_SFPSHFT(0, 5, 0, 0);
            TTI_SFPEXEXP(
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG5,
                sfpi::SFPEXEXP_MOD1_NODEBIAS);
            TTI_SFPEXMAN(
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                sfpi::SFPEXMAN_MOD1_PAD9);
            TTI_SFPCAST(0, 0, 0);
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG6,
                0);
            TTI_SFPNOP;
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG6,
                0);
            TTI_SFPNOP;
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG6,
                0);
            TTI_SFPNOP;
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG0,
                0);
            TTI_SFPSHFT(0x017, 5, 5, 7);
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG0,
                0);
            TTI_SFPNOP;
            TTI_SFPARECIP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);
            TTI_SFPNOP;
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG12,
                ckernel::p_sfpu::LREG6,
                2);
            TTI_SFPNOP;
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG6, 0, 0);
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG6,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG5,
                3);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LCONST_neg1,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                0);
            TTI_SFPNOP;
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);
            TT_REPLAY_REST(sfpi::kHyperbolicEvenTtiBodySlots, 31);
            TT_REPLAY_TILE_EPILOGUE();

            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xbf80);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
            TTI_SFPCONFIG(0x5555, 11, 8);
        } else
#endif  // EXP_HW_HYPERBOLIC_TTI_CANDIDATE
        {
        sfpi::vFloat thr_hoist = 255.0f;
        // Bias + first-three below-prgm Horner coefficients: LAUNDERED through
        // hw_preload_pin so GCC cannot constant-fold them back into per-element
        // SFPADDI/SFPLOADI immediates inside the 32-element dst loop (that
        // constant re-materialization is exactly what the disassembly showed:
        // sfpmul+sfpaddi instead of one SFPMAD, plus 2 SFPLOADI per coeff per
        // element). After the pin they are opaque LREG-resident values; each
        // Horner step and the bias-add compile to a single SFPMAD.
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
        // Sigmoid compose keeps the reciprocal's working set live on top of the
        // Horner chain: pinning 3-4 more LREGs here overflows the register file
        // (measured: GCC ICE "cannot store sfpu register (register spill)" on
        // sigmoid_accurate p5). Compose paths keep the baseline foldable-local
        // schedule; the LREG-pin fast path below is for the plain exp kernels.
        sfpi::vFloat c127_hoist = (float)EXP_HW_EXPONENT_BIAS;
        sfpi::vFloat mult_hoist = EXP_HW_MULT;
#else
        sfpi::vFloat c127_hoist =
            sfpi::hw_preload_pin<sfpi::LRegs::LReg7>((float)EXP_HW_EXPONENT_BIAS);
#endif
#if defined(EXP_HW_FUSED)
        constexpr float exp_pin_a =
            (EXP_HW_DEGREE >= 2) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 2] * sfpi::exp_hw_fused_scale((int)EXP_HW_DEGREE - 2)
                                 : 0.0f;
        constexpr float exp_pin_b =
            (EXP_HW_DEGREE >= 3) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 3] * sfpi::exp_hw_fused_scale((int)EXP_HW_DEGREE - 3)
                                 : 0.0f;
        constexpr float exp_pin_c =
            (EXP_HW_DEGREE >= 4) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 4] * sfpi::exp_hw_fused_scale((int)EXP_HW_DEGREE - 4)
                                 : 0.0f;
#else
        constexpr float exp_pin_a = (EXP_HW_DEGREE >= 2) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 2] : 0.0f;
        constexpr float exp_pin_b = (EXP_HW_DEGREE >= 3) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 3] : 0.0f;
        constexpr float exp_pin_c = (EXP_HW_DEGREE >= 4) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 4] : 0.0f;
#endif
        sfpi::vFloat exp_cva = 0.0f;
        sfpi::vFloat exp_cvb = 0.0f;
        sfpi::vFloat exp_cvc = 0.0f;
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
        // No pin under sigmoid compose (register file pressure, see above):
        // plain locals -> GCC folds them back to in-body materialization,
        // which is the measured baseline schedule for compose kernels.
        exp_cva = exp_pin_a;
        exp_cvb = exp_pin_b;
        exp_cvc = exp_pin_c;
#else
        if constexpr (EXP_HW_DEGREE >= 2) {
            exp_cva = sfpi::hw_preload_pin<sfpi::LRegs::LReg6>(exp_pin_a);
        }
        if constexpr (EXP_HW_DEGREE >= 3) {
            exp_cvb = sfpi::hw_preload_pin<sfpi::LRegs::LReg5>(exp_pin_b);
        }
        if constexpr (EXP_HW_DEGREE >= 4) {
            exp_cvc = sfpi::hw_preload_pin<sfpi::LRegs::LReg4>(exp_pin_c);
        }
#endif
        float exp_cvspill[(EXP_HW_DEGREE >= 2) ? (EXP_HW_DEGREE - 1) : 1];
        if constexpr (EXP_HW_DEGREE >= 2) {
#pragma GCC unroll 16
            for (int k = 0; k < (int)EXP_HW_DEGREE - 1; k++) {
                const int power = (int)EXP_HW_DEGREE - 2 - k;
#if defined(EXP_HW_FUSED)
                exp_cvspill[k] = EXP_HW_COEFFS[power] * sfpi::exp_hw_fused_scale(power);
#else
                exp_cvspill[k] = EXP_HW_COEFFS[power];
#endif
            }
        }
        for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
            sfpi::vUInt raw_u16 =
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
#endif
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::exp_hw_eval_preloaded<EXP_HW_DEGREE>(
                x,
                thr_hoist,
                c127_hoist,
                exp_cva,
                exp_cvb,
                exp_cvc,
                exp_cvspill
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
                ,
                mult_hoist
#endif
            );
            sfpi::finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR)
            // The fallback owns the original DST row until the store below.
            // Apply any compiler-declared raw-class terminal after the numeric
            // hyperbolic/exp composition, just as the generic polynomial and
            // abs-residual loops do.
            sfpi::finalize_encoded_raw_domain_actions(
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>(), y);
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
        }  // sfpi fallback branch (see kExpHwTtiReplay)
#elif defined(EXPONENT_ALU_LOG2)
#if LOG_HW_TTI_CANDIDATE
        if constexpr (sfpi::kLogHwTtiReplay) {
            // TTI replay fast path (see the kLogHwTtiReplay comment block for
            // the schedule and byte-identity argument). Per tile: pin the
            // below-prgm Horner coeffs c[DEG-2]..c[DEG-5] into LREG7..LREG4
            // (the same map the sfpi hw_preload_pin fallback uses), record the
            // single-element body once (executing element 0), then 31
            // single-issue replays. The per-tile SETRWC issued by the
            // copy_tile scaffolding has already zeroed the D-RWC counter;
            // ADDR_MOD_6 advances it +2 per SFPSTORE.
            constexpr uint32_t kLogTtiC7 = __builtin_bit_cast(
                uint32_t, (LOG_HW_DEGREE >= 3) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 2] : 0.0f);
            constexpr uint32_t kLogTtiC6 = __builtin_bit_cast(
                uint32_t, (LOG_HW_DEGREE >= 4) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 3] : 0.0f);
            constexpr uint32_t kLogTtiC5 = __builtin_bit_cast(
                uint32_t, (LOG_HW_DEGREE >= 5) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 4] : 0.0f);
            constexpr uint32_t kLogTtiC4 = __builtin_bit_cast(
                uint32_t, (LOG_HW_DEGREE >= 6) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 5] : 0.0f);
            if constexpr (LOG_HW_DEGREE >= 3) {
                TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG7, kLogTtiC7);
            }
            if constexpr (LOG_HW_DEGREE >= 4) {
                TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG6, kLogTtiC6);
            }
            if constexpr (LOG_HW_DEGREE >= 5) {
                TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG5, kLogTtiC5);
            }
            if constexpr (LOG_HW_DEGREE >= 6) {
                TT_REPLAY_PIN_FP32(ckernel::p_sfpu::LREG4, kLogTtiC4);
            }
            // Record slot count: the LM tail fold shortens the body.
            constexpr uint32_t kLogTtiRecordSlots =
                sfpi::kLogHwLmReplay
                    ? sfpi::kLogHwLmBodySlots
                    : sfpi::kLogHwTtiBodySlots + sfpi::kLogHwTtiRouteSlots;
            // PROBE: raw replay bodies bypass _llk_math_eltwise_sfpu_start_, so on
            // Wormhole ADDR_MOD_SET_Base is 0 and AddrMod field 2 resolves to
            // section 2 -- A2D datacopy's, dest incr 8 -- not our section 6.
            TT_REPLAY_RECORD(kLogTtiRecordSlots);  // record + execute element 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
#if defined(TT_EXPONENT_ALU_LOG2_NONPOSITIVE_CLASS)
            // Eight instructions surround the already-required SFPEXEXP and
            // replace the plain body's leading NOP, so the route costs seven
            // replay slots. L3 defaults to qNaN. The first CC window combines
            // x>=0 with x<target MIN_NORMAL, overwriting +zero and every
            // positive BF16 subnormal with -Inf. The threshold comes from the
            // verifier-bound BF16+DAZ target lattice, not an activation or
            // max-subnormal constant. After closing that action window, the
            // ordered MIN_NORMAL<=x predicate plus biased exponent<255 opens
            // the unchanged core on exactly positive-normal-finite x. L0 is
            // dead until ABS below, so both threshold and exponent scratch are
            // overwritten normally. NODEBIAS SFPEXEXP and the selected core's
            // original L1-=127 remain in their original arithmetic order.
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0x7FC0);  // qNaN default / load gap
            TTI_SFPEXEXP(
                0,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG1,
                sfpi::SFPEXEXP_MOD1_NODEBIAS);  // selected-core biased exponent
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                0,
                TT_EXPONENT_ALU_LOG2_INPUT_MIN_NORMAL_BITS >> 16);  // BF16 MIN_NORMAL
            TTI_SFPSETCC(
                0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPGT(
                0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 1);  // x < MIN_NORMAL
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0xFF80);  // positive exponent-zero -> -Inf
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPLE(
                0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 1);  // MIN_NORMAL <= x
            TTI_SFPIADD(
                0xF01,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG0,
                sfpi::SFPIADD_MOD1_ARG_IMM |
                    sfpi::SFPIADD_MOD1_CC_LT0);  // biased exponent - 255 < 0
#else
            TTI_SFPNOP;  // load -> exexp gap (no counter-evidence that 0 is safe)
            TTI_SFPEXEXP(
                0,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG1,
                sfpi::SFPEXEXP_MOD1_NODEBIAS);
#endif
            TTI_SFPSETEXP(127, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG2, 1);  // m (imm exponent)
            TTI_SFPIADD(0xF81, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 5);  // e -= 127 (imm, no CC)
#ifdef LOG_HW_BASIS_M_MINUS_1
            // Operand order mirrors the sfpi-emitted word exactly
            // (a=LCONST_1, b=m): 1.0*m + (-1.0) — bit-identical either way.
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LCONST_neg1,
                ckernel::p_sfpu::LREG2,
                0);  // u = m + (-1.0) (hw const L11; MAD-unit producer)
#endif
            TTI_SFPABS(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);  // |e| (int); u-gap filler
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG3,
                0);  // h = c[D]*u + c[D-1]
            TTI_SFPCAST(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // float(|e|); h-gap filler
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG2,
                (LOG_HW_DEGREE >= 3) ? ckernel::p_sfpu::LREG7 : ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG3,
                0);  // rung 1: + c[D-2] (or c0 == 0 -> hw zero at DEG 2)
            TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 0);  // e_f={sgn e,mag |e|}
            if constexpr (LOG_HW_DEGREE >= 3) {
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG3,
                    ckernel::p_sfpu::LREG2,
                    (LOG_HW_DEGREE >= 4) ? ckernel::p_sfpu::LREG6 : ckernel::p_sfpu::LCONST_0,
                    ckernel::p_sfpu::LREG3,
                    0);  // rung 2 (gapped by the SETSGN above)
            }
            if constexpr (LOG_HW_DEGREE >= 4) {
                TTI_SFPNOP;
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG3,
                    ckernel::p_sfpu::LREG2,
                    (LOG_HW_DEGREE >= 5) ? ckernel::p_sfpu::LREG5 : ckernel::p_sfpu::LCONST_0,
                    ckernel::p_sfpu::LREG3,
                    0);  // rung 3
            }
            if constexpr (LOG_HW_DEGREE >= 5) {
                TTI_SFPNOP;
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG3,
                    ckernel::p_sfpu::LREG2,
                    (LOG_HW_DEGREE >= 6) ? ckernel::p_sfpu::LREG4 : ckernel::p_sfpu::LCONST_0,
                    ckernel::p_sfpu::LREG3,
                    0);  // rung 4
            }
            if constexpr (LOG_HW_DEGREE >= 6) {
                TTI_SFPNOP;
                TTI_SFPMAD(
                    ckernel::p_sfpu::LREG3,
                    ckernel::p_sfpu::LREG2,
                    ckernel::p_sfpu::LCONST_0,
                    ckernel::p_sfpu::LREG3,
                    0);  // rung 5: + c0 == 0 (hw zero, gate-proved)
            }
            if constexpr (sfpi::kLogHwLmReplay) {
                // LM tail fold (see the kLogHwLmReplay comment block).
                TTI_SFPNOP;  // h-gap (kept from the plain tail)
                // SFPLOADMACRO seq 0, VD=L0 (lreg_ind = macro_sel<<2 | VD&3,
                // VD>>2 rides dest_reg_addr bit 0): loads the input row into
                // L0 (dead data), latches this element's store address,
                // advances D-RWC += 2, arms MUL/RND/STORE per sequence reg 0.
                TTI_SFPLOADMACRO(0, 0, TT_ADDRMOD_DST_ADV, 0);
                TTI_SFPADD(
                    ckernel::p_sfpu::LCONST_1,
                    ckernel::p_sfpu::LREG1,
                    ckernel::p_sfpu::LREG3,
                    ckernel::p_sfpu::LREG0,
                    0);  // e_float + h -> L0 (=VD; feeds the macro chain)
            } else {
            TTI_SFPNOP;
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG3,
                0);  // e_float + h (a=LCONST_1 form, mirrors the sfpi sfpadd)
            if constexpr ((float)(LOG_HW_SCALE) != 1.0f) {
                TTI_SFPNOP;
                TTI_SFPMUL(
                    ckernel::p_sfpu::LREG3,
                    ckernel::p_sfpu::LREG12,
                    ckernel::p_sfpu::LCONST_0,
                    ckernel::p_sfpu::LREG3,
                    0);  // * LOG_HW_SCALE (prgm0; exact-0.0 addend == sfpmul)
            }
#if defined(TT_EXPONENT_ALU_LOG2_NONPOSITIVE_CLASS)
            // This occupies the existing final MAD-hazard gap and restores all
            // lanes before the common RNE/store.  Invalid and endpoint lanes
            // therefore carry only the prelude's class value through storage.
            TTI_SFPENCC(0, 0, 0, 0);
#else
            TTI_SFPNOP;
#endif
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG3,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_ADV, 0);  // dest += 2
            }  // kLogHwLmReplay
            TT_REPLAY_REST(kLogTtiRecordSlots, 31);  // elements 1..31
            if constexpr (sfpi::kLogHwLmReplay) {
                // Drain: the deepest macro slot (STORE) fires 6 instructions
                // past the final body slot (scale variant; 4 for no-scale).
                // 6 SFPNOPs cover both — nothing may follow until the last
                // in-flight macro op lands (L5/L10: SETRWC and the commit
                // must see a drained pipe).
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPNOP;
            }
            TT_REPLAY_TILE_EPILOGUE();
        } else
#endif  // LOG_HW_TTI_CANDIDATE
        {
        // Below-prgm Horner coefficients pinned in LREGs (see the exp2 branch
        // above for why plain hoisted locals are NOT enough: GCC constant-folds
        // them back into per-element SFPLOADI pairs — 2 instructions per fp32
        // coefficient per element, the measured log/log10/log2 loss mechanism).
        constexpr float log_pin_a = (LOG_HW_DEGREE >= 2) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 2] : 0.0f;
        constexpr float log_pin_b = (LOG_HW_DEGREE >= 3) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 3] : 0.0f;
        constexpr float log_pin_c = (LOG_HW_DEGREE >= 4) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 4] : 0.0f;
        constexpr float log_pin_d = (LOG_HW_DEGREE >= 5) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 5] : 0.0f;
        sfpi::vFloat log_cva = 0.0f;
        sfpi::vFloat log_cvb = 0.0f;
        sfpi::vFloat log_cvc = 0.0f;
        sfpi::vFloat log_cvd = 0.0f;
        if constexpr (LOG_HW_DEGREE >= 2) {
            log_cva = sfpi::hw_preload_pin<sfpi::LRegs::LReg7>(log_pin_a);
        }
        if constexpr (LOG_HW_DEGREE >= 3) {
            log_cvb = sfpi::hw_preload_pin<sfpi::LRegs::LReg6>(log_pin_b);
        }
        if constexpr (LOG_HW_DEGREE >= 4) {
            log_cvc = sfpi::hw_preload_pin<sfpi::LRegs::LReg5>(log_pin_c);
        }
        if constexpr (LOG_HW_DEGREE >= 5) {
            log_cvd = sfpi::hw_preload_pin<sfpi::LRegs::LReg4>(log_pin_d);
        }
        float log_cvspill[(LOG_HW_DEGREE >= 6) ? (LOG_HW_DEGREE - 5) : 1];
        if constexpr (LOG_HW_DEGREE >= 6) {
#pragma GCC unroll 16
            for (int k = 0; k < (int)LOG_HW_DEGREE - 5; k++) {
                log_cvspill[k] = LOG_HW_COEFFS[(int)LOG_HW_DEGREE - 6 - k];
            }
        }
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::log_hw_eval_preloaded<LOG_HW_DEGREE>(
                x, log_cva, log_cvb, log_cvc, log_cvd, log_cvspill);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
        }  // sfpi fallback branch (see kLogHwTtiReplay)
#elif defined(EXPONENT_ALU_LOG1P)
#if LOG1P_TTI_CANDIDATE
        if constexpr (sfpi::kLog1pTtiReplay) {
            // log1p TTI replay fast path (see the kLog1pTtiReplay comment
            // block). Per tile: 2 SFPLOADI pins (c0 fp32 halves -> L7, the
            // same register the sfpi path pins), record the single-element
            // body once (executing element 0), 31 replays, SETRWC SET_D.
            //
            // Body = 1:1 transcription of the sfpi/GCC BH loop of
            // log1p_hw_eval_preloaded<2> with explicit gap slots after every
            // MAD-family producer. Registers: L7 = c0 pin, L12 = ln2*2^-23
            // (prgm0), L13/L14 = c2/c1 (prgm1/2), scratch L0-L3.
            constexpr uint32_t kL1pC0Bits = __builtin_bit_cast(uint32_t, LOG1P_HW_COEFFS[0]);
            constexpr int kL1pBodyLen =
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
                sfpi::TT_NORMALIZED_LOG1P_BODY_SLOTS;
#else
                27 + LOG1P_HW_DEGREE;
#endif
            // C-VER-2 capacity check. kLog1pTtiReplay gates on shape only
            // (LOG1P_HW_DEGREE == 2), so nothing else bounds this body against
            // the 32-slot buffer; instructions past slot 31 execute once at
            // record time and then vanish from every replay.
            static_assert(sfpi::tti_replay_fits(kL1pBodyLen),
                          "log1p replay body exceeds the 32-slot replay buffer");
            TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, (kL1pC0Bits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, (kL1pC0Bits & 0xFFFFu));
            TT_REPLAY_RECORD(kL1pBodyLen);  // record + execute element 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);            // x
            TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2,
                       ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG1, 0); // u = x + 1.0      [MAD]
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
            // The selected reduction already owns u=x+1.  Seed the single
            // verifier-proved target-egress quotient while the MAD retires,
            // then execute the unchanged core only for u>=0.  x==-1 remains
            // in the core and naturally returns -Inf; every x<-1 lane retains
            // +Inf, matching the BH BF16 ingress quotient shared by NaN/Inf.
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_UPPER,
                (sfpi::TT_NORMALIZED_LOG1P_QUOTIENT_BITS >> 16));
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG1,
                0,
                sfpi::SFPSETCC_MOD1_LREG_GTE0);
#endif
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                0,
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
                (sfpi::TT_NORMALIZED_LOG1P_LOWER_BITS >> 16));
#else
                0x3F40);
#endif
            TTI_SFPIADD(0, 1, 0, 6);                                          // e = bits(u) - bits(0.75)
            TTI_SFPSETMAN(0, 0, 0, 1);                                        // e: mantissa = 0
            TTI_SFPMOV(0, 0, 1, 2);                                           // L1 = e
            TTI_SFPIADD(0, 2, 1, 6);                                          // L1 = bits(x) - e  (2^-k x)
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0xC080);                  // L3 = -4.0
            TTI_SFPMOV(0, 0, 2, 2);                                           // L2 = e (x dead)
            TTI_SFPIADD(0, 3, 2, 6);                                          // L2 = bits(-4) - e (s)
            TTI_SFPMULI(0xBE80, ckernel::p_sfpu::LREG2, 0);                   // s *= -0.25 (exact) [MAD]
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
            TTI_SFPLOAD(
                ckernel::p_sfpu::LREG4,
                sfpi::SFPLOAD_MOD0_FMT_UINT16,
                TT_ADDRMOD_DST_HOLD,
                0);  // raw BF16 word in the mandatory MAD gap
#else
            TTI_SFPNOP;
#endif
            TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2,
                       ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG2, 0);  // t = . - 1.0  [MAD]
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG5,
                sfpi::SFPLOADI_MOD0_USHORT,
                sfpi::TT_NORMALIZED_LOG1P_RAW_EQUAL_VALUE);  // MAD gap
#else
            TTI_SFPNOP;
#endif
            TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG1,
                       ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, 0);    // r = 2^-k x + t   [MAD]
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
            TTI_SFPXOR(
                0,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG4,
                0);  // raw ^ 0x00ff in the mandatory MAD gap
#else
            TTI_SFPNOP;
#endif
            if constexpr (LOG1P_HW_DEGREE == 2) {
                TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG1,
                           ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG3, 0); // p = c2*r + c1 [MAD]
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
                TTI_SFPAND(
                    ckernel::p_sfpu::LREG5,
                    ckernel::p_sfpu::LREG4,
                    ckernel::p_sfpu::LREG5,
                    1);  // exponent==0xff iff result is zero; MAD gap
#else
                TTI_SFPNOP;
#endif
                TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1,
                           ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0); // p = p*r + c0 [MAD]
            } else {
                TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG1,
                           ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG3, 0); // p = c1*r + c0 [MAD]
            }
            TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0); // rr = r*r         [MAD]
            // LREG2 must retain rr until the next MAD, so use the otherwise
            // free LREG6 to hide exponent abs in rr's mandatory producer gap.
            // This is a dependency-only schedule move: the integer exponent
            // and all arithmetic operands are unchanged.
            TTI_SFPABS(0, 0, 6, 0);                                           // L6 = |e| (int); MAD gap
            TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);    // res = rr*p + r   [MAD]
            TTI_SFPCAST(6, 6, 0);                                             // L6 = e_float
            TTI_SFPSETSGN(0, 6, 0, 0);                                        // L0 = copysgn(e_float, e)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);    // res += e_f*(ln2*2^-23) [MAD]
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
            TTI_SFPENCC(3, 0, 0, 10);                                         // close core; MAD gap
#else
            TTI_SFPNOP;
#endif
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0,
                              ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0,
                              sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);          // fp32 -> bf16 RNE
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
            // Open the raw exponent-FF terminal as the final two recorded
            // issues.  The immediate suffix closes it and stores in the same
            // row before the next replay; no second element loop is needed.
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG5,
                0,
                sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_UPPER,
                (sfpi::TT_NORMALIZED_LOG1P_QUOTIENT_BITS >> 16));
            auto same_row_suffix = []() {
                TTI_SFPENCC(0, 0, 0, 0);
                TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);
            };
            same_row_suffix();
#else
            TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);           // store, dest += 2
#endif
#pragma GCC unroll 32
            for (int d = 1;
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
                 d < (int)sfpi::TT_NORMALIZED_LOG1P_ROWS;
#else
                 d < 32;
#endif
                 d++) {
                TTI_REPLAY(0, kL1pBodyLen, 0, 0);
#if defined(TT_NORMALIZED_LOG1P_REPLAY)
                same_row_suffix();
#endif
            }
            // Re-zero the shared D-RWC counter (see kExpHwTtiReplay).
            TT_REPLAY_TILE_EPILOGUE();
        } else
#endif  // LOG1P_TTI_CANDIDATE
        {
        // Below-prgm Horner coefficients pinned in LREGs (see the exp2 branch
        // above for why plain hoisted locals are NOT enough: GCC constant-folds
        // them back into per-element SFPLOADI pairs).
        constexpr float l1p_pin_a = (LOG1P_HW_DEGREE >= 2) ? LOG1P_HW_COEFFS[LOG1P_HW_DEGREE - 2] : 0.0f;
        constexpr float l1p_pin_b = (LOG1P_HW_DEGREE >= 3) ? LOG1P_HW_COEFFS[LOG1P_HW_DEGREE - 3] : 0.0f;
        constexpr float l1p_pin_c = (LOG1P_HW_DEGREE >= 4) ? LOG1P_HW_COEFFS[LOG1P_HW_DEGREE - 4] : 0.0f;
        sfpi::vFloat l1p_cva = 0.0f;
        sfpi::vFloat l1p_cvb = 0.0f;
        sfpi::vFloat l1p_cvc = 0.0f;
        if constexpr (LOG1P_HW_DEGREE >= 2) {
            l1p_cva = sfpi::hw_preload_pin<sfpi::LRegs::LReg7>(l1p_pin_a);
        }
        if constexpr (LOG1P_HW_DEGREE >= 3) {
            l1p_cvb = sfpi::hw_preload_pin<sfpi::LRegs::LReg6>(l1p_pin_b);
        }
        if constexpr (LOG1P_HW_DEGREE >= 4) {
            l1p_cvc = sfpi::hw_preload_pin<sfpi::LRegs::LReg5>(l1p_pin_c);
        }
        float l1p_cvspill[(LOG1P_HW_DEGREE >= 2) ? (LOG1P_HW_DEGREE - 1) : 1];
        if constexpr (LOG1P_HW_DEGREE >= 2) {
#pragma GCC unroll 16
            for (int k = 0; k < (int)LOG1P_HW_DEGREE - 1; k++) {
                l1p_cvspill[k] = LOG1P_HW_COEFFS[(int)LOG1P_HW_DEGREE - 2 - k];
            }
        }
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::log1p_hw_eval_preloaded<LOG1P_HW_DEGREE>(
                x, l1p_cva, l1p_cvb, l1p_cvc, l1p_cvspill);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
        }  // sfpi fallback branch (see kLog1pTtiReplay)
#endif
#endif
#if !defined(HW_PRELOAD) || \
    (!defined(EXPONENT_ALU_EXP2) && !defined(EXPONENT_ALU_LOG2) && !defined(EXPONENT_ALU_LOG1P))
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
#if defined(EXPONENT_ALU_EXP2)
            sfpi::vFloat y = sfpi::exp_hw_eval<EXP_HW_DEGREE>(x);
#elif defined(EXPONENT_ALU_LOG2)
            sfpi::vFloat y = sfpi::log_hw_eval<LOG_HW_DEGREE>(x);
#elif defined(EXPONENT_ALU_LOG1P)
#if defined(LOG1P_COMPOSE_ATANH) || defined(LOG1P_COMPOSE_ASINH)
            // log1p COMPOSE arms (atanh/asinh): a precision-generic SFPI
            // precompose (sqrt/reciprocal) wraps the shared Juffa log1p core
            // (log1p_hw_eval), then a postcompose restores the op. The arm owns
            // its own poles/overflow shoulders (domain actions empty), exactly
            // like the reduce_exp compose arms. Host twin:
            // ttpoly/precision/rangered.py::_eval_log1p_compose (atanh 2.79 ULP,
            // asinh 1.60 ULP, 0 invalid over the full fp32 domain).
            sfpi::vFloat y;
            {
                using namespace sfpi;
                vFloat a = setsgn(x, 0);                          // |x|
#if defined(LOG1P_COMPOSE_ATANH)
                // atanh(x) = copysign(0.5 * log1p(2|x|/(1-|x|)), x). 1-|x| is
                // exact by Sterbenz for |x| in [0.5,1) (no near-pole cancel);
                // the poles |x|>=1 are owned by the declared class terminals.
                vFloat den = 1.0f - a;
                vFloat z = (2.0f * a) *
                    ckernel::sfpu::sfpu_reciprocal_iter<2>(den);
                y = copysgn(0.5f * log1p_hw_eval<LOG1P_HW_DEGREE>(z), x);
#else
                // asinh(x): RECIPROCAL-FREE two-zone (avoids the sqrt/reciprocal
                // interaction that shipped as a large silicon error).
                //   |x| <  ASINH_SMALL_THRESHOLD -> |x|*small_correction(x^2), the
                //     declared even-ratio core (NO sqrt, NO reciprocal; exact at 0).
                //   |x| >= ASINH_SMALL_THRESHOLD -> log1p(|x| + (sqrt(x^2+1)-1)) with
                //     root-1 DIRECT: for |x|>=threshold, root=sqrt(x^2+1)>=sqrt(1+thr^2)
                //     so root-1 has NO catastrophic cancellation and needs NO reciprocal.
                //   |x| > sqrt(FP32_MAX): x^2 overflows, fold to ln(2|x|)=ln2+log1p(|x|).
                // sfpu_sqrt_custom is self-contained (no Prgm init). Host twin:
                // rangered._eval_log1p_compose asinh (1.9 ULP, 0 invalid, full domain).
                constexpr float ASINH_SAFE = 1.8e19f;             // sqrt(FP32_MAX)
                constexpr float ASINH_LN2 = 0.69314718055994530942f;
                vFloat s = a * a;
                // small zone: |x| * small_correction(x^2) (natural [0,thr^2] basis Horner)
                vFloat ratio = ASINH_SMALL_COEFFS[ASINH_SMALL_DEGREE];
#pragma GCC unroll 16
                for (int i = (int)ASINH_SMALL_DEGREE - 1; i >= 0; i--) {
                    ratio = ratio * s + ASINH_SMALL_COEFFS[i];
                }
                vFloat small_val = a * ratio;
                // large zone: log1p(|x| + (sqrt(x^2+1)-1)); root-1 direct, no reciprocal
                vFloat root = ckernel::sfpu::sfpu_sqrt_custom<false>(s + 1.0f);
                vFloat z = a + (root - 1.0f);
                v_if(a > ASINH_SAFE) { z = a; }
                v_endif;
                vFloat l = log1p_hw_eval<LOG1P_HW_DEGREE>(z);
                v_if(a > ASINH_SAFE) { l = l + ASINH_LN2; }
                v_endif;
                v_if(a < ASINH_SMALL_THRESHOLD) { l = small_val; }
                v_endif;
                y = copysgn(l, x);
#endif
            }
#else
            sfpi::vFloat y = sfpi::log1p_hw_eval<LOG1P_HW_DEGREE>(x);
#endif
#elif defined(EXPONENT_ALU_POW)
            sfpi::vFloat y = sfpi::pow_hw_eval<POW_HW_DEGREE>(x);
#else
#error "EVAL_METHOD_EXPONENT_ALU requires EXPONENT_ALU_EXP2/LOG2/LOG1P/POW"
#endif
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_NEWTON_ROOT || defined(EVAL_METHOD_NEWTON_ROOT)
        (void)p_lut;
#if defined(TT_NEWTON_TYPED_RAW_TERMINAL_PREPASS)
        sfpi::newton_typed_raw_terminal_prepass();
#endif
#if defined(NEWTON_ROOT_BACKEND_NATIVE_SFPU_TILE) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    (((NEWTON_ROOT_N == 3) && defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)) || \
     ((NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL) && \
      defined(NEWTON_ROOT_ALGORITHM_SQRT_23BIT)))
#if defined(TT_INLINE_PROGRAM_OVERLAY) && \
    !defined(TT_INLINE_PROGRAM_TARGET_EXACT_CORE_IDENTITY)
#error "typed core overlay requires route-aware native root lowering; refusing evaluator demotion"
#else
#if NEWTON_CBRT_EXACT_DESCRIPTOR
        // The helper contains the single shared physical-row body.  The
        // validation define selects 32 direct issues inside the helper;
        // production records once and replays the other 31 rows.
        sfpi::newton_cbrt_tti_tile();
#elif NEWTON_ROOT_N == 3
        ckernel::cbrt_tile(0);
#else
#if NEWTON_RSQRT_TTI_CANDIDATE
        if constexpr (sfpi::kNewtonRsqrtTtiReplay) {
            // Exact replay transcription of Blackhole's public
            // _calculate_sqrt_body_<false, true, false>.  Arithmetic order,
            // program constants, terminal predicates, RNE conversion, and
            // output classes are unchanged; five NOPs are the mandatory
            // producer/consumer gaps of the replay issue contract.  Advancing
            // the destination in the store removes only the surrounding RISC
            // loop/increment.
            TT_REPLAY_RECORD(sfpi::kNewtonRsqrtTtiBodySlots);
            TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, TT_ADDRMOD_DST_HOLD, 0);  // 0: x
            TTI_SFPSHFT(
                0xFFF,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG0,
                5);  // 1: i = bits(x) >> 1
            TTI_SFPIADD(
                0,
                ckernel::p_sfpu::LREG12,
                ckernel::p_sfpu::LREG0,
                6);  // 2: y = MAGIC - i
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG2,
                0);  // 3: xy = x*y
            TTI_SFPNOP;  // 4: xy producer gap
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG2,
                1);  // 5: c = -(y*xy)
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0x7F80);  // 6: +inf / c gap
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG4,
                0);  // 7: C2 + c
            TTI_SFPNOP;  // 8: add producer gap
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG2,
                0);  // 9: t = c*(C2+c) + C1
            TTI_SFPNOP;  // 10: MAD producer gap
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG0,
                0);  // 11: y = y*t
            TTI_SFPNOP;  // 12: y producer gap
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG2,
                0);  // 13: xy = x*y
            TTI_SFPNOP;  // 14: xy producer gap
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG2,
                1);  // 15: one_minus_xyy = 1 - y*xy
            TTI_SFPDIVP2(
                0xFFF,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG5,
                1);  // 16: half_y = y/2; independent gap
            TTI_SFPMOV(
                0,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG4,
                2);  // 17: x_bits
            TTI_SFPIADD(
                0,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG4,
                6);  // 18: infinity_bits - x_bits
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG4,
                0,
                sfpi::SFPSETCC_MOD1_LREG_NE0);  // 19: x != inf
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG1,
                0,
                sfpi::SFPSETCC_MOD1_LREG_NE0);  // 20: and x != 0
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                0);  // 21: corrected y
            TTI_SFPCOMPC(0, 0, 0, 0);  // 22: else
            TTI_SFPMOV(
                0,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG0,
                0);  // 23: zero/inf terminal
            TTI_SFPENCC(3, 0, 0, 10);  // 24: close terminal branch
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);  // 25: x < 0
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x7FC0);  // 26: qNaN
            TTI_SFPENCC(3, 0, 0, 10);  // 27: close negative branch
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // 28: RNE
            TTI_SFPSTORE(
                ckernel::p_sfpu::LREG0,
                0,
                TT_ADDRMOD_DST_ADV,
                0);  // 29: dest += 2
            TT_REPLAY_REST(sfpi::kNewtonRsqrtTtiBodySlots, 31);
            TT_REPLAY_TILE_EPILOGUE();
        } else
#endif  // NEWTON_RSQRT_TTI_CANDIDATE
        {
        ckernel::rsqrt_tile(0);
        }
#endif
#endif
#else
#if NEWTON_TTI_CANDIDATE
        if constexpr (sfpi::kNewtonSqrtTtiReplay) {
            // Plain-replay newton sqrt (see the kNewtonSqrtTtiReplay comment
            // block for the transcription and byte-identity argument). No
            // per-tile pins: everything loop-invariant is in prgm0/1/2.
            TT_REPLAY_RECORD(sfpi::kNewtonSqrtTtiBodySlots);  // record + exec elem 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // 0: x
            TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0x7F80);  // 1: +inf (load-gap filler)
#if defined(ARCH_WORMHOLE)
            // Mod1=5 sets SFPSHFT_MOD1_ARG_IMM_USE_VC (=4), which is
            // Blackhole-only (BlackholeA0/.../SFPSHFT.md:39-40,60). Wormhole
            // masks Mod1 &= 1 and takes the shifted source from VB == VD
            // (WormholeB0/.../SFPSHFT.md:20-26), so this would ignore LREG2 and
            // compute LREG0 >>= 1 -- and LREG0 still holds the PREVIOUS
            // element's result from slot 24's SFP_STOCH_RND. That self-feeding
            // seed overflows to inf and is the measured 1598-non-finite defect.
            // Stage x into the shift's destination first.
            TTI_SFPMOV(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);  // 2a: L0 = x
            TTI_SFPSHFT(0xFFF, 0, ckernel::p_sfpu::LREG0, 1);                  // 2: i = bits(x)>>1
#else
            TTI_SFPSHFT(0xFFF, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 5);  // 2: i = bits(x)>>1
#endif
            TTI_SFPIADD(0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 6);  // 3: y = MAGIC - i
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG1,
                0);  // 4: xy = x*y
#if defined(ARCH_WORMHOLE)
            // SFPMAD_MOD1_NEGATE_VA (=1) is Blackhole-only
            // (BlackholeA0/.../SFPMAD.md:50-51); WH SFPMAD Mod1 has only
            // INDIRECT_VA/VD (WormholeB0/.../SFPMAD.md:44-47) and SFPMUL is
            // "identical to SFPMAD" (WormholeB0/.../SFPMUL.md:3), so mod1=1 is
            // silently dropped and this would compute +y*xy. Materialise -y with
            // SFPMOV_MOD1_NEGATE (=1, WormholeB0/.../SFPMOV.md:41-43): an exact
            // sign-bit XOR on the simple sub-unit, whose page has no scheduling
            // clause, so it fits the gap NOP slot 5 already needed and is
            // readable on the next cycle. L3 is dead here (written slot 8, last
            // read slot 10), so no slot is added.
            TTI_SFPMOV(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, 1);  // 5: ny = -y
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG1,
                0);  // 6: c = ny*xy
#else
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
            TTI_SFPLOAD(
                ckernel::p_sfpu::LREG7,
                sfpi::SFPLOAD_MOD0_FMT_UINT16,
                TT_ADDRMOD_DST_HOLD,
                0);  // 5: original physical BF16
#else
            TTI_SFPNOP;  // 5
#endif
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG1,
                1);  // 6: c = -(y*xy)
#endif
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG5,
                sfpi::SFPLOADI_MOD0_USHORT,
                sfpi::TT_SELECTED_NEWTON_SQRT_RAW_EXPONENT_MASK);  // 7
#else
            TTI_SFPNOP;  // 7
#endif
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG14,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                0);  // 8: C2 + c
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
            TTI_SFPNOP;  // 9
#else
            TTI_SFPNOP;  // 9
#endif
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG13,
                ckernel::p_sfpu::LREG1,
                0);  // 10: t = c*(C2+c) + C1
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
            TTI_SFPAND(
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG7,
                ckernel::p_sfpu::LREG3,
                1);  // 11: raw exponent marker after L3's final core read
#else
            TTI_SFPNOP;  // 11
#endif
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG0,
                0);  // 12: y = y*t
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG5,
                sfpi::SFPLOADI_MOD0_USHORT,
                sfpi::TT_SELECTED_NEWTON_SQRT_NEGATIVE_SIGN_MASK);  // 13
#else
            TTI_SFPNOP;  // 13
#endif
            TTI_SFPMUL(
                ckernel::p_sfpu::LREG2,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LCONST_0,
                ckernel::p_sfpu::LREG1,
                0);  // 14: xy = x*y
#if defined(ARCH_WORMHOLE)
            // Second NEGATE_VA site; same substitution, same dead L3, same
            // reuse of the gap NOP at slot 15. No slot added.
            TTI_SFPMOV(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, 1);  // 15: ny = -y
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG4,
                0);  // 16: 1 - y*xy
#else
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
            TTI_SFPAND(
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG7,
                ckernel::p_sfpu::LREG7,
                1);  // 15: raw sign marker
#else
            TTI_SFPNOP;  // 15
#endif
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG4,
                1);  // 16: 1 - y*xy
#endif
            TTI_SFPDIVP2(0xFFF, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5, 1);  // 17: xy/2
            TTI_SFPIADD(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG6, 2);  // 18: CC: x < +inf
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG0,
                0);  // 19: y = (1-y*xy)*(xy/2) + xy (predicated)
            TTI_SFPENCC(3, 0, 0, 10);  // 20: CC-balance window 1
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, 0);  // 21: CC: x < 0
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x7FC0);  // 22: y = qNaN (predicated)
            TTI_SFPENCC(3, 0, 0, 10);  // 23: CC-balance window 2
#if defined(TT_SELECTED_NEWTON_SQRT_RAW_ZERO_TERMINAL)
#if defined(TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_TERMINAL)
            // The graph-derived partition is complete and disjoint: biased
            // exponent zero maps to +0, then its negative-sign partition maps
            // to the requested +Inf class. Classifier work occupies the
            // existing MAD gaps; this six-slot suffix keeps the body 32/32.
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG3,
                0,
                sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 24: exponent-zero
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_FLOATB,
                0x0000);  // 25: DAZ +0
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG7,
                0,
                sfpi::SFPSETCC_MOD1_LREG_NE0);  // 26: negative exponent-zero
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_FLOATB,
                sfpi::TT_SELECTED_NEWTON_SQRT_NEGATIVE_EXP_ZERO_OUTPUT_BF16);  // 27
            TTI_SFPENCC(0, 0, 0, 0);  // 28: closes combined predicate
            TTI_SFPNOP;  // 29
#else
            // The Newton seed consumes raw sign/mantissa bits before ordinary
            // arithmetic can observe the target DAZ quotient.  Reload the
            // original physical U16 only after every Newton temporary is dead;
            // LOADI supplies the raw-load hazard gap.  Physical exponent bits
            // are 7:0, so AND==0 selects exactly all 256 exponent-zero words.
            TTI_SFPLOAD(
                ckernel::p_sfpu::LREG3,
                sfpi::SFPLOAD_MOD0_FMT_UINT16,
                TT_ADDRMOD_DST_HOLD,
                0);  // 24: raw physical BF16
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG1,
                sfpi::SFPLOADI_MOD0_USHORT,
                sfpi::TT_SELECTED_NEWTON_SQRT_RAW_EXPONENT_MASK);  // 25: load gap/mask
            TTI_SFPAND(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG3,
                1);  // 26: raw exponent
            TTI_SFPSETCC(
                0,
                ckernel::p_sfpu::LREG3,
                0,
                sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 27: exponent-zero
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_FLOATB,
                0x0000);  // 28: declared DAZ result +0
            TTI_SFPENCC(0, 0, 0, 0);  // 29: row-local CC closure
#endif
#endif
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // 30: fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);  // 31: dest += 2
            TT_REPLAY_REST(sfpi::kNewtonSqrtTtiBodySlots, 31);  // elements 1..31
            TT_REPLAY_TILE_EPILOGUE();
        } else
#endif  // NEWTON_TTI_CANDIDATE
        {
        for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
            sfpi::vUInt raw_u16 =
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
#endif
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::newton_root_eval<0>(x);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
            sfpi::finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
        }  // sfpi fallback branch (see kNewtonSqrtTtiReplay)
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_TRIG_RESIDUAL || defined(EVAL_METHOD_TRIG_RESIDUAL)
        (void)p_lut;
        // NOTE: TRIG_RESIDUAL_DEGREE is a constexpr, NOT a macro — it cannot
        // be tested with #if (the preprocessor silently evaluates it as 0).
#if TRIG_TTI_CANDIDATE
        if constexpr (sfpi::kTrigTtiReplay) {
            // TTI replay fast path (see the kTrigTtiReplay comment block).
            // Per tile: 4 (deg-5) / 6 (deg-7) SFPLOADI pins for the odd poly
            // coefficients, record the single-element body once (executing
            // element 0), 31 single-issue replays, then SETRWC SET_D.
            //
            // The body transcribes the sfpi/GCC BH loop 1:1 (same arithmetic
            // ops, same order, same rounding); NOP/LOADI fillers provide the
            // >=2-instruction gap after every MAD-family producer. Loop-
            // invariant bf16/fp16 immediates (BIAS/half/two/P0/P1) stay as
            // in-body SFPLOADIs — they occupy slots a gap NOP would occupy
            // anyway and cost no LREG pins.
            //
            // Registers: L4 = c[top], L5 = c[top-2] (+ L6 = c[top-4], deg 7)
            // pinned per tile; L12/L13 = Cody-Waite tail (prgm0/1), L14 =
            // 1/pi (prgm2); scratch L0-L3 (+L7 as x home for sine).
            constexpr int trig_top =
                (TRIG_RESIDUAL_DEGREE % 2 == 1) ? (int)TRIG_RESIDUAL_DEGREE : (int)TRIG_RESIDUAL_DEGREE - 1;
            constexpr uint32_t kTrigChBits = __builtin_bit_cast(uint32_t, TRIG_RESIDUAL_COEFFS[trig_top]);
            constexpr uint32_t kTrigCmBits = __builtin_bit_cast(uint32_t, TRIG_RESIDUAL_COEFFS[trig_top - 2]);
            constexpr uint32_t kTrigClBits =
                __builtin_bit_cast(uint32_t, TRIG_RESIDUAL_COEFFS[(trig_top >= 7) ? (trig_top - 4) : 1]);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, (kTrigChBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, (kTrigChBits & 0xFFFFu));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, (kTrigCmBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, (kTrigCmBits & 0xFFFFu));
            if constexpr (trig_top >= 7) {
                TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, (kTrigClBits >> 16));
                TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, (kTrigClBits & 0xFFFFu));
            }
#if defined(TT_EXPERIMENT_PRELUDE_ISSUES)
            // EXPERIMENT ONLY: element-0 prelude, issued BEFORE the record so
            // it is not captured into the replay buffer.
            if constexpr (TT_EXPERIMENT_PRELUDE_ISSUES > 0) {
                sfpi::tt_experiment_prelude_row();
            }
#endif
#if defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD)
            // sine reduce: z = x/pi; tmp = z + BIAS; sign = tmp<<31;
            // j = tmp - BIAS; a = (((x + j*P0) + j*P1) + j*T2) + j*T3.
#if defined(ARCH_WORMHOLE)
            constexpr int kTrigWhShftFix = 1;  // extra SFPMOV in the WH sign-extract form
#else
            constexpr int kTrigWhShftFix = 0;
#endif
            constexpr int kTrigBodyLen = ((trig_top >= 7) ? 27 : 26) + kTrigWhShftFix;
            static_assert(sfpi::tti_replay_fits(kTrigBodyLen), "trig sine replay body exceeds 32 slots");
            TT_REPLAY_RECORD(kTrigBodyLen);  // record + execute element 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG7, 0, TT_ADDRMOD_DST_HOLD, 0);            // x
            TTI_SFPMUL(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG14,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0); // z = x*(1/pi)     [MAD]
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x4B40);                  // L0 = BIAS (gap)
            TTI_SFPADDI(0x4B40, ckernel::p_sfpu::LREG1, 0);                   // tmp = z + BIAS   [MAD]
            TTI_SFPNOP;                                                       // (tmp gap)
#if defined(ARCH_WORMHOLE)
            // SFPSHFT_MOD1_ARG_IMM_USE_VC (=4) is Blackhole-only
            // (BlackholeA0/.../SFPSHFT.md:58-60). Wormhole masks Mod1 &= 1 and
            // takes the shifted source from VB == VD (WormholeB0/.../SFPSHFT.md
            // :20-26), so it would shift L2's stale content. Stage tmp into L2.
            TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 0);  // L2 = tmp
            TTI_SFPSHFT(0x01F, 0, 2, 1);                                      // L2 = sign = tmp<<31
            // SFPMAD_MOD1_NEGATE_VA (=1) is Blackhole-only
            // (BlackholeA0/.../SFPMAD.md:50-51); Wormhole defines only
            // INDIRECT_VA/INDIRECT_VD and has no negate step, so it would
            // compute tmp + BIAS. Negate via LCONST_neg1 instead: -1.0 * BIAS
            // is exact (BIAS = 1.5*2^23), so the MAD sees bit-identical inputs.
            TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
#else
            TTI_SFPSHFT(0x01F, 1, 2, 7);                                      // L2 = sign = tmp<<31
            TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 1);
#endif    // j = tmp - BIAS   [MAD]
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0xC049);                  // L3 = P0 (gap)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3,
                       ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);    // a = j*P0 + x     [MAD]
            TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 1, 0x93EC);                  // L1 = P1 fp16a (gap)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                       ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);    // a += j*P1        [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);    // a += j*T2        [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG13,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);    // a += j*T3        [MAD]
            TTI_SFPNOP;
            TTI_SFPXOR(0, 2, 0, 0);                                           // a ^= sign
#elif defined(TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD)
            // cosine reduce: z = x/pi + 0.5; tmp = z + BIAS; sign = tmp<<31;
            // q = tmp - BIAS; j = q*2 - 1; a = Cody-Waite tail as above.
#if defined(ARCH_WORMHOLE)
            constexpr int kTrigWhShftFix = 1;  // extra SFPMOV in the WH sign-extract form
#else
            constexpr int kTrigWhShftFix = 0;
#endif
            constexpr int kTrigBodyLen = ((trig_top >= 7) ? 30 : 29) + kTrigWhShftFix;
            static_assert(sfpi::tti_replay_fits(kTrigBodyLen), "trig cosine replay body exceeds 32 slots");
            TT_REPLAY_RECORD(kTrigBodyLen);  // record + execute element 0
            TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, TT_ADDRMOD_DST_HOLD, 0);            // x
            TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 0, 0x3F00);                  // L1 = 0.5
            TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG14,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);    // z = x*(1/pi)+0.5 [MAD]
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x4B40);                  // L0 = BIAS (gap)
            TTI_SFPADDI(0x4B40, ckernel::p_sfpu::LREG1, 0);                   // tmp = z + BIAS   [MAD]
            TTI_SFPNOP;
#if defined(ARCH_WORMHOLE)
            // SFPSHFT_MOD1_ARG_IMM_USE_VC (=4) is Blackhole-only
            // (BlackholeA0/.../SFPSHFT.md:58-60). Wormhole masks Mod1 &= 1 and
            // takes the shifted source from VB == VD (WormholeB0/.../SFPSHFT.md
            // :20-26), so it would shift L2's stale content. Stage tmp into L2.
            TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 0);  // L2 = tmp
            TTI_SFPSHFT(0x01F, 0, 2, 1);                                      // L2 = sign = tmp<<31
            // SFPMAD_MOD1_NEGATE_VA (=1) is Blackhole-only
            // (BlackholeA0/.../SFPMAD.md:50-51); Wormhole defines only
            // INDIRECT_VA/INDIRECT_VD and has no negate step, so it would
            // compute tmp + BIAS. Negate via LCONST_neg1 instead: -1.0 * BIAS
            // is exact (BIAS = 1.5*2^23), so the MAD sees bit-identical inputs.
            TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
#else
            TTI_SFPSHFT(0x01F, 1, 2, 7);                                      // L2 = sign = tmp<<31
            TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 1);
#endif    // q = tmp - BIAS   [MAD]
            TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 0, 0x4000);                  // L1 = 2.0 (gap)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                       ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG0, 0);  // j = q*2 - 1  [MAD]
            TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 0, 0xBFC9);                  // L1 = P0 (gap)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                       ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 0);    // a = j*P0 + x     [MAD]
            TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 1, 0x8FEC);                  // L1 = P1 fp16a (gap)
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                       ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);    // a += j*P1        [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);    // a += j*T2        [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG13,
                       ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);    // a += j*T3        [MAD]
            TTI_SFPNOP;
            TTI_SFPXOR(0, 2, 0, 0);                                           // a ^= sign
#endif
            // Shared odd-polynomial tail: s = a*a; r = ch*s + cm
            // (deg 7: r = r*s + cl); c = a*s; res = r*c + a; bf16 RNE; store.
            TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0); // s = a*a          [MAD]
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG2,
                       ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, 0);    // r = ch*s + cm    [MAD]
            TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2,
                       ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 0); // c = a*s          [MAD]
            if constexpr (trig_top >= 7) {
                TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2,
                           ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1, 0);  // r = r*s + cl   [MAD]
            }
            TTI_SFPNOP;
            TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3,
                       ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);    // res = r*c + a    [MAD]
            TTI_SFPNOP;
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0,
                              ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0,
                              sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);          // fp32 -> bf16 RNE
            TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);           // store, dest += 2
#if defined(TT_EXPERIMENT_PRELUDE_ISSUES)
            // EXPERIMENT ONLY: per-row prelude ahead of each replayed body.
#pragma GCC unroll 32
            for (int d = 1; d < 32; d++) {
                if constexpr (TT_EXPERIMENT_PRELUDE_ISSUES > 0) {
                    sfpi::tt_experiment_prelude_row();
                }
                TTI_REPLAY(0, kTrigBodyLen, 0, 0);
            }
#else
#pragma GCC unroll 32
            for (int d = 1; d < 32; d++) {
                TTI_REPLAY(0, kTrigBodyLen, 0, 0);
            }
#endif
            // Re-zero the D-RWC the ADDR_MOD_6 walk advanced to 64: the next
            // tile's A2D datacopy MOP shares this counter (see the
            // kExpHwTtiReplay comment block; whole-tile corruption at
            // double-buffer-periodic strides without this).
            TT_REPLAY_TILE_EPILOGUE();
        } else
#endif  // TRIG_TTI_CANDIDATE
        {
            // Hoist the 3 (deg-7) / 2 (deg-5) odd coefficients into LREGs ONCE
            // per tile; the 32-iteration body then holds zero 32-bit constant
            // loads (was 2x SFPLOADI per coeff per element) and drops the
            // redundant SFPIADD quadrant subtract.
            constexpr int trig_top =
                (TRIG_RESIDUAL_DEGREE % 2 == 1) ? (int)TRIG_RESIDUAL_DEGREE : (int)TRIG_RESIDUAL_DEGREE - 1;
            // Pin the 3 odd-polynomial coefficients into LReg5/6/7 for the
            // whole 32-iteration dst loop (the ckernel_sfpu_gelu.h:220-260
            // l_reg idiom: load once, read into locals, write back after the
            // loop so the values are live-out and the register allocator can
            // neither clobber nor rematerialize them). A plain hoisted vFloat
            // local does NOT survive: SFPI/GCC rematerializes constant
            // sfpxloadi at every use (2x SFPLOADI per coefficient per element,
            // 6 wasted instructions/vector at degree 7).
            // LReg4/5/6 ONLY — never LReg7. Silicon (BH) intermittently
            // clobbers LReg7 mid-loop (observed: 6.5% of outputs corrupted,
            // same input value nondeterministically right/wrong — value-
            // independent register trash, not a math bug). Consistent with
            // tt-metal's own BH llk_sfpu headers, which pin LReg0/1/2/4/5/6
            // and never LReg7 (ckernel_sfpu_gelu.h etc.).
            sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vFloat(TRIG_RESIDUAL_COEFFS[trig_top]);
            sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vFloat(TRIG_RESIDUAL_COEFFS[trig_top - 2]);
            sfpi::l_reg[sfpi::LRegs::LReg6] =
                sfpi::vFloat(TRIG_RESIDUAL_COEFFS[(trig_top >= 7) ? (trig_top - 4) : 1]);
            sfpi::vFloat trig_ch = sfpi::l_reg[sfpi::LRegs::LReg4];
            sfpi::vFloat trig_cm = sfpi::l_reg[sfpi::LRegs::LReg5];
            sfpi::vFloat trig_cl = sfpi::l_reg[sfpi::LRegs::LReg6];
            // Indexed dst_reg[d] addressing, NOT the TTNN dst_reg[0]+dst_reg++
            // shape: on silicon the raw 32-iteration dst_reg++ walk drifts the
            // write counter at the harness's block boundaries (whole tiles at
            // ~16-tile strides came back as neighbors' results — value-mixed,
            // position-clustered corruption). TTNN only ever advances dst_reg++
            // inside 8-iteration LLK-managed chunks that reset the RWC.
            for (int d = 0; d < 32; d++) {
                sfpi::vFloat x_raw = sfpi::dst_reg[d];
                sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
                sfpi::vFloat y = sfpi::trig_residual_odd_eval_hoisted<TRIG_RESIDUAL_DEGREE>(
                    x, trig_ch, trig_cm, trig_cl);
                sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
                y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
                sfpi::dst_reg[d] = y;
            }
            sfpi::l_reg[sfpi::LRegs::LReg4] = trig_ch;
            sfpi::l_reg[sfpi::LRegs::LReg5] = trig_cm;
            sfpi::l_reg[sfpi::LRegs::LReg6] = trig_cl;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_TAN_STANDALONE || defined(EVAL_METHOD_TAN_STANDALONE)
        (void)p_lut;
        // 3rd Cody-Waite -pi/2 tail term. It must NOT live in Prgm0 (the
        // quadrant reciprocal owns it — PRGM CONSTANT OWNERSHIP RULE) and it
        // is not bf16-exact (0xb4a88000, mantissa bit 15 set), so a plain
        // local would re-materialize as 2x SFPLOADI per element: pin it into
        // an LREG once per tile instead (same write/read launder as the
        // exp preload pins).
        sfpi::vFloat tan_tail_p2 = sfpi::hw_preload_pin<sfpi::LRegs::LReg7>(-0x1.51p-22f);
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::tan_standalone_eval<TAN_STANDALONE_DEGREE>(x, tan_tail_p2);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ASIN_ACOS || defined(EVAL_METHOD_ASIN_ACOS)
#if defined(EVAL_METHOD_HALF_ANGLE_RATIO_1)
        for (int d = 0; d < 32; d++) {
#if defined(TT_HALF_ANGLE_TYPED_RAW_NEG_NAN_TERMINAL) || \
    defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
            sfpi::vUInt raw_u16 =
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>();
#endif
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat y =
                sfpi::asin_acos_eval<POLY_DEGREE, LUT_SIZE>(x_raw, *p_lut);
            // The closed form owns invalid-domain replacement in
            // asin_acos_eval; NaN/Inf, signed zero, and finite pass-through
            // are structural consequences of that same body.  Re-emitting
            // the generic finalizer would duplicate already-owned actions
            // and lengthen every in-domain lane. Encoded raw-class actions
            // remain a disjoint typed overlay and are applied only when the
            // selected semantic profile declares one.
#if defined(TT_HALF_ANGLE_TYPED_RAW_NEG_NAN_TERMINAL)
            // S55 proves over all 65,536 words that this sign/exponent prefix,
            // minus its one infinity word, is exactly the negative-NaN class.
            static_assert(TT_HALF_ANGLE_RAW_NEG_NAN_PATTERN_COUNT == 127u);
            static_assert(TT_HALF_ANGLE_RAW_NEG_NAN_PREDICATE_SLOTS == 3u);
            static_assert(TT_HALF_ANGLE_RAW_NEG_NAN_PREDICATE_PEAK_LIVE <= 2u);
            sfpi::vUInt exponent_and_sign =
                raw_u16 & sfpi::vUInt(TT_HALF_ANGLE_RAW_NEG_NAN_EQUAL_MASK);
            v_if(
                exponent_and_sign ==
                    sfpi::vUInt(TT_HALF_ANGLE_RAW_NEG_NAN_EQUAL_VALUE) &&
                raw_u16 !=
                    sfpi::vUInt(TT_HALF_ANGLE_RAW_NEG_NAN_EXCLUDED_WORD)) {
                static_assert(
                    TT_HALF_ANGLE_RAW_NEG_NAN_RESULT <= 4u,
                    "half-angle raw terminal requires a nonfinite/zero class");
                if constexpr (TT_HALF_ANGLE_RAW_NEG_NAN_RESULT == 0u) {
                    y = std::numeric_limits<float>::quiet_NaN();
                } else if constexpr (TT_HALF_ANGLE_RAW_NEG_NAN_RESULT == 1u) {
                    y = std::numeric_limits<float>::infinity();
                } else if constexpr (TT_HALF_ANGLE_RAW_NEG_NAN_RESULT == 2u) {
                    y = -std::numeric_limits<float>::infinity();
                } else if constexpr (TT_HALF_ANGLE_RAW_NEG_NAN_RESULT == 3u) {
                    y = sfpi::vFloat(0.0f);
                } else {
                    y = sfpi::setsgn(sfpi::vFloat(0.0f), 1);
                }
            }
            v_endif;
#endif
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL) && \
    !defined(TT_HALF_ANGLE_TYPED_RAW_NEG_NAN_TERMINAL)
            sfpi::finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#else
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat x_raw = sfpi::dst_reg[d];
            sfpi::vFloat x = sfpi::prepare_raw_domain_input(x_raw);
            sfpi::vFloat y = sfpi::asin_acos_eval(x);
            sfpi::finalize_raw_domain_actions(x_raw, y);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#endif
#else
#if defined(DST_COEFF_PROBE)
        // Silicon probe for the DST-resident coefficient store (see
        // dst_coeff_store.h). Replaces the eval entirely; output tile rows
        // carry the roundtrip byte decompositions and clobber sentinels.
        (void)p_lut;
        sfpi::dst_coeff_probe();
#elif defined(DST_COEFF_STORE) && DST_COEFF_ELIGIBLE
        // DST-resident coefficient store (see dst_coeff_store.h). The
        // constexpr apply gate keeps single-segment shapes on the TTI replay
        // path and enforces the parked-row capacity bound.
#if defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE)
        static_assert(
            sfpi::kDstCoeffApply,
            "typed same-row gradient finalization requires the DST schedule");
#endif
        if constexpr (sfpi::kDstCoeffApply) {
            sfpi::piecewise_generic_lut_dst_coeff<poly_degree, num_segments, lut_size>(*p_lut);
        } else {
            sfpi::piecewise_generic_lut_dispatch<poly_degree, num_segments, lut_size>(*p_lut);
        }
#else
        sfpi::piecewise_generic_lut_dispatch<poly_degree, num_segments, lut_size>(*p_lut);
#endif
#endif
#endif

#if defined(TT_TARGET_SELECTED_BF16_SCRATCH_BOUNDARY)
        constexpr auto cb_target_scratch = tt::CBIndex::c_24;
        // Cross the selected evaluator's native BF16 packer boundary before
        // invoking the independently declared target phase.
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_target_scratch, 1);
        pack_tile(0, cb_target_scratch);
        cb_push_back(cb_target_scratch, 1);
        tile_regs_release();

        // The input CB deliberately remains at its front.  Reload the raw tile
        // under the same BF16 destination mode, evaluate the parsed target
        // graph, and materialize that result through the same BF16 packer.
        tile_regs_acquire();
        copy_init(cb_in);
        copy_tile(cb_in, 0, 0);
#ifdef TRISC_MATH
        for (int d = 0; d < 32; ++d) {
            sfpi::vFloat x = sfpi::dst_reg[d];
            sfpi::dst_reg[d] = sfpi::target_native_rational_asymptotic_eval(x);
        }
#endif
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_target_scratch, 1);
        pack_tile(0, cb_target_scratch);
        cb_push_back(cb_target_scratch, 1);
        tile_regs_release();

        // Reload selected into D1 first and target into D0 last so SFPI's
        // destination base is D0.  Class comparison therefore sees two
        // independently BF16-egressed values, never an internal FP32 value.
        cb_wait_front(cb_target_scratch, 2);
        tile_regs_acquire();
        copy_init(cb_target_scratch);
        copy_tile(cb_target_scratch, 0, 1);
        copy_tile(cb_target_scratch, 1, 0);
#ifdef TRISC_MATH
        for (int d = 0; d < 32; ++d) {
            sfpi::vFloat target = sfpi::dst_reg[d];
            sfpi::vFloat selected = sfpi::dst_reg[32 + d];
            sfpi::vFloat target_class = sfpi::target_result_class(target);
            sfpi::vFloat selected_class = sfpi::target_result_class(selected);
            v_if(target_class != selected_class) { selected = target; }
            v_endif;
            sfpi::dst_reg[d] = selected;
        }
#endif
#endif

#if defined(FUSE_GRAD_MUL) && \
    !defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) && \
    !defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE)
#ifdef TRISC_MATH
        // Epilogue fusion: every cascade temporary is dead here, so this costs
        // two live vFloats and cannot widen the allocator ICE. See
        // sfpi::fuse_grad_mul above for the full rationale.
        sfpi::fuse_grad_mul();
#endif
#endif
