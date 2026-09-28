// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included inside sfpi. Exact selected evaluator and class repair.
template <class Config>
struct ExponentBucket {
    static inline void apply_target_selected_sfpi_class_terminal(vFloat input, vFloat& result) {
        static_assert(Config::TT_TARGET_CLASS_TERMINAL_PEAK_LIVE <= 8u);
        vUInt bits = as<vUInt>(input);
        vUInt sign = bits >> 31;
        vUInt magnitude = bits & 0x7fffffffu;
        if constexpr (!Config::kFuse) {
            // Only exact negative integer poles of the selected form have nonfinite
            // output on negative finite input. The complete finite lattice proves it.
            v_if((sign != 0u) && (magnitude < 0x7f800000u) && ((as<vUInt>(result) & 0x7fffffffu) >= 0x7f800000u)) {
#pragma GCC unroll 4
                for (uint32_t k = 0; k < Config::TT_TARGET_CLASS_PARTITIONS; ++k) {
                    v_if(magnitude >= ((Config::TT_TARGET_CLASS_FIRST[k] & 0x7fffu) << 16)) {
                        result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_OUTPUT[k] << 16);
                    }
                    v_endif;
                }
            }
            v_endif;
        }
        v_if(Config::kFuse ? (exexp(input, ExponentMode::Biased) == 0) : ((bits & 0x7f800000u) == 0u)) {
            result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_ZERO << 16);
        }
        v_endif;
        v_if(Config::kFuse ? (bits == 0x7f800000u) : ((sign == 0u) && (magnitude == 0x7f800000u))) {
            result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_POS_INF << 16);
        }
        v_endif;
        if constexpr (Config::kFuse) {
            static_assert(Config::TT_TARGET_CLASS_NEG_INF == Config::TT_TARGET_CLASS_NEG_NAN);
            v_if((sign != 0u) && (magnitude >= 0x7f800000u)) {
                result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_NEG_INF << 16);
            }
            v_endif;
        } else {
            v_if((sign != 0u) && (magnitude == 0x7f800000u)) {
                result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_NEG_INF << 16);
            }
            v_endif;
            v_if((sign != 0u) && (magnitude > 0x7f800000u)) {
                result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_NEG_NAN << 16);
            }
            v_endif;
        }
        v_if((sign == 0u) && (magnitude > 0x7f800000u) && ((as<vUInt>(result) & 0x7f800000u) == 0x7f800000u)) {
            result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_POS_NAN << 16);
        }
        v_endif;
    }

#if defined(ARCH_BLACKHOLE)
    template <uint32_t Hold, uint32_t Advance, class Epilogue>
    static inline void exponent_bucket_target_class_repair_tti_tile(Epilogue epilogue) {
        // L6/L7 are graph-derived BF16 transition boundaries.  They are pinned
        // once outside the replay and are disjoint from the selected evaluator,
        // whose temporaries have all retired before this phase starts.
        TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, Config::TT_TARGET_CLASS_TTI_NEG_INF_FIRST);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, Config::TT_TARGET_CLASS_TTI_POS_INF_FIRST);
        TTI_REPLAY(0, Config::TT_TARGET_CLASS_TTI_BODY_SLOTS, 1, 1);
        TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 64);  // selected BF16
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG1,
            0,
            Hold,
            2 * Config::TT_TARGET_CLASS_TTI_RAW_SHADOW_ROW_BASE);  // decoded raw input
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, Config::TT_TARGET_CLASS_TTI_POS_INF_RAW >> 16);
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, Config::TT_TARGET_CLASS_TTI_POS_INF_RAW & 0xffffu);
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3,
                   0);  // selected +Inf delta

        // A selected +Inf on a negative input above the first target transition
        // is an exact negative-integer pole whose target class is finite.
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);  // raw input < 0
        TTI_SFPGT(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1,
                  1);  // raw input > first transition
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, Config::TT_TARGET_CLASS_TTI_FINITE_INTEGER);
        TTI_SFPENCC(0, 0, 0, 0);

        // The closed interval [first -Inf transition, next +Inf transition) is
        // target -Inf.  BF16 spacing proves every value there is an integer, so
        // no observed-value membership table is needed.
        TTI_SFPLE(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1,
                  1);  // raw input <= first transition
        TTI_SFPGT(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1,
                  1);  // raw input > next transition
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, Config::TT_TARGET_CLASS_TTI_NEG_INF);
        TTI_SFPENCC(0, 0, 0, 0);

        // Raw exponent zero (both signs) has a finite target class.
        TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, Config::TT_TARGET_CLASS_TTI_ZERO_SUBNORMAL);
        TTI_SFPENCC(0, 0, 0, 0);

        // Graph-derived representatives carry class parity, not payload/value equality.
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3,
                   1);  // exponent delta
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, Config::TT_TARGET_CLASS_TTI_MAN_MASK >> 16);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, Config::TT_TARGET_CLASS_TTI_MAN_MASK & 0xffffu);
        TTI_SFPAND(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5,
                   1);  // mantissa bits
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, Config::TT_TARGET_CLASS_TTI_NAN);
        if constexpr (Config::TT_TARGET_CLASS_TTI_NEG_NAN != Config::TT_TARGET_CLASS_TTI_NAN) {
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, Config::TT_TARGET_CLASS_TTI_NEG_NAN);
        }
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);
#pragma GCC unroll 32
        for (int replay = 0; replay < 31; ++replay) {
            TTI_REPLAY(0, Config::TT_TARGET_CLASS_TTI_BODY_SLOTS, 0, 0);
        }
        epilogue();
    }
#endif

    template <uint32_t INDEX>
    static inline auto exponent_bucket_core_coefficient() {
        if constexpr (Config::kStore) {
            return vFloat(dst_reg[64u + INDEX].mode<DataLayout::F32>());
        } else {
            return Config::TT_EXPONENT_BUCKET_COEFFS[INDEX];
        }
    }

    template <uint32_t INDEX>
    static inline auto exponent_bucket_reflection_coefficient() {
        static_assert(INDEX > 0u && INDEX <= 8u && INDEX % 2u == 0u);
        if constexpr (Config::kStore) {
            return vFloat(dst_reg[70u + INDEX / 2u].mode<DataLayout::F32>());
        } else {
            return Config::TT_EXPONENT_BUCKET_REFLECTION_COEFFS[INDEX];
        }
    }

    template <class Reciprocal>
    static inline vFloat exponent_bucket_log_derivative_1_eval(
        vFloat x, uint32_t handoff_row, Reciprocal finite_reciprocal) {
        static_assert(Config::TT_EXPONENT_BUCKET_COUNT == 0u);
        static_assert(Config::TT_EXPONENT_BUCKET_NATIVE_LOG == 0u);
        static_assert(Config::TT_EXPONENT_BUCKET_RECIPROCAL_DEGREE == 2u);
        static_assert(
            Config::TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS == 1u ||
            Config::TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS == 2u);
        static_assert(Config::TT_EXPONENT_BUCKET_REFLECTION_DEGREE == 8u);

        // The selector-free compact form has room to keep inverse_z live through
        // its P3 core.  Keeping this phase in LREGs avoids the first destination
        // handoff and the duplicate abs(x)+1 coordinate reconstruction.  The core
        // still closes once before the independent reflection suffix.
        {
            vFloat z = setsgn(x, 0) + 1.0f;
            vFloat inverse_z = finite_reciprocal(z);
            vInt exponent = exexp(z, ExponentMode::Biased) - 127;
            vFloat mantissa = setexp(z, 127);
            vFloat core = exponent_bucket_core_coefficient<3>() * mantissa + exponent_bucket_core_coefficient<2>();
            core = core * mantissa + exponent_bucket_core_coefficient<1>();
            core = core * mantissa + exponent_bucket_core_coefficient<0>();
            vFloat exponent_value = convert<vFloat>(as<vSMag>(exponent), RoundMode::Nearest);
            core = exponent_value * exponent_bucket_core_coefficient<4>() + core;
            vFloat inverse_correction =
                exponent_bucket_core_coefficient<6>() * inverse_z + exponent_bucket_core_coefficient<5>();
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
        vFloat inverse =
            ckernel::sfpu::sfpu_reciprocal_iter<Config::TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS>(divisor);

        vFloat fraction_square = fraction * fraction;
        vFloat scaled_cot = exponent_bucket_reflection_coefficient<8>();
        scaled_cot = scaled_cot * fraction_square + exponent_bucket_reflection_coefficient<6>();
        scaled_cot = scaled_cot * fraction_square + exponent_bucket_reflection_coefficient<4>();
        scaled_cot = scaled_cot * fraction_square + exponent_bucket_reflection_coefficient<2>();
        scaled_cot = scaled_cot * fraction_square + Config::TT_EXPONENT_BUCKET_REFLECTION_COEFFS[0];

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
            if constexpr (Config::kFuse) {
                result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_OUTPUT[0] << 16);
#pragma GCC unroll 4
                for (uint32_t k = 1; k < Config::TT_TARGET_CLASS_PARTITIONS; ++k) {
                    v_if(as<vUInt>(x) >= (Config::TT_TARGET_CLASS_FIRST[k] << 16)) {
                        result = __builtin_bit_cast(float, Config::TT_TARGET_CLASS_OUTPUT[k] << 16);
                    }
                    v_endif;
                }
            } else {
                result = std::numeric_limits<float>::quiet_NaN();
            }
        }
        v_endif;
        return result;
    }
};
