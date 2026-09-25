// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// The activation policy is mathematical: it applies after the compiler-bound
// target ingress transform above. Codegen supplies class codes from that
// target record, never from an activation name.
#if defined(TT_SPECIAL_VALUE_POLICY) &&                                                                \
    (defined(TT_TARGET_BH_BF16_SPECIAL_FINALIZER) || defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT) ||  \
     defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT) || \
     defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT))

#ifndef TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT
#define TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT 0
#endif

static_assert(
    TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT <= 3,
    "BH BF16 target special finalizer has invalid structural compare count");
template <int CODE>
inline vFloat target_special_class_value(vFloat computed, float constant = 0.0f) {
    static_assert(CODE >= 0 && CODE <= 6, "unsupported target special class");
    if constexpr (CODE == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    } else if constexpr (CODE == 1) {
        return std::numeric_limits<float>::infinity();
    } else if constexpr (CODE == 2) {
        return -std::numeric_limits<float>::infinity();
    } else if constexpr (CODE == 3) {
        return vFloat(0.0f);
    } else if constexpr (CODE == 4) {
        return setsgn(vFloat(0.0f), 1);
    } else if constexpr (CODE == 5) {
        return computed;  // finite_other is owned by the selected evaluator.
    } else {
        return vFloat(constant);
    }
}

#if defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF) || defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO)
#if defined(TT_SPECIAL_COMPARE_POS_INF) || defined(TT_SPECIAL_COMPARE_NEG_INF) || \
    defined(TT_SPECIAL_COMPARE_POS_ZERO) || defined(TT_TARGET_BH_BF16_ACTION_COORDINATE)
#error "magnitude finalizer arms exclude the per-class compare schedule"
#endif
inline void apply_target_special_policy(vFloat x_effective, vFloat& result) {
    // Merged-magnitude quotient: codegen emits these arms only when the
    // finalizer owns exactly {+Inf, -Inf, +0} effective ingress and the
    // egress narrowing maps the +Inf-class and -Inf-class results to one
    // output class, so both signed infinities may share the +Inf-policy
    // value.  Raw NaN of either sign has already canonicalized to +Inf at
    // the common SFPU DST load; raw -0 and both subnormal signs are exactly
    // the open MIN_NORMAL magnitude band, so no DAZ coordinate is needed.
    static_assert(
        TT_SPECIAL_POS_INF != 5 && TT_SPECIAL_POS_ZERO != 5,
        "magnitude finalizer requires direct structural class values");
    vFloat magnitude = setsgn(x_effective, 0);
#if defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF)
    {
        constexpr float bf16_max_finite = 3.3895313892515355e+38f;
        v_if(magnitude > bf16_max_finite) { result = target_special_class_value<TT_SPECIAL_POS_INF>(result); }
        v_endif;
    }
#endif
#if defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO)
    {
        constexpr float min_normal = std::numeric_limits<float>::min();
        v_if(magnitude < min_normal) { result = target_special_class_value<TT_SPECIAL_POS_ZERO>(result); }
        v_endif;
    }
#endif
}
#else
inline void apply_target_special_policy(vFloat x_effective, vFloat& result) {
    // Only the three classes reachable through BH BF16 ingress are tested.
    // Both raw NaN signs share the +Inf arm on the common BH SFPU path; raw -0
    // and both subnormal signs share the +0 arm. Apply mathematical policy to
    // that exact target quotient.
#if defined(TT_SPECIAL_COMPARE_POS_INF)
    if constexpr (TT_SPECIAL_POS_INF != 5) {
        constexpr float bf16_max_finite = 3.3895313892515355e+38f;
        vFloat positive_special_coordinate = x_effective;
        v_if(positive_special_coordinate > bf16_max_finite) {
            result = target_special_class_value<TT_SPECIAL_POS_INF>(result);
        }
        v_endif;
    }
#endif
#if defined(TT_SPECIAL_COMPARE_NEG_INF)
    negative_infinity_terminal<TT_SPECIAL_NEG_INF>(x_effective, result);
#endif
#if defined(TT_SPECIAL_COMPARE_POS_ZERO)
    zero_class_terminal<TT_SPECIAL_POS_ZERO>(x_effective,result);
#endif
}
#endif  // merged-magnitude vs per-class compare schedule

#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR)
inline void apply_target_raw_nan_policy(vUInt raw_u16, vFloat& result) {
#if TT_SPECIAL_NAN == 5
#error "raw NaN finite-other policy has no direct structural value"
#else
    nan_union_class_terminal<TT_SPECIAL_NAN>(raw_u16, result);
#endif
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) && !defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT)
inline void apply_target_raw_pos_nan_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x80ffu);
    v_if(exponent_delta == 0u) {
        v_if((raw_u16 & vUInt(0x8000u)) == 0u) {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) {
#if TT_SPECIAL_NAN == 5
#error "raw positive NaN finite-other policy has no direct structural value"
#else
                result = target_special_class_value<TT_SPECIAL_NAN>(result);
#endif
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) && !defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT)
inline void apply_target_raw_neg_nan_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(exponent_delta == 0u) {
        v_if((raw_u16 & vUInt(0x8000u)) != 0u) {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) {
#if TT_SPECIAL_NAN == 5
#error "raw negative NaN finite-other policy has no direct structural value"
#else
                result = target_special_class_value<TT_SPECIAL_NAN>(result);
#endif
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER)
inline void apply_target_raw_signed_nan_policy(vUInt raw_u16, vFloat& result) {
    signed_nan_class_terminal<TT_SPECIAL_POS_INF, TT_SPECIAL_NEG_INF>(raw_u16, result);
}
#endif

#else
inline void apply_target_special_policy(vFloat x_effective, vFloat& result) {
    (void)x_effective;
    (void)result;
}
#endif
