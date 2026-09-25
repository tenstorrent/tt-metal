// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#if defined(TT_TARGET_BH_BF16_PACK_RELU_RAW_NEG_NAN_REPAIR)
inline void apply_pack_relu_raw_neg_nan_repair(vUInt raw_u16, vFloat& result) {
    // STACC_RELU clears every sign-bit datum, including negative NaN payloads.
    // BH BF16 ingress maps both raw NaN signs to +Inf; restore that quotient
    // before packing so mode 1 passes +Inf and mode 3 applies its finite cap.
    // In the U16 DST view the bytes are swapped relative to an IEEE BF16
    // word: exponent[7] is bit 15, exponent[6:0] are bits 6:0, and sign is
    // bit 7.  0x80ff therefore denotes negative infinity; a nonzero
    // mantissa in bits 14:8 makes exactly the 127 negative NaN payloads.
    vUInt negative_special_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(negative_special_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
    }
    v_endif;
}
#endif


#if defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR)
inline void apply_target_raw_nonfinite_policy(vUInt raw_u16, vFloat& result) {
    // In the physical U16 DST view, exponent[7:0] is bits 7:0. Ignoring the
    // sign therefore selects exactly both infinities and all 254 NaN payloads.
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NONFINITE_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_NONFINITE_CONSTANT);
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR)
inline void apply_target_raw_pos_nonfinite_policy(vUInt raw_u16, vFloat& result) {
    positive_nonfinite_terminal<TT_TARGET_BH_BF16_RAW_POS_NONFINITE_RESULT>(
        raw_u16, result, TT_TARGET_BH_BF16_RAW_POS_NONFINITE_CONSTANT);
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT)
inline void apply_target_raw_signed_nonfinite_split(vUInt raw_u16, vFloat& result) {
    signed_nonfinite_split_terminal<TT_TARGET_BH_BF16_RAW_POS_NONFINITE_RESULT>(
        raw_u16, result, TT_TARGET_BH_BF16_RAW_POS_NONFINITE_CONSTANT);
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR)
inline void apply_target_raw_pos_inf_policy(vUInt raw_u16, vFloat& result) {
    encoded_word_terminal<0x00ffu,TT_TARGET_BH_BF16_RAW_POS_INF_RESULT>(raw_u16,result,TT_TARGET_BH_BF16_RAW_POS_INF_CONSTANT);
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR)
inline void apply_target_raw_neg_inf_policy(vUInt raw_u16, vFloat& result) {
    encoded_word_terminal<0x80ffu,TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT>(raw_u16,result,TT_TARGET_BH_BF16_RAW_NEG_INF_CONSTANT);
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) && defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT)
inline void apply_target_raw_pos_nan_policy(vUInt raw_u16, vFloat& result) {
    positive_nan_class_terminal<TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT>(
        raw_u16, result, TT_TARGET_BH_BF16_RAW_POS_NAN_CONSTANT);
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) && defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT)
inline void apply_target_raw_neg_nan_policy(vUInt raw_u16, vFloat& result) {
#if TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT == 6
    dense_negative_nan_constant<__builtin_bit_cast(uint32_t, TT_TARGET_BH_BF16_RAW_NEG_NAN_CONSTANT)>(raw_u16, result);
#else
    if constexpr (TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT <= 4) {
        negative_nan_class_terminal<TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT, true>(raw_u16, result);
    } else {
    vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(exponent_and_sign_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) {
            result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT>(
                result, TT_TARGET_BH_BF16_RAW_NEG_NAN_CONSTANT);
        }
        v_endif;
    }
    v_endif;
    }
#endif
}
#endif

#if defined(TT_WH_RAW_NAN_UNION)
#if !defined(ARCH_WORMHOLE) || !defined(USE_BF16)
#error "raw NaN union requires WH BF16 transport"
#endif
static_assert(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT == TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT);
static_assert(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT >= 0 && TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT <= 4);
inline void apply_target_raw_nan_union_policy(vUInt raw_u16, vFloat& result) {
    nan_union_class_terminal<TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT>(raw_u16, result);
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR)
inline void apply_target_raw_pos_zero_policy(vUInt raw_u16, vFloat& result) {
    v_if(raw_u16 == vUInt(0x0000u)) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_POS_ZERO_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_POS_ZERO_CONSTANT);
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
inline void apply_target_raw_subnormal_policy(vUInt raw_u16, vFloat& result) {
#if defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR)
    constexpr int positive_code=TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_RESULT;
    constexpr float positive_constant=TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_CONSTANT;
#else
    constexpr int positive_code=-1;
    constexpr float positive_constant=0.0f;
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
    constexpr int negative_code=TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_RESULT;
    constexpr float negative_constant=TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_CONSTANT;
#else
    constexpr int negative_code=-1;
    constexpr float negative_constant=0.0f;
#endif
    encoded_subnormal_terminal<positive_code,negative_code>(raw_u16,result,positive_constant,negative_constant);
}
#endif
