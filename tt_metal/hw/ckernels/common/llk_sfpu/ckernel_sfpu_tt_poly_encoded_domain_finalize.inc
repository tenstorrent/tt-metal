// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR) ||     \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT) ||                                                          \
    defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR) ||                                                     \
    defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) ||   \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) || defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR) ||    \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR) ||                                                           \
    defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) ||                                                     \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
#if defined(ARCH_WORMHOLE) && !defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) && \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT) && \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_RESULT) && \
    TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT == TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_RESULT && \
    TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT != 6
#define TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED 1
#endif
#define TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL 1
inline void finalize_encoded_raw_domain_actions(vUInt raw_u16, vFloat& result) {
#if defined(TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED)
    // Equal class actions cover the full negative exponent-zero partition.
    // One mask avoids keeping separate exponent/mantissa/sign temporaries
    // live beside the ordinary log evaluator's pinned coefficients.
    v_if((raw_u16 & vUInt(0x80ffu)) == vUInt(0x8000u)) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT>(result);
    }
    v_endif;
#endif
#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT)
    apply_target_raw_signed_nonfinite_split(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR)
    apply_target_raw_nan_policy(raw_u16, result);
#endif
#if defined(TT_WH_RAW_NAN_UNION)
    apply_target_raw_nan_union_policy(raw_u16, result);
#else
#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR)
    apply_target_raw_pos_nan_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR)
    apply_target_raw_neg_nan_policy(raw_u16, result);
#endif
#endif
#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER)
    apply_target_raw_signed_nan_policy(raw_u16, result);
#endif
// Compiled raw-class actions are the terminal overlay and therefore run after
// the target ingress quotient's generic NaN repair. Their exact predicates
// are disjoint from every remaining individual action below.
#if defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR)
    apply_target_raw_nonfinite_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR)
    apply_target_raw_pos_nonfinite_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) && !defined(TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED)
#if defined(ARCH_WORMHOLE)
    // WH U16 exposes the SFPU sign at bit 15, exponent in bits 7:0.
    v_if(raw_u16 == vUInt(0x8000u)) {
#else
    // Retain the existing BH transport until its separate raw-layout proof.
    v_if(raw_u16 == vUInt(0x0080u)) {
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT)
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_NEG_ZERO_CONSTANT);
#else
        result = setsgn(vFloat(0.0f), 1);
#endif
    }
    v_endif;
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR)
    apply_target_raw_neg_inf_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR)
    apply_target_raw_pos_inf_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR)
    apply_target_raw_pos_zero_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
#if !defined(TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED)
    apply_target_raw_subnormal_policy(raw_u16, result);
#endif
#endif
}

inline void finalize_encoded_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    (void)x_raw;
    finalize_encoded_raw_domain_actions(raw_u16, result);
}

#if defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
inline void finalize_post_round_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    finalize_post_round_raw_domain_actions(x_raw, result);
    finalize_encoded_raw_domain_actions(raw_u16, result);
}
#endif

inline void finalize_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    finalize_raw_domain_actions(x_raw, result);
    finalize_encoded_raw_domain_actions(x_raw, raw_u16, result);
}
#endif
