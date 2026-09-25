// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Single-pass evaluator integration point.  Every evaluator calls this while
// its raw-coordinate vector is still live, after reconstruction/postcompose
// and before precision conversion, gradient multiplication, or store.  The
// no-action arm compiles away, preserving legacy generated source behavior.
inline void finalize_raw_domain_actions(vFloat x_raw, vFloat& result) {
    vFloat x_effective = target_domain_action_coordinate(x_raw);
#if defined(TT_SELECTED_WH_SIGNED_ABS_RECIPROCAL)
    vFloat magnitude = setsgn(x_raw, 0);
    v_if(magnitude > TT_SELECTED_RECIP_CORE_BOUND) {
        result = TT_SELECTED_RECIP_COMPLEMENT - result;
    }
    v_endif;
    result = copysgn(result, x_raw);
#endif
#if defined(TT_WH_NORMALIZED_LOG1P_TERMINALS)
    // WH BF16 ingress aliases positive NaN payloads with positive infinity.
    // The ordinary normalized-log owner consumes this effective class once.
    v_if(exexp(x_raw, ExponentMode::Biased) == 255 && x_raw >= 0.0f) {
        result = std::numeric_limits<float>::infinity();
    }
    v_endif;
#endif
#if defined(TT_WH_EXPONENT_ALU_LOG2_TERMINALS)
    // The typed log owner includes DAZ and both signed nonfinite classes.
    v_if(x_raw < 0.0f) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    vFloat magnitude = setsgn(x_raw, 0);
    v_if(magnitude < std::numeric_limits<float>::min()) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
    vInt exponent = exexp(x_raw, ExponentMode::Biased);
    v_if(exponent == 255) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
#endif
#if defined(TT_SELECTED_CORE_AGGREGATE_STIRLING)
    apply_selected_core_aggregate_stirling(x_raw, result);
#elif defined(TT_SELECTED_CORE_TOTAL_FORM)
    apply_selected_core_total_form(x_effective, result);
#endif
#if defined(TT_DOMAIN_ACTION_PROGRAM)
    apply_raw_pre_domain_actions(x_effective, result);
#if defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP)
#if defined(ARCH_WORMHOLE)
    result = dense_lower_clamp<true, __builtin_bit_cast(uint32_t, TT_DOMAIN_ACTION_LOWER_TERMINAL_VALUE)>(result);
#else
    result = dense_lower_clamp<false, __builtin_bit_cast(uint32_t, TT_DOMAIN_ACTION_LOWER_TERMINAL_VALUE)>(result);
#endif
#endif
#else
    (void)x_raw;
    (void)result;
#endif
    // Ordered finite/domain return_class actions own their lanes first.  The
    // effective special policy then has final class authority; target rounding
    // and FTZ happen in the evaluator's existing conversion/store epilogue.
#if !defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR) && !defined(TT_SELECTED_WH_AGGREGATE_STIRLING) && \
    !defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND)
    apply_target_special_policy(x_effective, result);
#endif
#if defined(TT_TARGET_BH_FP32_RAW_NEG_ZERO_DISCRIMINATOR)
    // FP32 ingress preserves the source bits.  Restore -0 after affine/clamp
    // arithmetic, which may otherwise canonicalize 1*(-0)+0 to +0.
    v_if(as<vUInt>(x_raw) == vUInt(0x80000000u)) { result = setsgn(vFloat(0.0f), 1); }
    v_endif;
#endif
}
