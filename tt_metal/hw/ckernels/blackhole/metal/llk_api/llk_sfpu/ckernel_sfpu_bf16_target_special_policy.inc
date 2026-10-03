// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The activation policy is mathematical: it applies after the compiler-bound
// target ingress transform above. Codegen supplies class codes from that
// target record, never from an activation name.

static_assert(
    TT_TARGET_BF16_SPECIAL_COMPARE_COUNT <= 3,
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

inline void apply_target_special_policy([[maybe_unused]] vFloat x_effective, [[maybe_unused]] vFloat& result) {
    // Only the three classes reachable through BH BF16 ingress are tested.
    // Both raw NaN signs share the +Inf arm on the common BH SFPU path; raw -0
    // and both subnormal signs share the +0 arm. Apply mathematical policy to
    // that exact target quotient.
    if constexpr (TT_SPECIAL_POS_INF != 5) {
        constexpr float bf16_max_finite = 3.3895313892515355e+38f;
        vFloat positive_special_coordinate = x_effective;
        v_if(positive_special_coordinate > bf16_max_finite) {
            result = target_special_class_value<TT_SPECIAL_POS_INF>(result);
        }
        v_endif;
    }
    negative_infinity_terminal<TT_SPECIAL_NEG_INF>(x_effective, result);
}
