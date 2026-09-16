// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared Zve32f lowering for typed raw-coordinate domain actions.
// Include only from TRISC_PACK after <riscv_vector.h>.

#pragma once

#if defined(TT_DOMAIN_ACTION_PROGRAM)

#if defined(TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS)

static inline vfloat32m2_t tt_rvv_apply_symmetric_constant_raw_tails(
    vfloat32m2_t x_raw, vfloat32m2_t result, size_t vl) {
    static_assert(TT_DOMAIN_ACTION_COUNT == 2, "symmetric tails require exactly two actions");
    constexpr auto first = TT_DOMAIN_ACTION_DATA[0];
    constexpr auto second = TT_DOMAIN_ACTION_DATA[1];
    static_assert(first.action_kind == 0 && second.action_kind == 0, "symmetric tails require constant actions");
    static_assert(first.inclusive == second.inclusive, "symmetric tails require matching inclusivity");
    static_assert(
        first.inclusive == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE,
        "symmetric tail certificate does not match action records");
    static_assert(
        (first.direction == 0 && second.direction == 1 && first.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
         second.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) ||
            (first.direction == 1 && second.direction == 0 && first.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
             second.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND),
        "symmetric tail certificate does not match action bounds");
    static_assert(
        first.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE && second.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE,
        "symmetric tail certificate does not match action values");

    vfloat32m2_t absolute_raw = __riscv_vfsgnjx_vv_f32m2(x_raw, x_raw, vl);
    vbool16_t matches;
    if constexpr (TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE != 0) {
        matches = __riscv_vmfge_vf_f32m2_b16(absolute_raw, TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND, vl);
    } else {
        matches = __riscv_vmfgt_vf_f32m2_b16(absolute_raw, TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND, vl);
    }
    vfloat32m2_t action_value = __riscv_vfmv_v_f_f32m2(TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE, vl);
    return __riscv_vmerge_vvm_f32m2(result, action_value, matches, vl);
}

#else

template <uint32_t INDEX>
static inline vfloat32m2_t tt_rvv_apply_domain_action(vfloat32m2_t x_raw, vfloat32m2_t result, size_t vl) {
    constexpr auto record = TT_DOMAIN_ACTION_DATA[INDEX];
    static_assert(record.action_kind <= 4, "unsupported domain-action kind");
    static_assert(record.return_class <= 4, "unsupported domain-action return class");
    vfloat32m2_t action_value = __riscv_vfmv_v_f_f32m2(record.value, vl);
    if constexpr (record.action_kind == 1) {
        action_value = x_raw;
    } else if constexpr (record.action_kind == 2) {
        action_value = __riscv_vfmadd_vf_f32m2(x_raw, record.scale, __riscv_vfmv_v_f_f32m2(record.bias, vl), vl);
    } else if constexpr (record.action_kind == 3) {
        float class_value = __builtin_nanf("");
        if constexpr (record.return_class == 1) {
            class_value = __builtin_inff();
        } else if constexpr (record.return_class == 2) {
            class_value = -__builtin_inff();
        } else if constexpr (record.return_class == 3) {
            class_value = 0.0f;
        } else if constexpr (record.return_class == 4) {
            class_value = -0.0f;
        }
        action_value = __riscv_vfmv_v_f_f32m2(class_value, vl);
    } else if constexpr (record.action_kind == 4) {
        vfloat32m2_t magnitude = __riscv_vfmv_v_f_f32m2(__builtin_inff(), vl);
        action_value = __riscv_vfsgnj_vv_f32m2(magnitude, x_raw, vl);
    }
    vbool16_t matches;
    if constexpr (record.direction == 0 && record.inclusive != 0) {
        matches = __riscv_vmfle_vf_f32m2_b16(x_raw, record.bound, vl);
    } else if constexpr (record.direction == 0) {
        matches = __riscv_vmflt_vf_f32m2_b16(x_raw, record.bound, vl);
    } else if constexpr (record.inclusive != 0) {
        matches = __riscv_vmfge_vf_f32m2_b16(x_raw, record.bound, vl);
    } else {
        matches = __riscv_vmfgt_vf_f32m2_b16(x_raw, record.bound, vl);
    }
    result = __riscv_vmerge_vvm_f32m2(result, action_value, matches, vl);
    if constexpr (INDEX > 0) {
        return tt_rvv_apply_domain_action<INDEX - 1>(x_raw, result, vl);
    }
    return result;
}

#endif  // TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS

#endif  // TT_DOMAIN_ACTION_PROGRAM

// Single-pass RVV evaluator integration point.  Call immediately before the
// evaluator's existing store, while x_raw is already resident in a vector.
static inline vfloat32m2_t tt_rvv_finalize_domain_actions(vfloat32m2_t x_raw, vfloat32m2_t result, size_t vl) {
#if defined(TT_DOMAIN_ACTION_PROGRAM)
    static_assert(TT_DOMAIN_ACTION_COUNT > 0, "domain-action program must not be empty");
#if defined(TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS)
    return tt_rvv_apply_symmetric_constant_raw_tails(x_raw, result, vl);
#else
    return tt_rvv_apply_domain_action<TT_DOMAIN_ACTION_COUNT - 1>(x_raw, result, vl);
#endif
#else
    (void)x_raw;
    (void)vl;
    return result;
#endif
}

// ---------------------------------------------------------------------------
// Declared special-value policy (activations/<op>.json special_value_policy).
//
// The compensated evaluator had NO special-value handling at all: an infinite
// input ran through segment indexing and compensated Horner and emerged NaN,
// and a negative zero lost its sign in `0 + (-0)*k = +0`. Both contradict the
// policy the spec already declares and the full-domain gate already checks
// (device_conformance_status / special_policy_mismatch_count).
//
// Class codes, emitted by run_csv.sh codegen:
//   0 nan   1 pos_inf   2 neg_inf   3 pos_zero   4 neg_zero
//   5 finite_other -> PASS THROUGH. The policy only asserts finiteness, not a
//     value; the evaluator owns the number. Substituting a constant here would
//     be inventing an asymptote (see T5_EXHAUSTIVE_BLOCKERS.md, cause A).
//
// NOTE ON SUBNORMALS: BF16 RVV staging applies target DAZ to the raw encoding
// before widening and preserves its sign. Consequently this finalizer sees a
// signed zero for a DAZ input and applies the declared zero-class action. The
// final BF16 pack separately applies target output FTZ after RNE.
// ---------------------------------------------------------------------------
#if defined(TT_SPECIAL_VALUE_POLICY)

template <int CODE>
static inline vfloat32m2_t tt_rvv_special_class_value(vfloat32m2_t computed, size_t vl, float constant = 0.0f) {
    if constexpr (CODE == 0) {
        return __riscv_vfmv_v_f_f32m2(__builtin_nanf(""), vl);
    } else if constexpr (CODE == 1) {
        return __riscv_vfmv_v_f_f32m2(__builtin_inff(), vl);
    } else if constexpr (CODE == 2) {
        return __riscv_vfmv_v_f_f32m2(-__builtin_inff(), vl);
    } else if constexpr (CODE == 3) {
        return __riscv_vfmv_v_f_f32m2(0.0f, vl);
    } else if constexpr (CODE == 4) {
        return __riscv_vfmv_v_f_f32m2(-0.0f, vl);
    } else if constexpr (CODE == 6) {
        return __riscv_vfmv_v_f_f32m2(constant, vl);
    } else {
        return computed;  // finite_other: the evaluator's value stands
    }
}

static inline vfloat32m2_t tt_rvv_finalize_special_values(vfloat32m2_t x_raw, vfloat32m2_t result, size_t vl) {
#if defined(TT_RVV_TTNN_RAW_CLASS_POLICY)
    // The RVV implementation is being checked against the TTNN Blackhole
    // API profile.  Apply the compiler-composed raw-class contract, not only
    // its mathematical middle layer: TTNN's BF16 ingress/egress can map the
    // two NaN signs and signed zero differently from IEEE RVV arithmetic.
    vbool16_t m_nan = __riscv_vmfne_vv_f32m2_b16(x_raw, x_raw, vl);
    vfloat32m2_t sign_key = __riscv_vfsgnj_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), x_raw, vl);
    vbool16_t m_negative = __riscv_vmflt_vf_f32m2_b16(sign_key, 0.0f, vl);
    vfloat32m2_t nan_value = __riscv_vmerge_vvm_f32m2(
        tt_rvv_special_class_value<TT_RVV_TTNN_RAW_POS_NAN>(result, vl, TT_RVV_TTNN_RAW_POS_NAN_CONSTANT),
        tt_rvv_special_class_value<TT_RVV_TTNN_RAW_NEG_NAN>(result, vl, TT_RVV_TTNN_RAW_NEG_NAN_CONSTANT),
        m_negative,
        vl);
    result = __riscv_vmerge_vvm_f32m2(result, nan_value, m_nan, vl);

    vbool16_t m_pos_inf = __riscv_vmfeq_vf_f32m2_b16(x_raw, __builtin_inff(), vl);
    result = __riscv_vmerge_vvm_f32m2(
        result,
        tt_rvv_special_class_value<TT_RVV_TTNN_RAW_POS_INF>(result, vl, TT_RVV_TTNN_RAW_POS_INF_CONSTANT),
        m_pos_inf,
        vl);
    vbool16_t m_neg_inf = __riscv_vmfeq_vf_f32m2_b16(x_raw, -__builtin_inff(), vl);
    result = __riscv_vmerge_vvm_f32m2(
        result,
        tt_rvv_special_class_value<TT_RVV_TTNN_RAW_NEG_INF>(result, vl, TT_RVV_TTNN_RAW_NEG_INF_CONSTANT),
        m_neg_inf,
        vl);

    vbool16_t m_zero = __riscv_vmfeq_vf_f32m2_b16(x_raw, 0.0f, vl);
    vfloat32m2_t zero_value = __riscv_vmerge_vvm_f32m2(
        tt_rvv_special_class_value<TT_RVV_TTNN_RAW_POS_ZERO>(result, vl, TT_RVV_TTNN_RAW_POS_ZERO_CONSTANT),
        tt_rvv_special_class_value<TT_RVV_TTNN_RAW_NEG_ZERO>(result, vl, TT_RVV_TTNN_RAW_NEG_ZERO_CONSTANT),
        m_negative,
        vl);
    return __riscv_vmerge_vvm_f32m2(result, zero_value, m_zero, vl);
#else
    // NaN input: x != x. Done first so a NaN can never be reclassified below.
    if constexpr (TT_SPECIAL_NAN != 5) {
        vbool16_t m = __riscv_vmfne_vv_f32m2_b16(x_raw, x_raw, vl);
        result = __riscv_vmerge_vvm_f32m2(result, tt_rvv_special_class_value<TT_SPECIAL_NAN>(result, vl), m, vl);
    }
    if constexpr (TT_SPECIAL_POS_INF != 5) {
        vbool16_t m = __riscv_vmfeq_vf_f32m2_b16(x_raw, __builtin_inff(), vl);
        result = __riscv_vmerge_vvm_f32m2(result, tt_rvv_special_class_value<TT_SPECIAL_POS_INF>(result, vl), m, vl);
    }
    if constexpr (TT_SPECIAL_NEG_INF != 5) {
        vbool16_t m = __riscv_vmfeq_vf_f32m2_b16(x_raw, -__builtin_inff(), vl);
        result = __riscv_vmerge_vvm_f32m2(result, tt_rvv_special_class_value<TT_SPECIAL_NEG_INF>(result, vl), m, vl);
    }
    // Zeros. +0 and -0 compare EQUAL, so the sign is recovered from a
    // sign-carrying magnitude rather than an integer compare (which keeps this
    // to intrinsics already used elsewhere in this tree). The per-lane action
    // value is selected by sign first, then merged only where x == 0, so no
    // mask algebra is needed.
    if constexpr (TT_SPECIAL_POS_ZERO != 5 || TT_SPECIAL_NEG_ZERO != 5) {
        vbool16_t m_zero = __riscv_vmfeq_vf_f32m2_b16(x_raw, 0.0f, vl);
        vfloat32m2_t sgn = __riscv_vfsgnj_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), x_raw, vl);
        vbool16_t m_neg = __riscv_vmflt_vf_f32m2_b16(sgn, 0.0f, vl);
        vfloat32m2_t zval = __riscv_vmerge_vvm_f32m2(
            tt_rvv_special_class_value<TT_SPECIAL_POS_ZERO>(result, vl),
            tt_rvv_special_class_value<TT_SPECIAL_NEG_ZERO>(result, vl),
            m_neg,
            vl);
        result = __riscv_vmerge_vvm_f32m2(result, zval, m_zero, vl);
    }
    return result;
#endif
}

#else
static inline vfloat32m2_t tt_rvv_finalize_special_values(vfloat32m2_t x_raw, vfloat32m2_t result, size_t vl) {
    (void)x_raw;
    (void)vl;
    return result;
}
#endif  // TT_SPECIAL_VALUE_POLICY
