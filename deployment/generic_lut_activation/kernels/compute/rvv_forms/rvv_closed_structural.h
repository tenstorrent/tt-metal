// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Typed closed-structural evaluators for the TRISC2 RVV backend.  Admission is
// by the S55 HardwareSchedule kind; this file consumes only the schedule payload
// emitted by S60.  Unsupported structural kinds remain compile-time errors in
// piecewise_rvv.cpp.

#ifndef RVV_FORMS_RVV_CLOSED_STRUCTURAL_H_
#define RVV_FORMS_RVV_CLOSED_STRUCTURAL_H_

#if defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1)
#define RVK_CLOSED_SHARED_INVERSE_SQUARE 1
#define RVK_FORM_CLOSED_STRUCTURAL 1
#else
#define RVK_FORM_CLOSED_STRUCTURAL 0
#endif

#if RVK_FORM_CLOSED_STRUCTURAL
#if !defined(USE_BF16) || !defined(RVV_FORMS_RVV_BW_IO_H_)
#error "rvv_closed_structural: shared inverse-square requires typed BF16 stage-in/repack"
#endif
#if !defined(TT_SHARED_INVERSE_SQUARE_REPLAY)
#error "rvv_closed_structural: incomplete shared inverse-square schedule payload"
#endif
static_assert(
    TT_SHARED_INVERSE_SQUARE_BODY_SLOTS == 32u, "rvv_closed_structural: unexpected shared inverse-square body");
static_assert(
    TT_SHARED_INVERSE_SQUARE_CORE_PEAK_LIVE <= 8u,
    "rvv_closed_structural: shared inverse-square live set exceeds certificate");
#endif

#if RVK_FORM_CLOSED_STRUCTURAL && defined(TRISC_PACK)

#include <riscv_vector.h>

static inline vuint32m2_t rvk_closed_bits(vfloat32m2_t value) { return __riscv_vreinterpret_v_f32m2_u32m2(value); }

static inline vfloat32m2_t rvk_closed_float(vuint32m2_t value) { return __riscv_vreinterpret_v_u32m2_f32m2(value); }

// Zve32f exposes no division instruction on this target.  Three Newton steps
// from the standard integer reciprocal seed converge beyond fp32 precision.
// The explicit zero/large masks reproduce the typed finite-coordinate
// reciprocal contract used by the SFPU form: signed zero -> signed infinity,
// |x| >= 2^126 -> signed zero.
static inline vfloat32m2_t rvk_closed_reciprocal(vfloat32m2_t x, size_t vl) {
    vfloat32m2_t magnitude = __riscv_vfsgnjx_vv_f32m2(x, x, vl);
    vuint32m2_t seed_bits = __riscv_vrsub_vx_u32m2(rvk_closed_bits(magnitude), 0x7EF311C3u, vl);
    vfloat32m2_t y = rvk_closed_float(seed_bits);
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
#pragma GCC unroll 3
    for (int iteration = 0; iteration < 3; ++iteration) {
        vfloat32m2_t residual = __riscv_vfnmsac_vv_f32m2(one, magnitude, y, vl);
        y = __riscv_vfmacc_vv_f32m2(y, y, residual, vl);
    }
    vbool16_t large = __riscv_vmfge_vf_f32m2_b16(magnitude, 0x1p126f, vl);
    vbool16_t zero = __riscv_vmfeq_vf_f32m2_b16(magnitude, 0.0f, vl);
    y = __riscv_vfmerge_vfm_f32m2(y, 0.0f, large, vl);
    y = __riscv_vfmerge_vfm_f32m2(y, __builtin_inff(), zero, vl);
    return __riscv_vfsgnj_vv_f32m2(y, x, vl);
}

static inline vfloat32m2_t rvk_closed_shared_inverse_square(vfloat32m2_t x, size_t vl) {
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t half = __riscv_vfmv_v_f_f32m2(0.5f, vl);
    vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t absolute = __riscv_vfsgnjx_vv_f32m2(x, x, vl);
    vfloat32m2_t z = __riscv_vfadd_vf_f32m2(absolute, 1.0f, vl);

    // The 1.5*2^23 add/sub is the schedule's declared nearest-integer
    // coordinate construction and is exact for the widened BF16 inputs.
    vfloat32m2_t rounded = __riscv_vfadd_vf_f32m2(x, 0x1.8p23f, vl);
    rounded = __riscv_vfsub_vf_f32m2(rounded, 0x1.8p23f, vl);
    vbool16_t negative = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    vfloat32m2_t negative_weight = __riscv_vmerge_vvm_f32m2(zero, one, negative, vl);
    vfloat32m2_t weighted_integer = __riscv_vfmul_vv_f32m2(negative_weight, rounded, vl);
    vfloat32m2_t q = __riscv_vfsub_vv_f32m2(x, weighted_integer, vl);

    vfloat32m2_t u = rvk_closed_reciprocal(z, vl);
    vfloat32m2_t core = __riscv_vfmul_vf_f32m2(u, SHARED_INVERSE_SQUARE_P0, vl);
    core = __riscv_vfadd_vv_f32m2(core, half, vl);
    core = __riscv_vfmul_vv_f32m2(core, u, vl);
    core = __riscv_vfadd_vv_f32m2(core, one, vl);
    core = __riscv_vfmul_vv_f32m2(core, u, vl);
    core = __riscv_vfsgnj_vv_f32m2(core, x, vl);

    vfloat32m2_t inverse_q = rvk_closed_reciprocal(q, vl);
    vfloat32m2_t result = __riscv_vfmul_vv_f32m2(inverse_q, inverse_q, vl);
    result = __riscv_vfadd_vv_f32m2(result, core, vl);

    // Mask q before q^2 on the positive arm.  This preserves the form's
    // overflow avoidance rather than relying on a late select over NaN.
    vfloat32m2_t regularizer_q = __riscv_vfmul_vv_f32m2(negative_weight, q, vl);
    vfloat32m2_t q2 = __riscv_vfmul_vv_f32m2(regularizer_q, regularizer_q, vl);
    vfloat32m2_t regularizer = __riscv_vfmul_vf_f32m2(q2, SHARED_INVERSE_SQUARE_P2[2], vl);
    regularizer = __riscv_vfadd_vf_f32m2(regularizer, SHARED_INVERSE_SQUARE_P2[1], vl);
    regularizer = __riscv_vfmul_vv_f32m2(regularizer, q2, vl);
    regularizer = __riscv_vfadd_vf_f32m2(regularizer, SHARED_INVERSE_SQUARE_P2[0], vl);
    regularizer = __riscv_vfmul_vv_f32m2(negative_weight, regularizer, vl);
    result = __riscv_vfadd_vv_f32m2(result, regularizer, vl);

    vbool16_t pole = __riscv_vmand_mm_b16(negative, __riscv_vmfeq_vf_f32m2_b16(q, 0.0f, vl), vl);
    result = __riscv_vfmerge_vfm_f32m2(result, __builtin_inff(), pole, vl);

    vbool16_t zero_input = __riscv_vmfeq_vf_f32m2_b16(x, 0.0f, vl);
    result = __riscv_vfmerge_vfm_f32m2(result, __builtin_inff(), zero_input, vl);
    vbool16_t top_coordinate = __riscv_vmfeq_vf_f32m2_b16(x, 0x1p126f, vl);
    result = __riscv_vfmerge_vfm_f32m2(result, 0x1p-126f, top_coordinate, vl);
    return result;
}

static inline void rvk_eval_tile_closed_structural(const float* input, float* output) {
    size_t vl = __riscv_vsetvl_e32m2(8);
    for (int chunk = 0; chunk < 128; ++chunk) {
        vfloat32m2_t x = __riscv_vle32_v_f32m2(input + chunk * 8, vl);
        vfloat32m2_t y = rvk_closed_shared_inverse_square(x, vl);
        y = tt_rvv_finalize_domain_actions(x, y, vl);
        y = tt_rvv_finalize_special_values(x, y, vl);
        __riscv_vse32_v_f32m2(output + chunk * 8, y, vl);
    }
}

#endif  // RVK_FORM_CLOSED_STRUCTURAL && TRISC_PACK
#endif  // RVV_FORMS_RVV_CLOSED_STRUCTURAL_H_
