// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// rvv_forms/rvv_logodds.h — RVV (Zve32f, TRISC2 pack thread) lowering of the
// anonymous unit-interval NORMALIZED LOG-ODDS SEPARABLE basis for the RVV-only
// kernel piecewise_rvv.cpp (the bf16 logit-class winner shape).
// =============================================================================
//
// PRODUCTION REFERENCE (mirrored step for step): piecewise_generic.cpp
// normalized_log_odds_separable_eval, whose validated arithmetic model is
// ttpoly/spec/basis.py evaluate_normalized_log_odds_separable. For
// q = min(x, 1-x) = m * 2^e with m in [1,2):
//
//     result = copysign(-ln2 - e*ln2 + (q-0.5)*C(q-0.5) - (m-1)*L(m-1), x-0.5)
//
// with the whole-kernel exterior closes applied AFTER the reconstruction, in
// the production order:
//     x <  0.0  -> +inf
//     x >= 1.0  -> +inf
//     exponent-zero input (zero or subnormal magnitude, either sign) -> -inf
// The last close reads the BF16 raw exponent on the SFPU; here the rvv_bw_io
// stage-in widen is EXACT (bf16 << 16), so "raw exponent zero" is exactly
// "widened fp32 exponent field zero" — the (bits & 0x7F800000) == 0 predicate
// below selects the identical input set. In fp32 mode the same predicate
// selects the fp32 zero/subnormal class, matching the SFPU fp32 dispatch of
// the same body.
//
// CSV layout (flattened, one polynomial-layout segment; static-asserted like
// production): LUT_DATA = [b0, b1, C0, C1, L0, L1, L2]. All five coefficients
// are compile-time constants — no staging, no gathers; the LUT is never read
// at runtime (same stance as the lowering forms).
//
// ROUNDING STANCE (base-kernel contract): every multiply-add below is one
// fused vfmadd/vfmacc where production issues one sfpu_mad, and the shared
// fp32 ln2 constant keeps the structural root at x = 0.5 exactly as in
// production. The RVV engine's MAD rounds independently of the SFPU issue
// order; SFPU byte identity is NOT claimed — the exhaustive host proof /
// harness ULP report vs golden is the arbiter. Both engines are DAZ+FTZ.
//
// This header is self-contained: preprocessor detection is visible to ALL
// three TUs; vector code stays strictly inside TRISC_PACK.
// =============================================================================

#ifndef RVV_FORMS_RVV_LOGODDS_H_
#define RVV_FORMS_RVV_LOGODDS_H_

#if defined(BASIS_NORMALIZED_LOG_ODDS_SEPARABLE)
#define RVK_FORM_LOGODDS 1
#else
#define RVK_FORM_LOGODDS 0
#endif

#if RVK_FORM_LOGODDS
// First qualified separable shape only, exactly like production.
static_assert(BASIS_LOG_ODDS_CORRECTION_DEGREE == 1, "rvv_logodds: first qualified separable correction is P1");
static_assert(BASIS_LOG_ODDS_LOG_DEGREE == 2, "rvv_logodds: first qualified separable log ratio is P2");
static_assert(LUT_SIZE == 7, "rvv_logodds: unit-interval boundaries plus five flattened coefficients");
// The separable basis owns the whole reconstruction: refuse any co-emitted
// basis/parity modifier this form does not reproduce (production emits none).
#if defined(BASIS_INPUT_ABS_X) || defined(BASIS_MUL_ABS_X_BEFORE_POST) || defined(BASIS_MUL_SQRT_1_MINUS_ABS) || \
    defined(BASIS_AFFINE_EVEN) || defined(BASIS_CLAMP_MAX) || defined(BASIS_POST_SIGN_X) ||                      \
    defined(BASIS_POST_REFLECT_PI) || defined(BASIS_LEFT_TAIL_ZERO) || defined(BASIS_RIGHT_TAIL_IDENTITY) ||     \
    defined(BASIS_RIGHT_TAIL_ABS_AFFINE) || defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
#error "rvv_logodds: other basis/parity modifiers are not supported with the separable log-odds basis"
#endif
#if defined(POSTCOMPOSE_AFFINE_Y) || defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT) || defined(PRECOMPOSE_INPUT_AFFINE)
#error "rvv_logodds: pre/postcompose wrappers are not implemented for the log-odds basis"
#endif
#endif  // RVK_FORM_LOGODDS

#if RVK_FORM_LOGODDS && defined(TRISC_PACK)

#include <riscv_vector.h>

// ---- whole-tile evaluator: 1024 fp32 elements, e32m2 (vl = 8), 128 chunks,
// 2-way interleave (element-wise vector ops only — no gathers, no tables).
static inline vfloat32m2_t rvk_logodds_eval_elem(vfloat32m2_t x, size_t vl) {
    // Flattened coefficient layout (production kCoeff = 2): C0,C1 then L0,L1,L2.
    constexpr float kC0 = LUT_DATA[2];
    constexpr float kC1 = LUT_DATA[3];
    constexpr float kL0 = LUT_DATA[4];
    constexpr float kL1 = LUT_DATA[5];
    constexpr float kL2 = LUT_DATA[6];
    constexpr float kLn2 = 0.69314718055994530942f;  // production kLn2 (fp32)

    // q = min(x, 1-x) via the production predicate (x < 0.5 -> q = x), with
    // the complement born from its own subtract exactly as in production.
    vfloat32m2_t comp = __riscv_vfrsub_vf_f32m2(x, 1.0f, vl);  // 1 - x
    vbool16_t m_low = __riscv_vmflt_vf_f32m2_b16(x, 0.5f, vl);
    vfloat32m2_t q = __riscv_vmerge_vvm_f32m2(comp, x, m_low, vl);

    // log_reduce(q): e = exponent field - 127; m = setexp(q, 127). In-domain q
    // is strictly positive and normal (exponent-zero inputs are closed below).
    vuint32m2_t qb = __riscv_vreinterpret_v_f32m2_u32m2(q);
    vint32m2_t e_int = __riscv_vsub_vx_i32m2(
        __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(qb, 23, vl), 0xFFu, vl)),
        127,
        vl);
    vfloat32m2_t m = __riscv_vreinterpret_v_u32m2_f32m2(
        __riscv_vor_vx_u32m2(__riscv_vand_vx_u32m2(qb, 0x807FFFFFu, vl), (uint32_t)(127u << 23), vl));

    vfloat32m2_t r = __riscv_vfsub_vf_f32m2(m, 1.0f, vl);
    vfloat32m2_t u = __riscv_vfsub_vf_f32m2(q, 0.5f, vl);

    // correction = C1*u + C0; log_ratio = (L2*r + L1)*r + L0 (production
    // sfpu_mad order, each step one fused MAD here).
    vfloat32m2_t corr = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(kC0, vl), kC1, u, vl);
    vfloat32m2_t lr = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(kL1, vl), kL2, r, vl);
    lr = __riscv_vfmadd_vv_f32m2(lr, r, __riscv_vfmv_v_f_f32m2(kL0, vl), vl);
    vfloat32m2_t rlr = __riscv_vfmul_vv_f32m2(r, lr, vl);

    // log_expand: log_q = e*ln2 + r*L(r). The int->float convert is RNE and
    // exact for every reachable exponent, same as production int32_to_float.
    vfloat32m2_t e_f = __riscv_vfcvt_f_x_v_f32m2(e_int, vl);
    vfloat32m2_t log_q = __riscv_vfmacc_vf_f32m2(rlr, kLn2, e_f, vl);

    // log(1-q) = u*correction - ln2 (one fused MAD, the shared-ln2 root pin).
    vfloat32m2_t l1mq = __riscv_vfmadd_vv_f32m2(u, corr, __riscv_vfmv_v_f_f32m2(-kLn2, vl), vl);

    vfloat32m2_t res = __riscv_vfsub_vv_f32m2(l1mq, log_q, vl);
    // Lower-half sign restore (production v_if(x < 0.5) result = -result).
    vfloat32m2_t neg = __riscv_vfsgnjn_vv_f32m2(res, res, vl);
    res = __riscv_vmerge_vvm_f32m2(res, neg, m_low, vl);

    // Exterior closes, production order (each a masked overwrite; NaN inputs
    // fail every ordered compare and flow through the log arithmetic exactly
    // like the SFPU lanes).
    res = __riscv_vfmerge_vfm_f32m2(res, __builtin_inff(), __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl), vl);
    res = __riscv_vfmerge_vfm_f32m2(res, __builtin_inff(), __riscv_vmfge_vf_f32m2_b16(x, 1.0f, vl), vl);
    vbool16_t m_expzero = __riscv_vmseq_vx_u32m2_b16(
        __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x), 0x7F800000u, vl), 0u, vl);
    res = __riscv_vfmerge_vfm_f32m2(res, -__builtin_inff(), m_expzero, vl);
    return res;
}

static inline void rvk_eval_tile_logodds(const float* x, float* yout) {
    size_t vl = __riscv_vsetvl_e32m2(8);
#if RVK_ILV == 1
    // Serial reference (A/B knob): one chunk per iteration, bit-exact.
    for (int c = 0; c < 128; c++) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, rvk_logodds_eval_elem(xA, vl), vl);
    }
#else
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        vfloat32m2_t yA = rvk_logodds_eval_elem(xA, vl);
        vfloat32m2_t yB = rvk_logodds_eval_elem(xB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, yB, vl);
    }
#endif  // RVK_ILV
}

#endif  // RVK_FORM_LOGODDS && TRISC_PACK
#endif  // RVV_FORMS_RVV_LOGODDS_H_
