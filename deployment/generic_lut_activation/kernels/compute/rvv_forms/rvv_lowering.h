// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// rvv_lowering.h — RVV (Zve32f, TRISC2 pack thread) implementations of the
// whole-function ALGEBRAIC LOWERINGS emitted by run_csv.sh codegen, for the
// RVV-only kernel piecewise_rvv.cpp.
// =============================================================================
//
// WHAT A LOWERING IS (see run_csv.sh "Whole-function algebraic collapses" and
// piecewise_generic.cpp sfpi::*_eval): the CSV metadata declares an
// eval_method template and the codegen PROVES it from the coefficient /
// boundary algebra, then emits TT_ACT_EVAL_KIND + a tiny macro payload and
// does NOT define EVAL_METHOD_POLY_CASCADE. The kernel bypasses segment
// selection and Horner entirely: the LUT is still embedded (standard layout,
// so piecewise_rvv.cpp's static_asserts hold) but is never read.
//
// LOWERINGS IMPLEMENTED HERE (exact production semantics from
// piecewise_generic.cpp, dispatch block at TT_ACT_EVAL_KIND):
//   kind 10 AFFINE_IDENTITY            y = x                       (identity)
//   kind 11 AFFINE_COLLAPSE            y = C1*x + C0               (affine)
//   kind 12 CLAMPED_AFFINE_COLLAPSE    y = min(max(C1*x+C0, MIN?), MAX?)
//                                      (relu, relu6, relu_min, relu_max,
//                                       hardtanh, hardsigmoid, threshold)
//   kind 16 ABS_VALUE                  y = |x|                     (abs)
//   kind 13 THRESHOLD_IDENTITY_SELECT  y = |x|<=LAMBDA ? 0 : x     (hardshrink)
//   kind 17 THRESHOLD_SOFTSHIFT_SELECT y = |x|>LAMBDA ? sign(x)*(|x|-LAMBDA)
//                                          : 0                     (softshrink)
//   kind 14 GATED_AFFINE_PRODUCT       y = x * clamp(Q1*x+Q0, 0, 1)
//                                      (hardswish, hardmish)
//   kind 18 SLOPE_MAX                  y = max/min(SHI*x, SLO*x)   (leaky_relu,
//                                      prelu — BH bf16 corpus; the fp32
//                                      frontier winners for these two are
//                                      plain poly_cascade, already handled by
//                                      the base kernel)
//
// NOT implemented here (stay behind the base kernel's #errors):
//   kind 15 ABS_DENOMINATOR_RATIONAL — rational-lane property (rvv_rational.h).
//   POSTCOMPOSE_AFFINE_Y / PRECOMPOSE_INPUT_AFFINE — never co-emitted with the
//   fp32 frontier winners of the 13 lowering ops; base #errors kept.
//   SLOPE_MAX under SLOPE_MAX_DISABLE (the C-LOW-2 A/B escape falls back to
//   the poly cascade in production; that fallback re-wire is NOT implemented
//   here, so the combination is a LOUD compile error, never silent wrong math).
//
// EVALUATION ORDER (documented per the RVV contract — gates are ULP-vs-golden,
// not SFPU byte identity):
//   * Affine steps use a single-rounding RVV FMA (vfmacc: C0 + C1*x) where the
//     production sfpi path issues one SFPMAD. Rounding may differ per-op; the
//     harness ULP report is the arbiter (identical stance to the base kernel's
//     Horner).
//   * Clamp order matches production exactly: max(lo, y) BEFORE min(y, hi)
//     (clamped_affine), and max(0,gate) BEFORE min(gate,1) (gated product).
//   * Select semantics match production branch-for-branch, including NaN
//     behaviour: threshold_identity keeps y = x when the |x| <= LAMBDA compare
//     FAILS (so NaN passes through, as in the SFPU v_if); threshold_softshift
//     keeps the y = 0 default when the |x| > LAMBDA compare fails (so NaN
//     maps to 0, as in the SFPU). softshift's sign step uses vfsgnj(t, x)
//     which equals the production "if (x < 0) y = -y" for every input that
//     can reach it (the branch requires |x| > LAMBDA >= 0, so x != +/-0
//     there; negative x gives -t either way).
//   * abs is a sign-bit clear (vfsgnjx x,x), bit-identical to sfpi setsgn(x,0).
//   * Both engines are DAZ+FTZ.
//
// This header is self-contained: preprocessor detection is visible to ALL
// three TUs; vector code stays strictly inside TRISC_PACK (Zve32f exists only
// on the pack RISC).
// =============================================================================

#ifndef RVV_FORMS_RVV_LOWERING_H_
#define RVV_FORMS_RVV_LOWERING_H_

// ---------------------------------------------------------------------------
// Lowering detection (preprocessor, all TUs). Exactly one kind may be active;
// codegen emits at most one algebraic macro block, and each recognizer above
// it in run_csv.sh suppresses the ones below (affine > clamped_affine >
// threshold_identity > abs_value > slope_max > threshold_softshift > gated).
// ---------------------------------------------------------------------------
#if defined(AFFINE_IDENTITY)
#define RVK_LOWERING_IDENTITY 1
#define RVK_LOWERING_ACTIVE 1
#elif defined(AFFINE_COLLAPSE)
#define RVK_LOWERING_AFFINE 1
#define RVK_LOWERING_ACTIVE 1
#elif defined(CLAMPED_AFFINE_COLLAPSE)
#define RVK_LOWERING_CLAMPED_AFFINE 1
#define RVK_LOWERING_ACTIVE 1
#elif defined(ABS_VALUE)
#define RVK_LOWERING_ABS_VALUE 1
#define RVK_LOWERING_ACTIVE 1
#elif defined(THRESHOLD_IDENTITY_SELECT)
#define RVK_LOWERING_THRESHOLD_IDENTITY 1
#define RVK_LOWERING_ACTIVE 1
#elif defined(THRESHOLD_SOFTSHIFT_SELECT)
#define RVK_LOWERING_THRESHOLD_SOFTSHIFT 1
#define RVK_LOWERING_ACTIVE 1
#elif defined(GATED_AFFINE_PRODUCT) || defined(GATED_QUADRATIC_COLLAPSE)
#define RVK_LOWERING_GATED_AFFINE_PRODUCT 1
#define RVK_LOWERING_ACTIVE 1
#elif defined(SLOPE_MAX)
// Production gates slope_max on SLOPE_MAX_ELIGIBLE (Blackhole && !DISABLE) and
// falls back to the poly cascade otherwise. The RVV kernel is BH-only, so the
// arch half is a given; the DISABLE escape's cascade fallback is NOT wired
// here — refuse loudly rather than silently diverge from the A/B contract.
#if defined(SLOPE_MAX_DISABLE) || defined(ARCH_WORMHOLE)
#error \
    "rvv_lowering: slope_max is disabled/ineligible here and the poly-cascade fallback is not implemented in the RVV kernel"
#endif
#define RVK_LOWERING_SLOPE_MAX 1
#define RVK_LOWERING_ACTIVE 1
#else
#define RVK_LOWERING_ACTIVE 0
#endif

#if RVK_LOWERING_ACTIVE
// Payload sanity: a lowering with a postcompose/precompose wrapper is not a
// combination this implementation reproduces (production applies
// apply_output_postcompose inside every lowering eval). The base kernel
// already #errors on these macros; assert again here so this header stays
// safe even if those base guards are ever relaxed for another form.
#if defined(POSTCOMPOSE_AFFINE_Y) || defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT) || defined(PRECOMPOSE_INPUT_AFFINE)
#error "rvv_lowering: pre/postcompose wrappers are not implemented for lowered forms"
#endif
// bf16 is supported THROUGH the rvv_bw_io.h I/O layer only: piecewise_rvv.cpp
// stages the bf16 CB page in with an exact widen, runs this lowering evaluator
// unchanged in fp32 (payload constants are the same fp32 macros in both
// precisions; every compare/select sees the exact widened values production's
// fp32 registers see), and repacks with the bit-exact RNE convert contract.
// Keep a loud guard against a bf16 build that lacks that layer.
#if defined(USE_BF16) && !defined(RVV_FORMS_RVV_BW_IO_H_)
#error "rvv_lowering: bf16 requires the rvv_bw_io.h stage-in/repack layer (include rvv_forms/rvv_bw_io.h first)"
#endif
#endif  // RVK_LOWERING_ACTIVE

// ---------------------------------------------------------------------------
// RVV evaluator (pack thread only).
// ---------------------------------------------------------------------------
#if RVK_LOWERING_ACTIVE && defined(TRISC_PACK)

#include <riscv_vector.h>

// Loop-invariant vector constants, declared once per tile call (conditionally-
// DEFINED macros, base-kernel style — never #if inside a macro body).
#if defined(RVK_LOWERING_AFFINE)
#define RVK_LOW_DECL_CONSTS() vfloat32m2_t vlc0_ = __riscv_vfmv_v_f_f32m2(AFFINE_C0, vl)
#elif defined(RVK_LOWERING_CLAMPED_AFFINE)
#define RVK_LOW_DECL_CONSTS() vfloat32m2_t vlc0_ = __riscv_vfmv_v_f_f32m2(CLAMPED_AFFINE_C0, vl)
#elif defined(RVK_LOWERING_GATED_AFFINE_PRODUCT)
#define RVK_LOW_DECL_CONSTS() vfloat32m2_t vlq0_ = __riscv_vfmv_v_f_f32m2(GATED_QUADRATIC_Q0, vl)
#elif defined(RVK_LOWERING_THRESHOLD_SOFTSHIFT)
#define RVK_LOW_DECL_CONSTS() vfloat32m2_t vlzero_ = __riscv_vfmv_v_f_f32m2(0.0f, vl)
#else
#define RVK_LOW_DECL_CONSTS() (void)0
#endif

// Optional clamp arms of clamped_affine (either bound may be absent).
#if defined(CLAMPED_AFFINE_HAS_MIN)
#define RVK_LOW_CA_MIN(y) y = __riscv_vfmax_vf_f32m2(y, CLAMPED_AFFINE_MIN, vl)
#else
#define RVK_LOW_CA_MIN(y) (void)0
#endif
#if defined(CLAMPED_AFFINE_HAS_MAX)
#define RVK_LOW_CA_MAX(y) y = __riscv_vfmin_vf_f32m2(y, CLAMPED_AFFINE_MAX, vl)
#else
#define RVK_LOW_CA_MAX(y) (void)0
#endif

// slope_max combine direction (production SLOPE_MAX_IS_MAX).
#if defined(SLOPE_MAX_IS_MAX)
#define RVK_LOW_SM_COMBINE(y, a, b) y = __riscv_vfmax_vv_f32m2(a, b, vl)
#else
#define RVK_LOW_SM_COMBINE(y, a, b) y = __riscv_vfmin_vv_f32m2(a, b, vl)
#endif

// One 8-element lane: y = lowering(x). References the RVK_LOW_DECL_CONSTS()
// invariants from the enclosing scope. All single-rounding FMAs are
// deliberate (see header comment on evaluation order).
#if defined(RVK_LOWERING_IDENTITY)
// identity: pure copy. (Production skips the SFPU body entirely.)
#define RVK_LOW_EVAL(y, x) vfloat32m2_t y = x

#elif defined(RVK_LOWERING_AFFINE)
// affine: y = C0 + C1*x (production: one SFPMAD).
#define RVK_LOW_EVAL(y, x) vfloat32m2_t y = __riscv_vfmacc_vf_f32m2(vlc0_, AFFINE_C1, x, vl)

#elif defined(RVK_LOWERING_CLAMPED_AFFINE)
// clamped_affine: y = C0 + C1*x; y = max(MIN, y) [if MIN]; y = min(y, MAX)
// [if MAX] — production order (vec_min_max lower arm first).
#define RVK_LOW_EVAL(y, x)                                                     \
    vfloat32m2_t y = __riscv_vfmacc_vf_f32m2(vlc0_, CLAMPED_AFFINE_C1, x, vl); \
    RVK_LOW_CA_MIN(y);                                                         \
    RVK_LOW_CA_MAX(y)

#elif defined(RVK_LOWERING_ABS_VALUE)
// abs: sign-bit clear, bit-identical to sfpi setsgn(x, 0).
#define RVK_LOW_EVAL(y, x) vfloat32m2_t y = __riscv_vfsgnjx_vv_f32m2(x, x, vl)

#elif defined(RVK_LOWERING_THRESHOLD_IDENTITY)
// threshold_identity: y = x, then y = 0 where |x| <= LAMBDA (equality belongs
// to zero; a failed compare — NaN — keeps x, exactly like the SFPU v_if).
#define RVK_LOW_EVAL(y, x)                                                                \
    vfloat32m2_t y##_ax = __riscv_vfsgnjx_vv_f32m2(x, x, vl);                             \
    vbool16_t y##_mz = __riscv_vmfle_vf_f32m2_b16(y##_ax, THRESHOLD_IDENTITY_LAMBDA, vl); \
    vfloat32m2_t y = __riscv_vfmerge_vfm_f32m2(x, 0.0f, y##_mz, vl)

#elif defined(RVK_LOWERING_THRESHOLD_SOFTSHIFT)
// threshold_softshift: y = 0 default; where |x| > LAMBDA, y = sign(x)*(|x|-LAMBDA).
// A failed compare — NaN — keeps the 0 default, exactly like the SFPU.
#define RVK_LOW_EVAL(y, x)                                                                 \
    vfloat32m2_t y##_ax = __riscv_vfsgnjx_vv_f32m2(x, x, vl);                              \
    vbool16_t y##_mg = __riscv_vmfgt_vf_f32m2_b16(y##_ax, THRESHOLD_SOFTSHIFT_LAMBDA, vl); \
    vfloat32m2_t y##_t = __riscv_vfsub_vf_f32m2(y##_ax, THRESHOLD_SOFTSHIFT_LAMBDA, vl);   \
    y##_t = __riscv_vfsgnj_vv_f32m2(y##_t, x, vl);                                         \
    vfloat32m2_t y = __riscv_vmerge_vvm_f32m2(vlzero_, y##_t, y##_mg, vl)

#elif defined(RVK_LOWERING_GATED_AFFINE_PRODUCT)
// gated_affine_product: gate = Q0 + Q1*x; gate = max(0, gate); gate =
// min(gate, 1); y = x * gate — production order exactly.
#define RVK_LOW_EVAL(y, x)                                                          \
    vfloat32m2_t y##_g = __riscv_vfmacc_vf_f32m2(vlq0_, GATED_QUADRATIC_Q1, x, vl); \
    y##_g = __riscv_vfmax_vf_f32m2(y##_g, 0.0f, vl);                                \
    y##_g = __riscv_vfmin_vf_f32m2(y##_g, 1.0f, vl);                                \
    vfloat32m2_t y = __riscv_vfmul_vv_f32m2(x, y##_g, vl)

#elif defined(RVK_LOWERING_SLOPE_MAX)
// slope_max: y = max/min(SHI*x, SLO*x) — the two-slope kink at 0.
#define RVK_LOW_EVAL(y, x)                                                   \
    vfloat32m2_t y##_hi = __riscv_vfmul_vf_f32m2(x, SLOPE_MAX_SLOPE_HI, vl); \
    vfloat32m2_t y##_lo = __riscv_vfmul_vf_f32m2(x, SLOPE_MAX_SLOPE_LO, vl); \
    vfloat32m2_t y;                                                          \
    RVK_LOW_SM_COMBINE(y, y##_hi, y##_lo)
#endif

// ---- whole-tile evaluator: 1024 fp32 elements, e32m2 (vl = 8), 128 chunks,
// 2-way interleave. No gathers, no tables — every kind is a handful of
// element-wise vector ops with a 1-2 op dependency chain, so two independent
// streams already cover the FP latency; register pressure is tiny (<= 8 m2
// regs live). Called by piecewise_rvv.cpp INSTEAD of the uniform/generic
// cascade evaluators when RVK_LOWERING_ACTIVE.
static inline void rvk_eval_tile_lowering(const float* x, float* yout) {
    size_t vl = __riscv_vsetvl_e32m2(8);
    RVK_LOW_DECL_CONSTS();
#if RVK_ILV == 1
    // Serial reference (A/B knob in piecewise_rvv.cpp): one chunk per
    // iteration, bit-exact vs the 2-way default.
    for (int c = 0; c < 128; c++) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        RVK_LOW_EVAL(yA, xA);
        yA = tt_rvv_finalize_domain_actions(xA, yA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
    }
#else
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        RVK_LOW_EVAL(yA, xA);
        RVK_LOW_EVAL(yB, xB);
        yA = tt_rvv_finalize_domain_actions(xA, yA, vl);
        yB = tt_rvv_finalize_domain_actions(xB, yB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, yB, vl);
    }
#endif  // RVK_ILV
}

#endif  // RVK_LOWERING_ACTIVE && TRISC_PACK
#endif  // RVV_FORMS_RVV_LOWERING_H_
