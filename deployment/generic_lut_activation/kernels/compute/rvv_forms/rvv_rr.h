// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// RVV RANGE-REDUCTION FORMS — rvv_rr.h  (RR lane of the piecewise_rvv backend)
// =============================================================================
// Included by piecewise_rvv.cpp immediately after eval_method.h. Two jobs:
//
//   1. MODE TAGS (pure preprocessor, every TRISC): defines RVK_RR_ACTIVE when
//      the generated adhoc defines select a range-reduced / standalone form
//      this lane implements. piecewise_rvv.cpp keys its support-matrix #errors
//      and its per-tile dispatch off RVK_RR_ACTIVE.
//
//   2. EVALUATORS (TRISC_PACK only, Zve32f): a tile evaluator
//      rvk_rr_eval_tile(const float* x, float* y) — 1024 fp32 elements,
//      e32m2 (vl = 8), 128 chunks, 2-way A/B chunk pairing (independent
//      chains; the compiler schedules across them).
//
// SEMANTICS CONTRACT. Each mode reproduces the production SFPU evaluator in
// piecewise_generic.cpp (the fp32 dispatch path) step for step: same reduction
// algebra, same coefficient order, same special-value policy. The RVV engine
// rounds independently of the SFPU (vfmadd shares the SFPU's partially fused MAD (fma_bits_bh, RTL-confirmed
// 2026-08-24): single final rounding over a 27-bit+sticky product; TwoProd residuals are bounded (2^-3 ulp), not exact;
// the tier's accuracy comes from the double-float coefficients; vector converts use frm = RNE), so byte identity with
// the SFPU is NOT claimed — the harness ULP-vs-golden report is the accuracy arbiter. Every deliberate evaluation-order
// choice is documented at its site.
//
// MODES IMPLEMENTED (fp32 frontier winners of the 14-op RR lane):
//   EXPONENT_ALU_EXP2  (exp, exp2, sigmoid, silu, swish, cosh, sinh)
//       exp_hw_eval/exp_hw_eval_preloaded: optional |x| ingress
//       (EXP_HW_INPUT_ABS), xlog2 = x*MULT + EXP_HW_EXPONENT_BIAS FMA,
//       optional [0,255] clamp, exact float->fixed 9.23 split, degree-N Horner
//       on the fraction (EXP_HW_FUSED coeff fold honored), BARE_SETEXP or
//       p*2^i recombine, sigmoid / sigmoid-product / minus-one /
//       hyperbolic-even (cosh: y + SCALE/y) / hyperbolic-odd-factor
//       (sinh: max(y - SCALE/y, origin cubic), signed) compose.
//   EXPONENT_ALU_LOG2  (log, log2, log10)
//       log_hw_eval*: exponent extract + mantissa normalize to [1,2),
//       m_minus_1 basis, Horner, (e + h) * LOG_HW_SCALE, optional specials.
//   EXPONENT_ALU_LOG1P (log1p)
//       log1p_hw_eval*: Juffa exact-reconstruction reduction on the
//       [0.75, 1.5) window, r + r^2*P(r), + k*ln2, optional specials.
//   EVAL_METHOD_NEWTON_ROOT (sqrt, rsqrt, cbrt/cbrt_magic)
//       newton_root_sqrt / newton_root_rsqrt / newton_root_cbrt_magic
//       (fp32 branches): magic seed + the exact production Newton/Householder
//       correction structure, inf/NaN/zero special policy included.
//   EVAL_METHOD_TRIG_RESIDUAL (sin, cos)
//       trig_residual_reduce + trig_residual_odd_eval: four-part Cody-Waite
//       pi (sine) / pi-2 (cosine) reduction, quadrant sign via the biased
//       round-magic LSB, odd polynomial in s = a^2 times a.
//   EVAL_METHOD_REDUCED_POLY + REDUCE_TAN (tan)
//       tan_reduce + the piecewise cascade on the REDUCED argument + tan_expand
//       (j odd -> -1/poly via the Newton reciprocal below).
//
// MODES DELIBERATELY NOT IMPLEMENTED (piecewise_rvv.cpp keeps its #error):
//   REDUCE_EXP / REDUCE_TRIG / REDUCE_LOG / REDUCE_CBRT reduced-poly cascades
//   (no fp32 frontier winner among the lane's 14 ops uses them — the exp/log
//   families win on the exponent-ALU standalones, sin/cos on trig_residual),
//   EXPONENT_ALU_POW (sqrt/rsqrt win on newton_root), EVAL_METHOD_TAN_STANDALONE
//   (the tan winner is the 2-segment REDUCE_TAN cascade), EVAL_METHOD_ASIN_ACOS,
//   and the division-free Householder cbrt (newton_root N==3 without the
//   cbrt_magic algorithm tag — no winner emits it).
// =============================================================================

#pragma once

// ---------------------------------------------------------------------------
// 1. MODE TAGS (all TRISCs; preprocessor only)
// ---------------------------------------------------------------------------
#if defined(EVAL_METHOD_EXPONENT_ALU) && defined(EXPONENT_ALU_EXP2)
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_EXP2 1
#elif defined(EVAL_METHOD_EXPONENT_ALU) && defined(EXPONENT_ALU_LOG2)
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_LOG2 1
#elif defined(EVAL_METHOD_EXPONENT_ALU) && defined(EXPONENT_ALU_LOG1P)
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_LOG1P 1
#elif defined(EVAL_METHOD_REDUCED_POLY) && defined(RANGE_REDUCTION_EXP) && \
    (defined(REDUCE_EXP_COMPOSE_ELU) || defined(REDUCE_EXP_COMPOSE_SELU))
// elu/celu/selu: the reduce_exp COMPOSE cascade (piecewise_generic.cpp's
// RANGE_REDUCTION_EXP + REDUCE_EXP_COMPOSE_ELU/_SELU). Only the elu/selu
// composes have an fp32 frontier winner on this lane; every OTHER
// RANGE_REDUCTION_EXP form (plain natural-exp, cosh/sinh/silu — which win on
// the exp2-ALU standalone instead) stays refused by piecewise_rvv.cpp's
// range-reduction #error, exactly as before.
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_REDUCE_EXP_ELU 1
#elif defined(EVAL_METHOD_REDUCED_POLY) && defined(RANGE_REDUCTION_RECIP_COMPLEMENT)
// atan: the signed-abs reciprocal/complement reduction (piecewise_generic.cpp's
// RANGE_REDUCTION_RECIP_COMPLEMENT). One even core ratio serves the direct leaf
// and the reciprocal-folded tail; the reduction owns its own +-pi/2 shoulders.
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_RECIP_COMPLEMENT 1
#elif defined(EVAL_METHOD_NEWTON_ROOT)
#if (NEWTON_ROOT_N == 3) && !defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)
#error "rvv_rr: newton_root cbrt without the cbrt_magic algorithm is not implemented"
#endif
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_NEWTON 1
#elif defined(EVAL_METHOD_TRIG_RESIDUAL)
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_TRIG 1
#elif defined(EVAL_METHOD_REDUCED_POLY) && defined(REDUCE_TAN) && !defined(TT_ACT_RATIONAL_LUT)
// A rational LUT with REDUCE_TAN (bf16 tan winner tan_n*d*_rational) is the
// RATIONAL lane's cascade: piecewise_rvv.cpp routes it to
// rvv_forms/rvv_rational.h (marker TT_ACT_RATIONAL_LUT from the rational
// adhoc template). This poly tan-cascade mode must NOT claim it.
#define RVK_RR_ACTIVE 1
#define RVK_RR_MODE_TAN_CASCADE 1
#endif

// The production standalone paths bypass the basis epilogues entirely, and the
// REDUCE_TAN winner emits none — refuse any combination this lane has not
// reproduced rather than silently dropping an epilogue step. POLY_PARITY_* is
// supported on the tan cascade only (the production cascade's stride-2
// x^2-Horner is reproduced there); the standalone modes never see it.
#if defined(RVK_RR_ACTIVE) && (defined(BASIS_INPUT_ABS_X) || defined(BASIS_MUL_ABS_X_BEFORE_POST) ||                   \
                               defined(BASIS_AFFINE_EVEN) || defined(BASIS_CLAMP_MAX) || defined(BASIS_POST_SIGN_X) || \
                               defined(BASIS_LEFT_TAIL_ZERO) || defined(BASIS_RIGHT_TAIL_IDENTITY))
#error "rvv_rr: basis modifiers are not supported on range-reduced forms"
#endif
#if defined(RVK_RR_ACTIVE) && !defined(RVK_RR_MODE_TAN_CASCADE) && \
    (defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN))
#error "rvv_rr: polynomial parity is only supported on the REDUCE_TAN cascade"
#endif

// ---------------------------------------------------------------------------
// 2. EVALUATORS (pack TRISC only — Zve32f exists only on TRISC2)
// ---------------------------------------------------------------------------
#if defined(RVK_RR_ACTIVE) && defined(TRISC_PACK)

#include <riscv_vector.h>

// ---- bit-view helpers (zero-cost reinterprets) ------------------------------
static inline vuint32m2_t rvk_rr_ub(vfloat32m2_t v) { return __riscv_vreinterpret_v_f32m2_u32m2(v); }
static inline vint32m2_t rvk_rr_ib(vfloat32m2_t v) { return __riscv_vreinterpret_v_f32m2_i32m2(v); }
static inline vfloat32m2_t rvk_rr_fu(vuint32m2_t v) { return __riscv_vreinterpret_v_u32m2_f32m2(v); }
static inline vfloat32m2_t rvk_rr_fi(vint32m2_t v) { return __riscv_vreinterpret_v_i32m2_f32m2(v); }

#if defined(RVK_RR_MODE_TAN_CASCADE) || defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) ||                       \
    defined(RVK_RR_MODE_RECIP_COMPLEMENT) || defined(LOG1P_COMPOSE_ATANH)
// ---- Newton reciprocal (no vfdiv on Zve32f; no SFPARECIP estimate here) -----
// Production: ckernel_sfpu_recip.h sfpu_reciprocal_iter<N> = hardware ~8-bit
// SFPARECIP seed + N Newton steps (fp32 compose uses N=2, tan_expand N=3;
// ~28-bit / converged). RVV has no hardware estimate, so the seed is the
// classic integer-magic reciprocal (0x7EF311C3 - bits(|d|), max rel err ~5e-2)
// and THREE fused Newton steps (5e-2 -> 2.5e-3 -> 6e-6 -> 4e-11): at least as
// converged as the production iteration counts. EVALUATION ORDER differs from
// the SFPU (seed source + fused r=1-d*y / y+=y*r steps vs the SFPU's negated-t
// form); the ULP gate is the arbiter. Special-value policy matches production
// approx_recip + its NaN-guarded Newton: |d| >= 2^126 -> signed 0 (covers
// +/-inf), d == +/-0 -> signed inf, NaN propagates.
static inline vfloat32m2_t rvk_rr_recip(vfloat32m2_t d, size_t vl) {
    vfloat32m2_t ad = __riscv_vfsgnjx_vv_f32m2(d, d, vl);  // |d|
    vfloat32m2_t y = rvk_rr_fu(__riscv_vrsub_vx_u32m2(rvk_rr_ub(ad), 0x7EF311C3u, vl));
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
#pragma GCC unroll 3
    for (int i = 0; i < 3; i++) {
        vfloat32m2_t r = __riscv_vfnmsac_vv_f32m2(one, ad, y, vl);  // 1 - ad*y   (fused)
        y = __riscv_vfmacc_vv_f32m2(y, y, r, vl);                   // y + y*r    (fused)
    }
    vbool16_t mbig = __riscv_vmfge_vf_f32m2_b16(ad, 0x1p126f, vl);
    vbool16_t mzero = __riscv_vmfeq_vf_f32m2_b16(ad, 0.0f, vl);
    y = __riscv_vfmerge_vfm_f32m2(y, 0.0f, mbig, vl);
    y = __riscv_vfmerge_vfm_f32m2(y, __builtin_inff(), mzero, vl);
    return __riscv_vfsgnj_vv_f32m2(y, d, vl);  // reciprocal keeps the input sign
}
#endif

#if defined(LOG1P_COMPOSE_ASINH)
// ---- Newton sqrt (no vfsqrt on Zve32f) --------------------------------------
// Production asinh's large zone calls ckernel_sfpu_sqrt_custom (self-contained,
// no Prgm init). RVV has no hardware sqrt/rsqrt estimate, so this uses the
// classic integer-magic inverse-sqrt seed (0x5f3759df, max rel err ~3.5e-2) and
// THREE fused Newton steps on the rsqrt recurrence y*(1.5 - 0.5*v*y*y)
// (3.5e-2 -> 1.8e-3 -> 4.9e-6 -> 3.6e-11), then sqrt(v) = v*rsqrt(v) — the
// residual is far below an fp32 ULP. It is only ever evaluated on v = x^2 + 1
// >= 1 (the large asinh zone); the |x|<threshold small zone and the |x|>SAFE
// overflow tail both discard this value. EVALUATION ORDER differs from the
// SFPU custom sqrt; the harness ULP gate is the arbiter.
static inline vfloat32m2_t rvk_rr_sqrt(vfloat32m2_t v, size_t vl) {
    vfloat32m2_t half_v = __riscv_vfmul_vf_f32m2(v, 0.5f, vl);
    vfloat32m2_t y = rvk_rr_fu(__riscv_vrsub_vx_u32m2(__riscv_vsrl_vx_u32m2(rvk_rr_ub(v), 1, vl), 0x5F3759DFu, vl));
    vfloat32m2_t c15 = __riscv_vfmv_v_f_f32m2(1.5f, vl);
#pragma GCC unroll 3
    for (int i = 0; i < 3; i++) {
        vfloat32m2_t yy = __riscv_vfmul_vv_f32m2(y, y, vl);
        vfloat32m2_t t = __riscv_vfnmsac_vv_f32m2(c15, half_v, yy, vl);  // 1.5 - half_v*yy
        y = __riscv_vfmul_vv_f32m2(y, t, vl);
    }
    return __riscv_vfmul_vv_f32m2(v, y, vl);  // v * rsqrt(v) = sqrt(v)
}
#endif

// ============================================================================
// EXPONENT_ALU_EXP2 — production reference: exp_hw_eval / exp_hw_eval_preloaded
// (piecewise_generic.cpp). The fp32 dispatch runs the HW_PRELOAD variant; the
// two are documented byte-identical, and where they spell one step two ways
// (recombine) this file follows the PRELOADED spelling and says so.
// ============================================================================
#if defined(RVK_RR_MODE_EXP2)

#ifndef EXP_HW_MULT
#define EXP_HW_MULT 1.4426950216293334961f  // production default: log2(e)
#endif
#ifndef EXP_HW_EXPONENT_BIAS
#define EXP_HW_EXPONENT_BIAS 127  // production default (exp_hw_eval: x*MULT + 127.0f)
#endif

// Compose support matrix: this mode reproduces the production composes listed
// below; any OTHER codegen-emitted EXP_HW_COMPOSE_* has no evaluator here and
// must refuse at compile time — never a silent plain-exp2 fallback (that
// exact silent-drop shipped the cosh/sinh hyperbolic gap this guard closes).
#if defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) || defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT)
#error "rvv_rr: this EXP_HW_COMPOSE_* post-transform is not implemented on the RVV exp2 lane"
#endif
// Hyperbolic composes consume the ORIGINAL x after an |x| ingress; they are
// only certified with the abs ingress (the codegen emits them together).
#if (defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR)) && \
    !defined(EXP_HW_INPUT_ABS)
#error "rvv_rr: hyperbolic exp2 composes require the EXP_HW_INPUT_ABS ingress"
#endif

// EXP_HW_FUSED coefficient fold: coeff at power k pre-scaled by 2^-23k so the
// Horner runs on the RAW 9.23 fraction (production exp_hw_fused_scale;
// constexpr float per-op arithmetic — identical fold).
constexpr float rvk_rr_exp_c(uint32_t k) {
#if defined(EXP_HW_FUSED)
    float s = 1.0f;
    for (uint32_t i = 0; i < k; i++) {
        s *= 0x1p-23f;
    }
    return EXP_HW_COEFFS[k] * s;
#else
    return EXP_HW_COEFFS[k];
#endif
}

static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
#if defined(EXP_HW_INPUT_ABS)
    // |x| ingress (production setsgn(x, 0)); the ORIGINAL x stays live for the
    // hyperbolic-odd sign/origin-cubic epilogue below.
    [[maybe_unused]] vfloat32m2_t x_orig = x;
    x = __riscv_vfsgnjx_vv_f32m2(x, x, vl);
#endif
    // xlog2 = x*MULT + BIAS — single-rounding FMA, same as the production
    // sfpu_mad (bias 127 for the plain exp winners; 126 for the hyperbolic
    // composes, so y = exp(|x|)/2).
    vfloat32m2_t xlog2 =
        __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2((float)(EXP_HW_EXPONENT_BIAS), vl), (float)(EXP_HW_MULT), x, vl);
#if !defined(EXP_HW_SKIP_INPUT_CLAMP)
    // Full-range safety clamp to [0, 255] (production vec_min_max pair).
    xlog2 = __riscv_vfmax_vf_f32m2(xlog2, 0.0f, vl);
    xlog2 = __riscv_vfmin_vf_f32m2(xlog2, 255.0f, vl);
#endif
    // Fixed-point 9.23 split. Production shifts the implicit-one mantissa by the
    // debiased exponent: for xlog2 in [0, 255] that is EXACTLY trunc(xlog2 * 2^23)
    // (the pow2 scale is exact in fp32; the value < 2^31 so rtz convert is exact),
    // so this is bit-identical, not merely equivalent. Vector rtz only — the
    // scalar fcvt is RNE-locked on this core.
    vint32m2_t zb = __riscv_vfcvt_rtz_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(xlog2, 0x1p23f, vl), vl);
    vuint32m2_t zu = __riscv_vreinterpret_v_i32m2_u32m2(zb);
    vuint32m2_t ep = __riscv_vsrl_vx_u32m2(zu, 23, vl);         // biased int-part exponent (i+127)
    vuint32m2_t fm = __riscv_vand_vx_u32m2(zu, 0x7FFFFFu, vl);  // fraction * 2^23

    // Fraction for the 2^f Horner (convert is exact: fm < 2^23).
    vfloat32m2_t f = __riscv_vfcvt_f_xu_v_f32m2(fm, vl);
#if !defined(EXP_HW_FUSED)
    f = __riscv_vfmul_vf_f32m2(f, 0x1p-23f, vl);
#endif

    // Degree-N Horner, high-to-low, fused — production coefficient order.
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(rvk_rr_exp_c(EXP_HW_DEGREE), vl);
#pragma GCC unroll 17
    for (int k = (int)EXP_HW_DEGREE - 1; k >= 0; k--) {
        p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(rvk_rr_exp_c((uint32_t)k), vl), vl);
    }

#if defined(EXP_HW_BARE_SETEXP)
    // BARE recombine: p in [1,2) by construction (c0 >= 1) — replace p's
    // exponent field with ep, keep sign+mantissa (production SFPSETEXP).
    vfloat32m2_t y = rvk_rr_fu(__riscv_vor_vv_u32m2(
        __riscv_vand_vx_u32m2(rvk_rr_ub(p), 0x807FFFFFu, vl), __riscv_vsll_vx_u32m2(ep, 23, vl), vl));
#else
    // Non-bare recombine, PRELOADED spelling: y = p * 2^i with 2^i synthesized
    // as bits(ep << 23) (exact pow2 product; production documents this
    // bit-identical to the pe-corrected setexp for every in-range input).
    vfloat32m2_t y = __riscv_vfmul_vv_f32m2(p, rvk_rr_fu(__riscv_vsll_vx_u32m2(ep, 23, vl)), vl);
#endif

#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    // y == exp(-x); sigmoid = 1/(1 + exp(-x)) (production fp32: reciprocal_iter<2>).
    y = rvk_rr_recip(__riscv_vfadd_vf_f32m2(y, 1.0f, vl), vl);
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    y = __riscv_vfmul_vv_f32m2(x, y, vl);  // silu/swish: x * sigmoid(x)
#endif
#elif defined(EXP_HW_COMPOSE_MINUS_ONE)
    y = __riscv_vfsub_vf_f32m2(y, 1.0f, vl);  // expm1
#elif defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN)
    // cosh: y == exp(|x|)/2 (bias 126). Production: inv = reciprocal(y);
    // y = SCALE*inv + y (one sfpu_mad; SCALE ~ 1/4, fitted). DOCUMENTED
    // DIVERGENCE: production bf16 runs sfpu_reciprocal_iter<1> and the fitted
    // SCALE absorbs that lane's ~2^-14 bias; rvk_rr_recip is ~2^-34-converged,
    // so the absorbed bias reappears as <= ~2^-16 relative on the 1/y term —
    // far below a bf16 ULP even at x=0 where the term is half the result.
    // The exhaustive host proof is the arbiter (base-kernel stance).
    vfloat32m2_t inv = rvk_rr_recip(y, vl);
    y = __riscv_vfmacc_vf_f32m2(y, (float)(EXP_HW_COMPOSE_SCALE), inv, vl);
#elif defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR)
    // sinh: y == exp(|x|)/2. Production: magnitude = y - SCALE*reciprocal(y);
    // lower = |x|*(1 + ORIGIN_CUBIC*|x|^2) (the cancellation-safe origin
    // cubic); magnitude = max(magnitude, lower); y = copysgn(magnitude, x).
    // (Reference semantics: ttpoly/precision/rangered.py hyperbolic_odd_factor
    // — np.maximum + copysign; the same reciprocal-tier divergence note as the
    // even compose above applies.)
    vfloat32m2_t inv = rvk_rr_recip(y, vl);
    vfloat32m2_t mag = __riscv_vfnmsac_vf_f32m2(y, (float)(EXP_HW_COMPOSE_SCALE), inv, vl);
    vfloat32m2_t sq = __riscv_vfmul_vv_f32m2(x, x, vl);  // x == |x_orig| here
    vfloat32m2_t ofac = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), (float)(EXP_HW_ORIGIN_CUBIC), sq, vl);
    vfloat32m2_t lower = __riscv_vfmul_vv_f32m2(x, ofac, vl);
    mag = __riscv_vfmax_vv_f32m2(mag, lower, vl);
    y = __riscv_vfsgnj_vv_f32m2(mag, x_orig, vl);
#endif
    return y;
}
#endif  // RVK_RR_MODE_EXP2

// ============================================================================
// EXPONENT_ALU_LOG2 — production reference: log_hw_eval / log_hw_eval_preloaded.
// ============================================================================
#if defined(RVK_RR_MODE_LOG2)

#ifndef LOG_HW_SCALE
#define LOG_HW_SCALE 1.0f
#endif
#ifndef LOG_HW_INPUT_OFFSET
#define LOG_HW_INPUT_OFFSET 0.0f
#endif

static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
    constexpr float kOffset = (float)(LOG_HW_INPUT_OFFSET);
    vfloat32m2_t xd = x;
    if constexpr (kOffset != 0.0f) {
        xd = __riscv_vfadd_vf_f32m2(x, kOffset, vl);
    }
    vuint32m2_t xb = rvk_rr_ub(xd);
    // e = exponent field - 127 (production exexp NoDebias reads the field,
    // sign-blind); m = setexp(xd, 127) keeps sign + mantissa.
    vint32m2_t e_int = __riscv_vsub_vx_i32m2(
        __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(xb, 23, vl), 0xFFu, vl)),
        127,
        vl);
    vfloat32m2_t m = rvk_rr_fu(__riscv_vor_vx_u32m2(__riscv_vand_vx_u32m2(xb, 0x807FFFFFu, vl), 0x3F800000u, vl));

#ifdef LOG_HW_BASIS_M_MINUS_1
    vfloat32m2_t u = __riscv_vfsub_vf_f32m2(m, 1.0f, vl);
#else
    vfloat32m2_t u = m;
#endif
    // e as float. Signed vector convert (frm = RNE) is exact for |e| <= 254 —
    // value-identical to the production abs+copysgn / sign-magnitude convert.
    vfloat32m2_t e_float = __riscv_vfcvt_f_x_v_f32m2(e_int, vl);

    vfloat32m2_t h = __riscv_vfmv_v_f_f32m2(LOG_HW_COEFFS[LOG_HW_DEGREE], vl);
#pragma GCC unroll 17
    for (int k = (int)LOG_HW_DEGREE - 1; k >= 0; k--) {
        h = __riscv_vfmadd_vv_f32m2(h, u, __riscv_vfmv_v_f_f32m2(LOG_HW_COEFFS[k], vl), vl);
    }

    vfloat32m2_t result = __riscv_vfadd_vv_f32m2(e_float, h, vl);
    if constexpr ((float)(LOG_HW_SCALE) != 1.0f) {  // preloaded spelling: *1.0 skipped
        result = __riscv_vfmul_vf_f32m2(result, (float)(LOG_HW_SCALE), vl);
    }

#ifndef LOG_HW_SKIP_SPECIALS
    // log(neg) = NaN, log(0) = -inf on the DECOMPOSE input (production order).
    result = __riscv_vfmerge_vfm_f32m2(result, __builtin_nanf(""), __riscv_vmflt_vf_f32m2_b16(xd, 0.0f, vl), vl);
    result = __riscv_vfmerge_vfm_f32m2(result, -__builtin_inff(), __riscv_vmfeq_vf_f32m2_b16(xd, 0.0f, vl), vl);
#endif
    return result;
}
#endif  // RVK_RR_MODE_LOG2

// ============================================================================
// EXPONENT_ALU_LOG1P — production reference: log1p_hw_eval(_preloaded), Juffa
// exact-reconstruction reduction (no 1+x cancellation).
// ============================================================================
#if defined(RVK_RR_MODE_LOG1P)

#ifndef LOG1P_LN2
#define LOG1P_LN2 0.69314718055994530942f
#endif
// Production folds e_float * (LOG1P_LN2 * 0x1p-23f): float*float at compile
// time (exact pow2 scale) — same constant here.
constexpr float RVK_RR_LOG1P_LN2M23 = (float)(LOG1P_LN2) * 0x1p-23f;

// The shared Juffa log1p core, evaluated on the argument z. Plain log1p passes
// z == x; the atanh/asinh compose arms below precompose z from x_orig and wrap
// this with a postcompose, exactly like the SFPU log1p_hw_eval the compose arms
// call (piecewise_generic.cpp's LOG1P_COMPOSE_ATANH/_ASINH). Specials fire on
// the reduction input 1+z; for the composes z>=0 so they stay inert and the
// poles/overflow are owned by the precompose (den->0 => z->inf; a>SAFE tail).
static inline vfloat32m2_t rvk_rr_log1p_core(vfloat32m2_t z, size_t vl) {
    vfloat32m2_t u = __riscv_vfadd_vf_f32m2(z, 1.0f, vl);  // 1+z, exponent read only

    // Juffa exponent isolation on the [0.75, 1.5) window: e = k << 23.
    // bits(0.75f) = 0x3F400000; setman(...,0) == keep sign+exponent bits.
    vint32m2_t e = __riscv_vand_vx_i32m2(__riscv_vsub_vx_i32m2(rvk_rr_ib(u), 0x3F400000, vl), (int32_t)0xFF800000, vl);

    // Reduced argument reconstructed from the ORIGINAL z:
    //   2^-k * z  = bits(z) - e;   2^-k - 1 = -0.25 * (-4 * 2^-k) - 1  (exact)
    vfloat32m2_t two_neg_k_x = rvk_rr_fi(__riscv_vsub_vv_i32m2(rvk_rr_ib(z), e, vl));
    vfloat32m2_t s = rvk_rr_fi(__riscv_vrsub_vx_i32m2(e, (int32_t)0xC0800000, vl));              // bits(-4)-e
    vfloat32m2_t t = __riscv_vfmadd_vf_f32m2(s, -0.25f, __riscv_vfmv_v_f_f32m2(-1.0f, vl), vl);  // fused
    vfloat32m2_t r = __riscv_vfadd_vv_f32m2(two_neg_k_x, t, vl);                                 // in [-0.25, 0.5)

    // P(r) Horner (production coefficient order), then r + r^2*P(r).
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(LOG1P_HW_COEFFS[LOG1P_HW_DEGREE], vl);
#pragma GCC unroll 17
    for (int k = (int)LOG1P_HW_DEGREE - 1; k >= 0; k--) {
        p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(LOG1P_HW_COEFFS[k], vl), vl);
    }
    // ORDER NOTE: r*r is a rounded mul; the (r*r)*p + r step is a fused FMA
    // (the sfpi compiler fuses the production spelling the same way).
    vfloat32m2_t rr = __riscv_vfmul_vv_f32m2(r, r, vl);
    vfloat32m2_t result = __riscv_vfmadd_vv_f32m2(rr, p, r, vl);

    // + k*ln2. e = k<<23 has only 8 significant bits -> signed convert exact;
    // fused mad with the folded LN2*2^-23 (production spelling).
    vfloat32m2_t e_float = __riscv_vfcvt_f_x_v_f32m2(e, vl);
    result = __riscv_vfmacc_vf_f32m2(result, RVK_RR_LOG1P_LN2M23, e_float, vl);

#ifndef LOG1P_HW_SKIP_SPECIALS
    // log1p(x < -1) = NaN, log1p(-1) = -inf, on the reduction input 1+x.
    result = __riscv_vfmerge_vfm_f32m2(result, __builtin_nanf(""), __riscv_vmflt_vf_f32m2_b16(u, 0.0f, vl), vl);
    result = __riscv_vfmerge_vfm_f32m2(result, -__builtin_inff(), __riscv_vmfeq_vf_f32m2_b16(u, 0.0f, vl), vl);
#endif
    return result;
}

#if defined(LOG1P_COMPOSE_ATANH)
// atanh(x) = copysign(0.5 * log1p(2|x|/(1-|x|)), x), domain (-1,1). 1-|x| is
// exact by Sterbenz for |x| in [0.5,1); the poles |x|>=1 fold through den->0 =>
// z->+inf => log1p(inf)=inf, sign-restored. Production twin: piecewise_generic
// LOG1P_COMPOSE_ATANH / rangered._eval_log1p_compose atanh (2.79 ULP).
static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
    vfloat32m2_t a = __riscv_vfsgnjx_vv_f32m2(x, x, vl);      // |x|
    vfloat32m2_t den = __riscv_vfrsub_vf_f32m2(a, 1.0f, vl);  // 1 - |x|
    vfloat32m2_t z =
        __riscv_vfmul_vv_f32m2(__riscv_vfmul_vf_f32m2(a, 2.0f, vl), rvk_rr_recip(den, vl), vl);  // 2|x|/(1-|x|)
    vfloat32m2_t y = __riscv_vfmul_vf_f32m2(rvk_rr_log1p_core(z, vl), 0.5f, vl);
    return __riscv_vfsgnj_vv_f32m2(y, x, vl);  // copysign(y, x)
}
#elif defined(LOG1P_COMPOSE_ASINH)
// asinh(x): RECIPROCAL-FREE two-zone (twin of piecewise_generic.cpp's
// LOG1P_COMPOSE_ASINH arm; rangered._eval_log1p_compose asinh, ~1.6-1.9 ULP).
//   |x| <  ASINH_SMALL_THRESHOLD -> |x| * small_correction(x^2)  (no sqrt, exact at 0)
//   |x| >= threshold             -> log1p(|x| + (sqrt(x^2+1)-1))  (root-1 direct)
//   |x| >  sqrt(FP32_MAX)        -> ln(2|x|) = ln2 + log1p(|x|)   (x^2 would overflow)
static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
    constexpr float ASINH_SAFE = 1.8e19f;  // sqrt(FP32_MAX)
    constexpr float ASINH_LN2 = 0.69314718055994530942f;
    vfloat32m2_t a = __riscv_vfsgnjx_vv_f32m2(x, x, vl);  // |x|
    vfloat32m2_t s = __riscv_vfmul_vv_f32m2(a, a, vl);    // x^2
    // small zone: |x| * small_correction(x^2), natural-basis Horner in s.
    vfloat32m2_t ratio = __riscv_vfmv_v_f_f32m2(ASINH_SMALL_COEFFS[ASINH_SMALL_DEGREE], vl);
#pragma GCC unroll 16
    for (int i = (int)ASINH_SMALL_DEGREE - 1; i >= 0; i--) {
        ratio = __riscv_vfmadd_vv_f32m2(ratio, s, __riscv_vfmv_v_f_f32m2(ASINH_SMALL_COEFFS[i], vl), vl);
    }
    vfloat32m2_t small_val = __riscv_vfmul_vv_f32m2(a, ratio, vl);
    // large zone: log1p(|x| + (sqrt(x^2+1)-1)); root-1 direct (no reciprocal).
    vfloat32m2_t root = rvk_rr_sqrt(__riscv_vfadd_vf_f32m2(s, 1.0f, vl), vl);
    vfloat32m2_t z = __riscv_vfadd_vv_f32m2(a, __riscv_vfsub_vf_f32m2(root, 1.0f, vl), vl);
    vbool16_t big = __riscv_vmfgt_vf_f32m2_b16(a, ASINH_SAFE, vl);
    z = __riscv_vmerge_vvm_f32m2(z, a, big, vl);  // a>SAFE: avoid 2a overflow
    vfloat32m2_t l = rvk_rr_log1p_core(z, vl);
    l = __riscv_vmerge_vvm_f32m2(l, __riscv_vfadd_vf_f32m2(l, ASINH_LN2, vl), big, vl);  // a>SAFE: +ln2 tail
    vbool16_t sm = __riscv_vmflt_vf_f32m2_b16(a, (float)(ASINH_SMALL_THRESHOLD), vl);
    l = __riscv_vmerge_vvm_f32m2(l, small_val, sm, vl);  // |x|<thr -> small zone
    return __riscv_vfsgnj_vv_f32m2(l, x, vl);            // copysign(l, x)
}
#else
static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
    return rvk_rr_log1p_core(x, vl);  // plain log1p
}
#endif
#endif  // RVK_RR_MODE_LOG1P

// ============================================================================
// REDUCE_EXP COMPOSE (elu/celu/selu) — production reference: the
// RANGE_REDUCTION_EXP + REDUCE_EXP_COMPOSE_ELU/_SELU arm (piecewise_generic.cpp)
// and rangered._eval_exp_reduce_compose. The single-segment reduced leaf is the
// CANCELLATION-FREE expm1(r)/r core (NOT exp(r)); reconstruct
//   expm1(x) = (2^k - 1) + 2^k * (r * P(r))
// so the leading -1 is EXACT (2^k-1 exact for k<=0, the whole reconstruction
// band) and never cancels — a direct e^x-1 poly floors ~1500 fp32 ULP, this
// reaches ~1 ULP. Coefficients come from the embedded LUT (single segment).
// ============================================================================
#if defined(RVK_RR_MODE_REDUCE_EXP_ELU)

static_assert(NUM_SEGMENTS == 1, "rvv_rr: reduce_exp elu/selu core is single-segment (LUT coeff read)");
constexpr uint32_t RVK_RR_EXP_COEFF_OFFSET = NUM_SEGMENTS + 1;  // [lo,hi] then c0..cD

static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x_orig, size_t vl) {
    // Cody-Waite reduce: z = x/ln2; k = round(z) via the biased round magic;
    // r = x - k*ln2 (two-part NEG_LN2_HI/LO), all fused — production exp_reduce.
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float RND_MAGIC = 12582912.0f;  // 0x4B400000 (1.5*2^23)
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vfloat32m2_t zc = __riscv_vfmul_vf_f32m2(x_orig, INV_LN2, vl);
    vfloat32m2_t tmp = __riscv_vfadd_vf_f32m2(zc, RND_MAGIC, vl);
    vfloat32m2_t kf = __riscv_vfsub_vf_f32m2(tmp, RND_MAGIC, vl);              // round(z) as float
    vint32m2_t k_int = __riscv_vsub_vx_i32m2(rvk_rr_ib(tmp), 0x4B400000, vl);  // bits(tmp)-bits(magic)
    vfloat32m2_t r = __riscv_vfmadd_vf_f32m2(kf, NEG_LN2_HI, x_orig, vl);      // kf*HI + x
    r = __riscv_vfmadd_vf_f32m2(kf, NEG_LN2_LO, r, vl);                        // kf*LO + r

    // P(r) ~= expm1(r)/r — single-segment Horner over the embedded coeffs.
    const float* c = &LUT_DATA[RVK_RR_EXP_COEFF_OFFSET];
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(c[POLY_DEGREE], vl);
#pragma GCC unroll 17
    for (int k = (int)POLY_DEGREE - 1; k >= 0; k--) {
        p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(c[k], vl), vl);
    }

    // expm1(x) = (2^k - 1) + 2^k * (r*P(r)); 2^k == setexp(1.0, 127+k) with the
    // production exp_expand underflow guard (new_exp<=0 -> 0).
    vfloat32m2_t reduced_expm1 = __riscv_vfmul_vv_f32m2(p, r, vl);  // r*P(r) = expm1(r)
    vint32m2_t new_exp = __riscv_vadd_vx_i32m2(k_int, 127, vl);
    vfloat32m2_t two_to_k = rvk_rr_fi(__riscv_vsll_vx_i32m2(__riscv_vand_vx_i32m2(new_exp, 0xFF, vl), 23, vl));
    two_to_k = __riscv_vfmerge_vfm_f32m2(two_to_k, 0.0f, __riscv_vmsle_vx_i32m2_b16(new_exp, 0, vl), vl);
    vfloat32m2_t base = __riscv_vfsub_vf_f32m2(two_to_k, 1.0f, vl);                     // 2^k - 1
    vfloat32m2_t expm1_x = __riscv_vfmadd_vv_f32m2(two_to_k, reduced_expm1, base, vl);  // 2^k*rexpm1 + base

#if defined(REDUCE_EXP_COMPOSE_SELU)
    // selu(x) = lambda*x (x>=0) | lambda*alpha*expm1(x) (x<0); saturates to
    // -lambda*alpha at SELU_SAT_BOUND (disjoint from x>=0, so merge order is free).
    constexpr float SELU_LAMBDA = 1.0507010221481323f;
    constexpr float SELU_LAMBDA_ALPHA = 1.7580993175506592f;
    constexpr float SELU_SAT_BOUND = -16.869848251342773f;
    vfloat32m2_t y = __riscv_vfmul_vf_f32m2(expm1_x, SELU_LAMBDA_ALPHA, vl);
    y = __riscv_vmerge_vvm_f32m2(
        y, __riscv_vfmul_vf_f32m2(x_orig, SELU_LAMBDA, vl), __riscv_vmfge_vf_f32m2_b16(x_orig, 0.0f, vl), vl);
    y = __riscv_vfmerge_vfm_f32m2(y, -SELU_LAMBDA_ALPHA, __riscv_vmfle_vf_f32m2_b16(x_orig, SELU_SAT_BOUND, vl), vl);
    // Sign-concordance: sign(selu(x))==sign(x); force sign from x_orig so -0 and
    // sign-preserving-DAZ subnormals recover their sign (matches the SFPU arm).
    y = __riscv_vfsgnj_vv_f32m2(y, x_orig, vl);
    return y;
#else
    // elu/celu(alpha=1): elu(x) = x (x>=0) | expm1(x) (x<0); saturates to -1.
    constexpr float ELU_SAT_BOUND = -17.32868003845215f;
    vfloat32m2_t y = expm1_x;
    y = __riscv_vmerge_vvm_f32m2(y, x_orig, __riscv_vmfge_vf_f32m2_b16(x_orig, 0.0f, vl), vl);
    y = __riscv_vfmerge_vfm_f32m2(y, -1.0f, __riscv_vmfle_vf_f32m2_b16(x_orig, ELU_SAT_BOUND, vl), vl);
    // Sign-concordance: sign(elu(x))==sign(x); force sign from x_orig so -0 and
    // sign-preserving-DAZ subnormals recover their sign (matches the SFPU arm).
    y = __riscv_vfsgnj_vv_f32m2(y, x_orig, vl);
    return y;
#endif
}
#endif  // RVK_RR_MODE_REDUCE_EXP_ELU

// ============================================================================
// RECIP_COMPLEMENT (atan) — production reference: the
// RANGE_REDUCTION_RECIP_COMPLEMENT arm (piecewise_generic.cpp) and
// rangered._eval_reciprocal_complement. reduced = min(|x|, 1/|x|) in [0,1]; ONE
// even core P(s) with s = reduced^2 serves both leaves:
//   |x| <= 1: mag = reduced * P             (reduced = |x|)
//   |x| >  1: mag = pi/2 - reduced * P       (reduced = 1/|x|)
// As |x|->inf, reduced->0, P->1, mag->pi/2 EXACTLY — the reduction owns its own
// +-inf shoulders; atan is odd so copysign restores the input sign.
// ============================================================================
#if defined(RVK_RR_MODE_RECIP_COMPLEMENT)

static_assert(NUM_SEGMENTS == 1, "rvv_rr: reciprocal_complement core is single-segment (LUT coeff read)");
constexpr uint32_t RVK_RR_RC_COEFF_OFFSET = NUM_SEGMENTS + 1;  // [lo,hi] then c0..cD

static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
    constexpr float RC_PI_HALF = 1.5707963267948966f;                    // complement (atan)
    constexpr float RC_CORE_BOUND = 1.0f;                                // core_bound
    vfloat32m2_t a = __riscv_vfsgnjx_vv_f32m2(x, x, vl);                 // |x|
    vfloat32m2_t inv_a = rvk_rr_recip(a, vl);                            // 1/|x|
    vbool16_t big = __riscv_vmfgt_vf_f32m2_b16(a, RC_CORE_BOUND, vl);    // |x| > 1
    vfloat32m2_t reduced = __riscv_vmerge_vvm_f32m2(a, inv_a, big, vl);  // min(|x|, 1/|x|)
    vfloat32m2_t s = __riscv_vfmul_vv_f32m2(reduced, reduced, vl);       // s = reduced^2

    const float* c = &LUT_DATA[RVK_RR_RC_COEFF_OFFSET];
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(c[POLY_DEGREE], vl);
#pragma GCC unroll 17
    for (int k = (int)POLY_DEGREE - 1; k >= 0; k--) {
        p = __riscv_vfmadd_vv_f32m2(p, s, __riscv_vfmv_v_f_f32m2(c[k], vl), vl);
    }
    vfloat32m2_t core = __riscv_vfmul_vv_f32m2(reduced, p, vl);         // reduced * P(s)
    vfloat32m2_t comp = __riscv_vfrsub_vf_f32m2(core, RC_PI_HALF, vl);  // pi/2 - core
    vfloat32m2_t mag = __riscv_vmerge_vvm_f32m2(core, comp, big, vl);
    return __riscv_vfsgnj_vv_f32m2(mag, x, vl);  // copysign(mag, x)
}
#endif  // RVK_RR_MODE_RECIP_COMPLEMENT

// ============================================================================
// EVAL_METHOD_NEWTON_ROOT — production reference: newton_root_sqrt /
// newton_root_rsqrt / newton_root_cbrt_magic (fp32 branches). The LUT is
// ignored on this path, exactly as in production.
// ============================================================================
#if defined(RVK_RR_MODE_NEWTON)

#ifndef NEWTON_ROOT_MAGIC
#define NEWTON_ROOT_MAGIC 0x5f1110a0
#endif
#ifndef NEWTON_ROOT_C1
#define NEWTON_ROOT_C1 2.2825186f
#endif
#ifndef NEWTON_ROOT_C2
#define NEWTON_ROOT_C2 2.2533049f
#endif
#ifndef NEWTON_ROOT_C0
#define NEWTON_ROOT_C0 0x1.c09806p0f
#endif
#ifndef NEWTON_ROOT_N
#define NEWTON_ROOT_N 2
#endif
#ifndef NEWTON_ROOT_ITERS
#define NEWTON_ROOT_ITERS 3
#endif
#ifndef NEWTON_ROOT_MAGIC_SCALE
#define NEWTON_ROOT_MAGIC_SCALE 256.0f
#endif
#ifndef NEWTON_ROOT_MAGIC_BIAS
#define NEWTON_ROOT_MAGIC_BIAS 8388608.0f
#endif
#ifndef NEWTON_ROOT_NEG_INV_N_SCALED
#define NEWTON_ROOT_NEG_INV_N_SCALED -0x1.555556p-10f
#endif

static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
#if (NEWTON_ROOT_N == 2) && !defined(NEWTON_ROOT_RECIPROCAL)
    // --- sqrt: magic seed + SQRT_23-bit double-Newton (production structure) --
    vfloat32m2_t y = rvk_rr_fu(
        __riscv_vrsub_vx_u32m2(__riscv_vsrl_vx_u32m2(rvk_rr_ub(x), 1, vl), (uint32_t)(NEWTON_ROOT_MAGIC), vl));
    vfloat32m2_t xy = __riscv_vfmul_vv_f32m2(x, y, vl);
    vfloat32m2_t c = __riscv_vfmul_vv_f32m2(__riscv_vfsgnjn_vv_f32m2(y, y, vl), xy, vl);  // (-y)*xy
    // y *= C1 + c*(C2 + c): add, then fused c*(C2+c)+C1, then mul (production order).
    vfloat32m2_t t = __riscv_vfadd_vf_f32m2(c, (float)(NEWTON_ROOT_C2), vl);
    t = __riscv_vfmadd_vv_f32m2(t, c, __riscv_vfmv_v_f_f32m2((float)(NEWTON_ROOT_C1), vl), vl);
    y = __riscv_vfmul_vv_f32m2(y, t, vl);
    xy = __riscv_vfmul_vv_f32m2(x, y, vl);
    // one_minus_xyy = 1 - y*xy (fused); half_xy = xy * 0.5 — the production
    // addexp(xy,-1) exponent decrement equals a *0.5 exact pow2 mul for every
    // normal xy (subnormal/zero xy only occur outside the fitted domain).
    vfloat32m2_t om = __riscv_vfnmsac_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), y, xy, vl);
    vfloat32m2_t hxy = __riscv_vfmul_vf_f32m2(xy, 0.5f, vl);
    vfloat32m2_t corr = __riscv_vfmadd_vv_f32m2(om, hxy, xy, vl);  // om*hxy + xy (fused)
    // Final correction only where bits(x) < bits(+inf) SIGNED — same predicate
    // as production (negatives take it too, then get overwritten by NaN).
    vbool16_t mfin = __riscv_vmslt_vx_i32m2_b16(rvk_rr_ib(x), 0x7F800000, vl);
    y = __riscv_vmerge_vvm_f32m2(y, corr, mfin, vl);
    y = __riscv_vfmerge_vfm_f32m2(y, __builtin_nanf(""), __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl), vl);
    return y;

#elif (NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL) && defined(NEWTON_ROOT_ALGORITHM_SQRT_23BIT)
    // --- rsqrt, SQRT_23-bit reciprocal reconstruction (production
    // newton_root_rsqrt SQRT_23BIT branch, piecewise_generic.cpp): the SAME
    // sqrt magic seed + C1/C2 quadratic refinement as newton_root_sqrt, with
    // the final Newton coordinate y/2 instead of x*y/2. The bf16 winner emits
    // ITERS==1 (refinement only); ITERS==2 adds the guarded Newton step with
    // the production inf/zero bit predicate. Any other count refuses loudly,
    // mirroring the production #error. NOTE the classic branch below would
    // silently mis-consume this family's C1=2.2825/C2=2.2533 payload as its
    // "1.5" slot — this branch existing is what closes that gap.
    static_assert(
        NEWTON_ROOT_ITERS == 1 || NEWTON_ROOT_ITERS == 2,
        "rvv_rr: SQRT23 reciprocal root requires one or two declared corrections");
    vfloat32m2_t y = rvk_rr_fu(
        __riscv_vrsub_vx_u32m2(__riscv_vsrl_vx_u32m2(rvk_rr_ub(x), 1, vl), (uint32_t)(NEWTON_ROOT_MAGIC), vl));
    vfloat32m2_t xy = __riscv_vfmul_vv_f32m2(x, y, vl);
    vfloat32m2_t c = __riscv_vfmul_vv_f32m2(__riscv_vfsgnjn_vv_f32m2(y, y, vl), xy, vl);  // (-y)*xy
    // y *= C1 + c*(C2 + c): add, then fused c*(C2+c)+C1, then mul — the exact
    // spelling of the sqrt branch above (production sfpu_mad order).
    vfloat32m2_t t = __riscv_vfadd_vf_f32m2(c, (float)(NEWTON_ROOT_C2), vl);
    t = __riscv_vfmadd_vv_f32m2(t, c, __riscv_vfmv_v_f_f32m2((float)(NEWTON_ROOT_C1), vl), vl);
    y = __riscv_vfmul_vv_f32m2(y, t, vl);
#if NEWTON_ROOT_ITERS == 2
    // Guarded final Newton: y = (1 - y*(x*y)) * (y/2) + y on ordinary lanes;
    // bits(inf) - bits(x) substituted where x is +inf (-> +0) or +0 (-> +inf),
    // exactly the production bit predicate (negative x falls through to NaN).
    xy = __riscv_vfmul_vv_f32m2(x, y, vl);
    vfloat32m2_t om = __riscv_vfnmsac_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), y, xy, vl);
    vfloat32m2_t half_y = __riscv_vfmul_vf_f32m2(y, 0.5f, vl);  // addexp(y,-1) equiv (normals)
    vfloat32m2_t corr = __riscv_vfmadd_vv_f32m2(om, half_y, y, vl);
    vint32m2_t inf_minus_x = __riscv_vrsub_vx_i32m2(rvk_rr_ib(x), 0x7F800000, vl);
    vbool16_t m_ord = __riscv_vmand_mm_b16(
        __riscv_vmsne_vx_i32m2_b16(inf_minus_x, 0, vl), __riscv_vmsne_vx_i32m2_b16(rvk_rr_ib(x), 0, vl), vl);
    y = __riscv_vmerge_vvm_f32m2(rvk_rr_fi(inf_minus_x), corr, m_ord, vl);
#endif
    // Production special order: x < 0 -> NaN (both iteration counts).
    y = __riscv_vfmerge_vfm_f32m2(y, __builtin_nanf(""), __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl), vl);
    return y;

#elif (NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL)
    // --- rsqrt: inverse-sqrt magic seed + y = y*(C1 - half_x*y*y) iterations --
    vfloat32m2_t y = rvk_rr_fu(
        __riscv_vrsub_vx_u32m2(__riscv_vsrl_vx_u32m2(rvk_rr_ub(x), 1, vl), (uint32_t)(NEWTON_ROOT_MAGIC), vl));
    vfloat32m2_t half_x = __riscv_vfmul_vf_f32m2(x, 0.5f, vl);  // addexp(x,-1) equiv (normals)
    vfloat32m2_t vc1 = __riscv_vfmv_v_f_f32m2((float)(NEWTON_ROOT_C1), vl);
#pragma GCC unroll 4
    for (int it = 0; it < (int)(NEWTON_ROOT_ITERS); it++) {
        vfloat32m2_t yy = __riscv_vfmul_vv_f32m2(y, y, vl);
        vfloat32m2_t t = __riscv_vfnmsac_vv_f32m2(vc1, half_x, yy, vl);  // C1 - half_x*yy (fused)
        y = __riscv_vfmul_vv_f32m2(y, t, vl);
    }
    // Production special order: x < 0 -> NaN, then x == 0 -> +inf (+/-0 both).
    y = __riscv_vfmerge_vfm_f32m2(y, __builtin_nanf(""), __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl), vl);
    y = __riscv_vfmerge_vfm_f32m2(y, __builtin_inff(), __riscv_vmfeq_vf_f32m2_b16(x, 0.0f, vl), vl);
    return y;

#else
    // --- cbrt: Moroz magic seed + two-stage correction (production fp32 branch
    // of newton_root_cbrt_magic; mirrors ckernel_sfpu_cbrt.h) ------------------
    vfloat32m2_t ax = __riscv_vfsgnjx_vv_f32m2(x, x, vl);  // |x| (odd function)
    // f = (float)bits(|x|) (RNE convert, production RoundMode::Nearest), then
    // f = f*(-1/3 * 2^-8-ish) + magic — magic folded in constexpr float per-op
    // arithmetic exactly as production writes it.
    constexpr float kCbrtMagic =
        ((float)((uint32_t)(NEWTON_ROOT_MAGIC))) / (float)(NEWTON_ROOT_MAGIC_SCALE) + (float)(NEWTON_ROOT_MAGIC_BIAS);
    vfloat32m2_t f = __riscv_vfcvt_f_xu_v_f32m2(rvk_rr_ub(ax), vl);
    f = __riscv_vfmadd_vf_f32m2(f, (float)(NEWTON_ROOT_NEG_INV_N_SCALED), __riscv_vfmv_v_f_f32m2(kCbrtMagic, vl), vl);
    vfloat32m2_t y = rvk_rr_fu(__riscv_vsll_vx_u32m2(rvk_rr_ub(f), 8, vl));
    // Stage 1: c = (ax*y)*(y*y); y *= c*(C2*c + C1) + C0 (production op order).
    vfloat32m2_t c = __riscv_vfmul_vv_f32m2(__riscv_vfmul_vv_f32m2(ax, y, vl), __riscv_vfmul_vv_f32m2(y, y, vl), vl);
    vfloat32m2_t t =
        __riscv_vfmadd_vf_f32m2(c, (float)(NEWTON_ROOT_C2), __riscv_vfmv_v_f_f32m2((float)(NEWTON_ROOT_C1), vl), vl);
    t = __riscv_vfmadd_vv_f32m2(t, c, __riscv_vfmv_v_f_f32m2((float)(NEWTON_ROOT_C0), vl), vl);
    y = __riscv_vfmul_vv_f32m2(y, t, vl);
    // Stage 2: d = ax*y^2; c = d*y - 1; t = c*(-1/3) + 1; y = copysgn(d,x)*(t*t).
    // -1/3 comes from addexp(NEG_INV_N_SCALED, +8) == * 2^8 exact.
    constexpr float kCbrtNegThird = (float)(NEWTON_ROOT_NEG_INV_N_SCALED) * 256.0f;
    vfloat32m2_t d = __riscv_vfmul_vv_f32m2(ax, __riscv_vfmul_vv_f32m2(y, y, vl), vl);
    c = __riscv_vfmadd_vv_f32m2(d, y, __riscv_vfmv_v_f_f32m2(-1.0f, vl), vl);
    t = __riscv_vfmadd_vf_f32m2(c, kCbrtNegThird, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
    d = __riscv_vfsgnj_vv_f32m2(d, x, vl);
    return __riscv_vfmul_vv_f32m2(d, __riscv_vfmul_vv_f32m2(t, t, vl), vl);
#endif
}
#endif  // RVK_RR_MODE_NEWTON

// ============================================================================
// EVAL_METHOD_TRIG_RESIDUAL — production reference: trig_residual_reduce +
// trig_residual_odd_eval (piecewise_generic.cpp). Constants below are the
// exact values the production dispatch programs into Prgm0/1/2.
// ============================================================================
#if defined(RVK_RR_MODE_TRIG)

#if defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD)
constexpr float RVK_RR_TRIG_P0 = -0x1.92p+1f;  // Cody-Waite -pi, 4 parts
constexpr float RVK_RR_TRIG_P1 = -0x1.fbp-11f;
constexpr float RVK_RR_TRIG_P2 = -0x1.51p-21f;      // production Prgm0
constexpr float RVK_RR_TRIG_P3 = -0x1.0b4612p-33f;  // production Prgm1
#elif defined(TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD)
constexpr float RVK_RR_TRIG_P0 = -0x1.92p+0f;  // Cody-Waite -pi/2, 4 parts
constexpr float RVK_RR_TRIG_P1 = -0x1.fbp-12f;
constexpr float RVK_RR_TRIG_P2 = -0x1.51p-22f;      // production Prgm0
constexpr float RVK_RR_TRIG_P3 = -0x1.0b4612p-34f;  // production Prgm1
#else
#error "rvv_rr: EVAL_METHOD_TRIG_RESIDUAL requires a supported TRIG_RESIDUAL_PHASE_*"
#endif
constexpr float RVK_RR_TRIG_INV_PI = 0.31830987334251404f;  // production Prgm2
constexpr float RVK_RR_TRIG_RND = 12582912.0f;              // 1.5*2^23 round magic (0x4B400000)

// Bounds-guarded coefficient read: the degree-5/7 shaped branches below are
// discarded by `if constexpr` at other degrees but still type-checked — keep
// every index formally in range.
constexpr float rvk_rr_trig_c(int k) {
    return (k >= 0 && k <= (int)TRIG_RESIDUAL_DEGREE) ? TRIG_RESIDUAL_COEFFS[k] : 0.0f;
}

static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
#if defined(TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD)
    // z = fma(x, 1/pi, 0.5) — production issues this as ONE sfpu_mad; fused here.
    vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0.5f, vl), RVK_RR_TRIG_INV_PI, x, vl);
    vfloat32m2_t tmp = __riscv_vfadd_vf_f32m2(z, RVK_RR_TRIG_RND, vl);  // RNE round magic
    // Quadrant sign straight off the biased float's LSB — production documents
    // bits(tmp)<<31 bit-exact vs the (int)tmp - (int)bias subtract.
    vint32m2_t sign = __riscv_vsll_vx_i32m2(rvk_rr_ib(tmp), 31, vl);
    vfloat32m2_t q = __riscv_vfsub_vf_f32m2(tmp, RVK_RR_TRIG_RND, vl);
    // j = 2q - 1 (production single sfpu_mad; fused).
    vfloat32m2_t j = __riscv_vfmadd_vf_f32m2(q, 2.0f, __riscv_vfmv_v_f_f32m2(-1.0f, vl), vl);
#else
    // Sine phase: SEPARATE mul then round-magic add — production keeps these
    // two-rounded (TRIG_FUSED_ROUND is default-OFF because fusing moves fp32
    // double-rounding boundary outputs); this file matches the two-op order.
    vfloat32m2_t z = __riscv_vfmul_vf_f32m2(x, RVK_RR_TRIG_INV_PI, vl);
    vfloat32m2_t tmp = __riscv_vfadd_vf_f32m2(z, RVK_RR_TRIG_RND, vl);
    vint32m2_t sign = __riscv_vsll_vx_i32m2(rvk_rr_ib(tmp), 31, vl);
    vfloat32m2_t j = __riscv_vfsub_vf_f32m2(tmp, RVK_RR_TRIG_RND, vl);
#endif
    // Four-part Cody-Waite tail, all fused mads, production order.
    vfloat32m2_t a = __riscv_vfmacc_vf_f32m2(x, RVK_RR_TRIG_P0, j, vl);
    a = __riscv_vfmacc_vf_f32m2(a, RVK_RR_TRIG_P1, j, vl);
    a = __riscv_vfmacc_vf_f32m2(a, RVK_RR_TRIG_P2, j, vl);
    a = __riscv_vfmacc_vf_f32m2(a, RVK_RR_TRIG_P3, j, vl);
    a = rvk_rr_fi(__riscv_vxor_vv_i32m2(rvk_rr_ib(a), sign, vl));  // quadrant sign flip
    vfloat32m2_t s = __riscv_vfmul_vv_f32m2(a, a, vl);

    // Odd residual polynomial — the exact production shapes per degree.
    if constexpr (TRIG_RESIDUAL_DEGREE == 7) {
        vfloat32m2_t r = __riscv_vfmadd_vf_f32m2(s, rvk_rr_trig_c(7), __riscv_vfmv_v_f_f32m2(rvk_rr_trig_c(5), vl), vl);
        vfloat32m2_t cas = __riscv_vfmul_vv_f32m2(a, s, vl);
        r = __riscv_vfmadd_vv_f32m2(r, s, __riscv_vfmv_v_f_f32m2(rvk_rr_trig_c(3), vl), vl);
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return __riscv_vfmadd_vv_f32m2(r, cas, a, vl);
#else
        vfloat32m2_t c1a = __riscv_vfmul_vf_f32m2(a, rvk_rr_trig_c(1), vl);
        return __riscv_vfmadd_vv_f32m2(r, cas, c1a, vl);
#endif
    } else if constexpr (TRIG_RESIDUAL_DEGREE == 5) {
        vfloat32m2_t r = __riscv_vfmadd_vf_f32m2(s, rvk_rr_trig_c(5), __riscv_vfmv_v_f_f32m2(rvk_rr_trig_c(3), vl), vl);
        vfloat32m2_t cas = __riscv_vfmul_vv_f32m2(a, s, vl);
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return __riscv_vfmadd_vv_f32m2(r, cas, a, vl);
#else
        vfloat32m2_t c1a = __riscv_vfmul_vf_f32m2(a, rvk_rr_trig_c(1), vl);
        return __riscv_vfmadd_vv_f32m2(r, cas, c1a, vl);
#endif
    } else {
        // Generic odd Horner in s, times a (production else-branch: includes
        // c1 in the chain regardless of C1_IS_ONE).
        constexpr int kTop =
            (TRIG_RESIDUAL_DEGREE % 2 == 1) ? (int)TRIG_RESIDUAL_DEGREE : (int)TRIG_RESIDUAL_DEGREE - 1;
        constexpr int kSteps = (kTop - 1) / 2;
        vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(rvk_rr_trig_c(kTop), vl);
#pragma GCC unroll 8
        for (int k = 1; k <= kSteps; k++) {
            r = __riscv_vfmadd_vv_f32m2(r, s, __riscv_vfmv_v_f_f32m2(rvk_rr_trig_c(kTop - 2 * k), vl), vl);
        }
        return __riscv_vfmul_vv_f32m2(r, a, vl);
    }
}
#endif  // RVK_RR_MODE_TRIG

// ============================================================================
// REDUCE_TAN cascade — production reference: tan_reduce + the piecewise
// cascade evaluated on the REDUCED argument + tan_expand (piecewise_generic
// piecewise_generic_lut with RANGE_REDUCTION_TAN).
// ============================================================================
#if defined(RVK_RR_MODE_TAN_CASCADE)

static_assert(NUM_SEGMENTS <= 16, "rvv_rr: tan-cascade linear segment select is sized for small S");

// Mirrors of the base kernel's AoS record layout. These are DETERMINISTIC
// functions of POLY_DEGREE and the fixed L1 plan; piecewise_rvv.cpp computes
// the same values as RVK_REC_SHIFT / RVK_COEFF_BASE and stages the records in
// rvk_build_tables() before the tile loop.
constexpr uint32_t rvk_rr_rec_bytes_pow2() {
    uint32_t need = (POLY_DEGREE + 1u) * 4u;
    uint32_t p2 = 4u;
    while (p2 < need) {
        p2 <<= 1;
    }
    return p2;
}
constexpr uint32_t rvk_rr_log2u(uint32_t v) {
    uint32_t sh = 0;
    while ((1u << sh) < v) {
        sh++;
    }
    return sh;
}
constexpr uint32_t RVK_RR_REC_SHIFT = rvk_rr_log2u(rvk_rr_rec_bytes_pow2());
constexpr uint32_t RVK_RR_COEFF_BASE = 0x164000;  // == piecewise_rvv.cpp RVK_COEFF_BASE

static inline vfloat32m2_t rvk_rr_eval_elem(vfloat32m2_t x, size_t vl) {
    // tan_reduce: j = round(x * 2/pi) via the RNE round magic; two-term
    // Cody-Waite a = x - j*(pi/2), all fused mads in production order.
    vfloat32m2_t z = __riscv_vfmul_vf_f32m2(x, 0.6366197723675814f, vl);
    vfloat32m2_t tmp = __riscv_vfadd_vf_f32m2(z, 12582912.0f, vl);
    vfloat32m2_t j = __riscv_vfsub_vf_f32m2(tmp, 12582912.0f, vl);
    // Quadrant parity from the biased float's LSB (== production j_int & 1).
    vbool16_t modd = __riscv_vmsne_vx_u32m2_b16(__riscv_vand_vx_u32m2(rvk_rr_ub(tmp), 1u, vl), 0u, vl);
    vfloat32m2_t a = __riscv_vfmacc_vf_f32m2(x, -1.5703125f, j, vl);  // hi
    a = __riscv_vfmacc_vf_f32m2(a, -0.0004837512969970703f, j, vl);   // lo

    // Segment select on the REDUCED argument. Production cascades
    // v_if(x >= lut[seg]) re-evaluations; the last segment whose left boundary
    // is <= a wins — identical to idx = sum(a >= b_k) (a breakpoint belongs to
    // its RIGHT segment). Boundaries are compile-time constants (LUT_DATA).
    vuint32m2_t idx = __riscv_vmv_v_x_u32m2(0, vl);
    vuint32m2_t vzero = __riscv_vmv_v_x_u32m2(0, vl);
#pragma GCC unroll 16
    for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
        vbool16_t m = __riscv_vmfge_vf_f32m2_b16(a, LUT_DATA[seg], vl);
        idx = __riscv_vadd_vv_u32m2(idx, __riscv_vmerge_vxm_u32m2(vzero, 1, m, vl), vl);
    }
    vuint32m2_t off = __riscv_vsll_vx_u32m2(idx, RVK_RR_REC_SHIFT, vl);

    // Polynomial over the reduced argument, gathered from the AoS records
    // rvk_build_tables() staged. POLY_PARITY_ODD/EVEN reproduce the production
    // cascade's stride-2 x^2-Horner (eval_polynomial_parity) — same coefficient
    // order, so parity-dropped even/odd coefficients are skipped exactly as in
    // production; otherwise the plain full-degree Horner (zero-padding keeps
    // adaptive degrees bit-identical, same as the base kernel's cascade).
    const float* ctab = (const float*)RVK_RR_COEFF_BASE;
#if defined(POLY_PARITY_ODD)
    constexpr int kTop = (POLY_DEGREE % 2 == 1) ? (int)POLY_DEGREE : (int)POLY_DEGREE - 1;
    constexpr int kSteps = (kTop - 1) / 2;
    vfloat32m2_t s2 = __riscv_vfmul_vv_f32m2(a, a, vl);
    vfloat32m2_t acc = __riscv_vluxei32_v_f32m2(ctab + kTop, off, vl);
#pragma GCC unroll 16
    for (int k = 1; k <= kSteps; k++) {
        vfloat32m2_t cj = __riscv_vluxei32_v_f32m2(ctab + kTop - 2 * k, off, vl);
        acc = __riscv_vfmadd_vv_f32m2(acc, s2, cj, vl);
    }
    acc = __riscv_vfmul_vv_f32m2(acc, a, vl);  // final *a for odd parity
#elif defined(POLY_PARITY_EVEN)
    constexpr int kTop = (POLY_DEGREE % 2 == 0) ? (int)POLY_DEGREE : (int)POLY_DEGREE - 1;
    constexpr int kSteps = kTop / 2;
    vfloat32m2_t s2 = __riscv_vfmul_vv_f32m2(a, a, vl);
    vfloat32m2_t acc = __riscv_vluxei32_v_f32m2(ctab + kTop, off, vl);
#pragma GCC unroll 16
    for (int k = 1; k <= kSteps; k++) {
        vfloat32m2_t cj = __riscv_vluxei32_v_f32m2(ctab + kTop - 2 * k, off, vl);
        acc = __riscv_vfmadd_vv_f32m2(acc, s2, cj, vl);
    }
#else
    vfloat32m2_t acc = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, off, vl);
#pragma GCC unroll 17
    for (int k = (int)POLY_DEGREE - 1; k >= 0; k--) {
        vfloat32m2_t cj = __riscv_vluxei32_v_f32m2(ctab + k, off, vl);
        acc = __riscv_vfmadd_vv_f32m2(acc, a, cj, vl);
    }
#endif

    // tan_expand: j odd -> -1/poly(a) (production sfpu_reciprocal_iter<3>;
    // Newton reciprocal above is at least as converged). Computed on all lanes
    // and merged — inactive-lane inf/NaN are merged away, no traps on RVV.
    vfloat32m2_t rec = rvk_rr_recip(acc, vl);
    rec = __riscv_vfsgnjn_vv_f32m2(rec, rec, vl);  // negate
    return __riscv_vmerge_vvm_f32m2(acc, rec, modd, vl);
}
#endif  // RVK_RR_MODE_TAN_CASCADE

// ---------------------------------------------------------------------------
// Shared tile evaluator: 1024 fp32 elements CB->CB, e32m2 (vl=8), 128 chunks,
// A/B chunk pairs (independent chains; -O3 schedules across them to hide the
// vfmadd latency — correctness-first shape, same as the base generic path).
// ---------------------------------------------------------------------------
static inline void rvk_rr_eval_tile(const float* x, float* yout) {
    size_t vl = __riscv_vsetvl_e32m2(8);
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        vfloat32m2_t yA = rvk_rr_eval_elem(xA, vl);
        vfloat32m2_t yB = rvk_rr_eval_elem(xB, vl);
        yA = tt_rvv_finalize_domain_actions(xA, yA, vl);
        yB = tt_rvv_finalize_domain_actions(xB, yB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, yB, vl);
    }
}

#endif  // RVK_RR_ACTIVE && TRISC_PACK
