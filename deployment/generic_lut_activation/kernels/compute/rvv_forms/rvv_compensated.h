// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// RVV COMPENSATED-HORNER FORM (selected by the typed s55/s60 compiler)
// =============================================================================
// Error-free-transformation evaluator for the <= 1 ULP accuracy tier. Exploits
// CORRECTED 2026-08-24: an earlier version of this comment claimed vfmadd /
// vfmsac provide "a fused multiply-add without an intervening FP32 rounding",
// "unlike the SFPU's 27-bit-plus-sticky partially fused MAD path". That is FALSE
// -- RTL shows tt_vfp_lane.sv instantiates the same tt_rv_mad.sv MAD as the
// SFPU tile (MAN_PROD_BITS_TRUNC=28, ADDER_WIDTH=30, MAN_BITS=24), so TwoProd is
// not exact here either (0/20000 pairs). It does not need to be: the unrecovered
// residual is bounded by exactly 2^-3 ulp(product) -- three guard bits -- and an
// erf ablation attributes essentially all of the tier's accuracy to the
// DOUBLE-FLOAT coefficients, not to EFT exactness (exact TwoProd is worth
// 0.118 of a 1.63 ULP improvement). See
// deployment/findings/TWOPROD_ON_BH_MAD.md. Silicon exhibits deterministic, bounded
// +/-1-LSB deviations near rounding ties (best predicted by fma_model_bh,
// 1164/1165 flips; mechanism attribution open pending RTL — see
// deployment/findings/TTSIM_TRUNCATION.md), so the EFT path is
// EMPIRICALLY — not unconditionally IEEE-exactly — validated to <=1 ULP
// for the tested contract domains on this silicon; TwoProd residuals are
// as-intended
// (pi = fl(s*t) residual via one vfmsac, inexact but bounded) and TwoSum chains
// ARE exact (add is correctly rounded: 20000/20000). What keeps this tier off the
// SFPU is the register-spill ICE (sfpi_classes.h:211) and DST/format round-trips,
// not the arithmetic.
//
// EVALUATION (Graillat/Langlois/Louvet CompHorner + double-float coefficients
// + exact argument centering):
//   per segment record: [x0, c0hi, c0lo, c1hi, c1lo, c2, .., cD]  (fp32 each)
//   t = x - x0                       -- EXACT: the fit driver proves the
//                                       Sterbenz condition per segment, and a
//                                       constexpr twin re-proves it here.
//   CompHorner over c_D..c_0 in t:   -- s tracks the plain Horner value, e the
//     p  = fl(s*t)                      exact accumulated correction:
//     pi = fma(s,t,-p)                  TwoProd residual (exact)
//     sn = fl(p + c_j)
//     sg = TwoSum residual (6-flop branchless Knuth TwoSum, exact)
//     e  = fma(e, t, pi + sg [+ c_jlo])
//     s  = sn
//   result = fl(s + e)               -- ONE final rounding.
// The double-float pairs (c0hi+c0lo, c1hi+c1lo) carry ~48-bit leading
// coefficients; their lo halves ride in the correction term for free. The
// evaluated polynomial value is faithful (error ~ u + u^2*cond) to the EXACT
// real value of the DF-coefficient polynomial, so the residual error budget
// is the FIT error, not the evaluation.
//
// PACKED CSV CONVENTION (fit drivers: scripts/fit_compensated.py and
// scripts/fit_ops.py; verification leg: deployment/generic_lut_activation/
// compensated/). A standard plain-polynomial coefficient CSV whose
// nominal degree is D_real + 3 and whose columns mean:
//   c0 = x0,  c1 = c0hi,  c2 = c0lo,  c3 = c1hi,  c4 = c1lo,  c5.. = c2..cD.
// The packed CSV flows through normal LUT_DATA emission, and the base kernel's
// AoS staging already lays the records out zero-padded at power-of-two strides.
// The typed artifact compiler validates the packed declaration in s55 and s60
// emits RVK_COMPENSATED plus its structured evaluator options.
//
// INDEX MODES (compile-time proven, else the build refuses):
//   RVKC_IDX_UNIFORM — affine map idx = rtz((x - base)*inv_w), clamp. Proven
//     with the base kernel's exact production-`>=` soundness argument; the
//     one-float near-boundary fuzz is tolerated only where the PACKED-layout
//     constexpr evaluation of both neighbor segments agrees to <= 2^-4
//     relative (fits here are continuous to ~1e-9 relative, and segment-k
//     centering keeps a misindexed boundary float inside the neighbor's
//     Sterbenz window, so its t stays exact too).
//   RVKC_IDX_LOG2 — bit-grid map idx = (int)(bits(x) - BITS0) >> m, clamp:
//     segments uniform in the fp32 BIT space (2^(23-m) segments per binade,
//     dyadic boundaries). EXACT by construction — integer arithmetic, no
//     rounding, no fuzz. This is the natural segmentation for log-like ops
//     (lgamma's pole side). x <= 0 / NaN clamp to segment 0 / S-1: outside
//     the op's evaluation range, same stance as the base kernel's clamp.
//
// SPECIAL VALUES: exact zeros are exact BY CONSTRUCTION — a segment whose
// special point z has x0 == z and c0hi == c0lo == 0 evaluates to exactly
// +0.0 at x == z (t = 0 collapses every product and TwoSum residual to zero).
// The fit driver pins erf's zero segments and lgamma's root segments (x=1,2)
// this way.
//
// REGISTER SHAPE: e32m2, 2-way chunk interleave by default. CompHorner's live
// set is ~9 m2 groups per chain at the TwoSum peak; 2 chains fit the 32-reg
// file only because temps die quickly — RVKC_SERIAL collapses to one chain
// per iteration if a disasm audit ever shows vs*r/vl*re spills (silicon
// lesson: never trust a measurement without that audit).
// =============================================================================
#pragma once

#if !defined(RVK_COMPENSATED)
#error "rvv_compensated.h included without RVK_COMPENSATED (include is guarded in piecewise_rvv.cpp)"
#endif

// ---------------------------------------------------------------------------
// Layout: nominal (packed) degree and real degree.
//   default: [x0, c0hi, c0lo, c1hi, c1lo, c2..cD]          nominal = D_real+3
//   RVKC_DF4: [x0, c0hi,c0lo, c1hi,c1lo, c2hi,c2lo, c3hi,c3lo, c4..cD]
//             nominal = D_real+5 (double-float on c0..c3 — ops whose zero has
//             order 2-3 with a non-representable leading coefficient, e.g.
//             tanhshrink's x^3/3).
// ---------------------------------------------------------------------------
#if defined(RVKC_DF4)
static_assert(POLY_DEGREE >= 6, "RVK_COMPENSATED/DF4: packed nominal degree = D_real + 5 >= 6");
constexpr uint32_t RVKC_NDF = 4;  // count of double-float leading coefficients
constexpr uint32_t RVKC_DREAL = POLY_DEGREE - 5;
constexpr uint32_t RVKC_FIELD_OF_C(uint32_t j) { return (j >= 4) ? (5u + j) : (1u + 2u * j); }
constexpr uint32_t RVKC_FIELD_OF_CLO(uint32_t j) { return 2u + 2u * j; }
#else
static_assert(POLY_DEGREE >= 4, "RVK_COMPENSATED: packed nominal degree = D_real + 3 >= 4");
constexpr uint32_t RVKC_NDF = 2;
constexpr uint32_t RVKC_DREAL = POLY_DEGREE - 3;  // real polynomial degree
// record field indices: 0=x0 1=c0hi 2=c0lo 3=c1hi 4=c1lo 5..=c2..cD
constexpr uint32_t RVKC_FIELD_OF_C(uint32_t j) { return (j >= 2) ? (3u + j) : (j == 1 ? 3u : 1u); }
constexpr uint32_t RVKC_FIELD_OF_CLO(uint32_t j) { return (j == 1) ? 4u : 2u; }
#endif
// Index-map input transforms (rollout, 2026-08-22; contract REFERENCE.md §8):
//   RVKC_ODD          — evaluate at |x|, copysign at the end (odd f).
//   RVKC_SHIFT_BITS   — log2 bit-grid runs on u = x - C (pole at C > 0).
//   RVKC_REFLECT_BITS — log2 bit-grid runs on u = C - |x|, segment order
//                       reversed (poles at ±C; implies RVKC_ODD).
#if defined(RVKC_SHIFT_BITS) && defined(RVKC_REFLECT_BITS)
#error "RVK_COMPENSATED: SHIFT and REFLECT are mutually exclusive"
#endif
#if defined(RVKC_REFLECT_BITS) && !defined(RVKC_ODD)
#error "RVK_COMPENSATED: REFLECT requires RVKC_ODD (reflected maps are defined on |x|)"
#endif
#if defined(RVKC_SHIFT_BITS)
constexpr float RVKC_SHIFT_C = __builtin_bit_cast(float, (uint32_t)(RVKC_SHIFT_BITS));
#else
constexpr float RVKC_SHIFT_C = 0.0f;
#endif
#if defined(RVKC_REFLECT_BITS)
constexpr float RVKC_REFLECT_C = __builtin_bit_cast(float, (uint32_t)(RVKC_REFLECT_BITS));
#else
constexpr float RVKC_REFLECT_C = 0.0f;
#endif

// ---------------------------------------------------------------------------
// Constexpr helpers (exact IEEE fp32 per operation, like the vector unit).
// ---------------------------------------------------------------------------
constexpr uint32_t rvkc_bits(float f) { return __builtin_bit_cast(uint32_t, f); }
constexpr float rvkc_pred_daz(float x) {
    if (x == 0.0f) {
        return -0x1p-126f;
    }
    const uint32_t b = __builtin_bit_cast(uint32_t, x);
    const float p = __builtin_bit_cast(float, (x > 0.0f) ? (b - 1u) : (b + 1u));
    if (p > -0x1p-126f && p < 0x1p-126f) {
        return 0.0f;
    }
    return p;
}
// Exact-real DF-polynomial value of packed segment s at x (double accumulate:
// a continuity CLASSIFIER for the fuzz tolerance, not the engine recurrence).
constexpr double rvkc_seg_value_at(uint32_t s, float x) {
    const uint32_t co = (NUM_SEGMENTS + 1u) + s * (POLY_DEGREE + 1u);
    const double t = (double)x - (double)LUT_DATA[co + 0];
    double acc = 0.0;
    for (int j = (int)RVKC_DREAL; j >= 0; j--) {
        double c = (double)LUT_DATA[co + RVKC_FIELD_OF_C((uint32_t)j)];
        if ((uint32_t)j < RVKC_NDF) {
            c += (double)LUT_DATA[co + RVKC_FIELD_OF_CLO((uint32_t)j)];
        }
        acc = acc * t + c;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Kink boundaries (rollout). The uniform affine map rounds fl(x - base): every
// float within F ~ ulp(b_k - base) of a boundary can index one segment off.
// Harmless when both neighbor fits are the SAME analytic function (erf-class:
// disagreement ~2^-30 rel), catastrophic when the op is kinked (prelu: 4x
// slope) or C1-only (celu/softsign: curvature jump -> multiple ULP inside the
// fuzz zone). Classifier: probe both neighbor DF polynomials at b_k +/- F;
// relative disagreement > 2^-26 flags a kink. Flagged boundaries get an EXACT
// vector-compare index correction (x >= b_k selects segment k, production
// convention, no rounding); up to 4 kinks, else the build refuses.
// ---------------------------------------------------------------------------
constexpr float rvkc_ulpf(float v) {
    v = (v < 0.0f) ? -v : v;
    if (v < 0x1p-126f) {
        return 0x1p-149f;
    }
    return __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, v) + 1u) - v;
}
constexpr bool rvkc_kink_at(uint32_t k) {
    const float w = LUT_DATA[2] - LUT_DATA[1];
    const float base = LUT_DATA[1] - w;
    const float bk = LUT_DATA[k];
    const float F = 2.0f * rvkc_ulpf(bk - base);
    for (int s = 0; s < 2; s++) {
        const float xp = s ? (bk + F) : (bk - F);
        const double pl = rvkc_seg_value_at(k - 1, xp);
        const double pr = rvkc_seg_value_at(k, xp);
        double d = pl - pr;
        d = (d < 0.0) ? -d : d;
        const double al = (pl < 0.0) ? -pl : pl;
        const double ar = (pr < 0.0) ? -pr : pr;
        const double sc = (al > ar) ? al : ar;
        if (d > sc * 0x1p-26) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Index-mode detection + soundness proofs.
// ---------------------------------------------------------------------------
constexpr bool rvkc_uniform_detect() {
    if (NUM_SEGMENTS < 3) {
        return false;
    }
    float w = LUT_DATA[2] - LUT_DATA[1];
    if (!(w > 0.0f)) {
        return false;
    }
    for (uint32_t k = 1; k + 1 < NUM_SEGMENTS; k++) {
        if (LUT_DATA[k + 1] - LUT_DATA[k] != w) {
            return false;
        }
    }
    return true;
}
constexpr bool rvkc_uniform_sound() {
    if (!rvkc_uniform_detect()) {
        return false;
    }
    const float w = LUT_DATA[2] - LUT_DATA[1];
    const float base = LUT_DATA[1] - w;
    const float inv_w = 1.0f / w;
    for (uint32_t k = 1; k < NUM_SEGMENTS; k++) {
        const float bk = LUT_DATA[k];
        if (bk != 0.0f && bk > -0x1p-126f && bk < 0x1p-126f) {
            return false;  // subnormal boundary: engine sees 0
        }
        const float vb = (bk - base) * inv_w;
        const float vp = (rvkc_pred_daz(bk) - base) * inv_w;
        const bool exact = (vb >= (float)k) && (k + 1 >= NUM_SEGMENTS || vb < (float)(k + 1)) && (vp < (float)k) &&
                           (k <= 1 || vp >= (float)(k - 1));
        if (exact) {
            continue;
        }
        if (!(vb >= (float)(k - 1) && (k + 1 >= NUM_SEGMENTS || vb < (float)(k + 1)))) {
            return false;  // > one segment off: never tolerable
        }
        if (!(vp < (float)(k + 1) && (k <= 1 || vp >= (float)(k - 1)))) {
            return false;
        }
        // Fuzz at b_k: tolerable iff the neighbor DF polynomials agree (same
        // analytic function) — or the boundary is kink-flagged, in which case
        // the exact vector-compare correction (below) makes the index exact.
        if (rvkc_kink_at(k)) {
            continue;
        }
        const double pl = rvkc_seg_value_at(k - 1, bk);
        const double pr = rvkc_seg_value_at(k, bk);
        const double d = (pl >= pr) ? (pl - pr) : (pr - pl);
        const double al = (pl >= 0.0) ? pl : -pl;
        const double ar = (pr >= 0.0) ? pr : -pr;
        const double sc = (al >= ar) ? al : ar;
        if (!(d <= sc * 0x1p-4)) {
            return false;
        }
    }
    return true;
}
// Bit-grid (log2) mode: every boundary must sit exactly on the uniform grid in
// fp32 bit space, all boundaries positive normals, step a power of two.
// Generalized (rollout): the grid may live in a TRANSFORMED coordinate
//   u_j = b_j - C            (SHIFT;   u ascending with j)
//   u_j = C - b_{S-j}        (REFLECT; x-boundaries descend as u ascends)
// with every u_j required EXACT in fp32 (checked against the double value).
constexpr float rvkc_u_of_boundary(uint32_t j) {
#if defined(RVKC_SHIFT_BITS)
    return (float)((double)LUT_DATA[j] - (double)RVKC_SHIFT_C);
#elif defined(RVKC_REFLECT_BITS)
    return (float)((double)RVKC_REFLECT_C - (double)LUT_DATA[NUM_SEGMENTS - j]);
#else
    return LUT_DATA[j];
#endif
}
constexpr bool rvkc_u_boundary_exact(uint32_t j) {
#if defined(RVKC_SHIFT_BITS)
    const double u = (double)LUT_DATA[j] - (double)RVKC_SHIFT_C;
#elif defined(RVKC_REFLECT_BITS)
    const double u = (double)RVKC_REFLECT_C - (double)LUT_DATA[NUM_SEGMENTS - j];
#else
    const double u = (double)LUT_DATA[j];
#endif
    return (double)(float)u == u;
}
constexpr bool rvkc_log2_sound() {
    if (NUM_SEGMENTS < 2) {
        return false;
    }
    if (!(rvkc_u_of_boundary(0) >= 0x1p-126f)) {
        return false;  // positive normals only (in u space)
    }
    const uint32_t b0 = rvkc_bits(rvkc_u_of_boundary(0));
    const uint32_t step = rvkc_bits(rvkc_u_of_boundary(1)) - b0;
    if (step == 0 || (step & (step - 1)) != 0) {
        return false;  // step must be a power of two
    }
    for (uint32_t k = 0; k <= NUM_SEGMENTS; k++) {
        if (!rvkc_u_boundary_exact(k)) {
            return false;  // transformed boundary would round: refuse
        }
        if (!(rvkc_u_of_boundary(k) > 0.0f)) {
            return false;
        }
        if (rvkc_bits(rvkc_u_of_boundary(k)) != b0 + k * step) {
            return false;
        }
    }
#if defined(RVKC_REFLECT_BITS)
    // Interior x may round in fl(C - |x|): the map can misassign floats within
    // ~ulp(u) of an interior boundary (more than one float where ulp(u) >
    // ulp(x)). Tolerable only where neighbor fits agree at the boundary —
    // require <= 2^-4 relative discrepancy, the uniform-mode fuzz bar; the
    // excursion (<= ulp(u_max) ~ 2^-24 absolute) is negligible vs any segment
    // width the 32KB table budget allows.
    for (uint32_t k = 1; k < NUM_SEGMENTS; k++) {
        const float bk = LUT_DATA[k];
        const double pl = rvkc_seg_value_at(k - 1, bk);
        const double pr = rvkc_seg_value_at(k, bk);
        const double d = (pl >= pr) ? (pl - pr) : (pr - pl);
        const double al = (pl >= 0.0) ? pl : -pl;
        const double ar = (pr >= 0.0) ? pr : -pr;
        const double sc = (al >= ar) ? al : ar;
        if (!(d <= sc * 0x1p-4)) {
            return false;
        }
    }
#endif
    return true;
}
constexpr uint32_t rvkc_log2u(uint32_t v) {
    uint32_t s = 0;
    while ((1u << s) < v) {
        s++;
    }
    return s;
}
#if defined(RVKC_SHIFT_BITS) || defined(RVKC_REFLECT_BITS)
constexpr bool RVKC_UNIFORM = false;  // transformed grids are bit-grids by construction
#else
constexpr bool RVKC_UNIFORM = rvkc_uniform_sound();
#endif
constexpr bool RVKC_LOG2 = !RVKC_UNIFORM && rvkc_log2_sound();
static_assert(
    RVKC_UNIFORM || RVKC_LOG2,
    "RVK_COMPENSATED: boundaries are neither a sound uniform grid nor a bit-space "
    "(log2, possibly shifted/reflected) grid — this form refuses to guess an index map");

// ---------------------------------------------------------------------------
// Small-|x| linear lane (auto-enabled when the table has a zero-through
// segment: x0 == 0 && c0hi == c0lo == 0, i.e. f(0) = 0 fits like erf).
// Rationale: for outputs below ~2^-101 the CompHorner correction terms
// (TwoProd residuals, c1lo*t) fall into the subnormal range and FTZ erases
// them, leaving out = RN(c1hi*x) — up to |C - RN32(C)| + 0.5 rounding
// ≈ 1.37 ULP (measured; C = the op's slope at 0). The lane rescales by an
// EXACT power of two first: xt = x*2^60; p = c1hi*xt; pi = fma(c1hi,xt,-p)
// (exact); y = fl(p + fl(fma(c1lo, xt, pi))); out = y*2^-60 (exact — outputs
// at |x| >= 2^-126 stay normal). One rounding beyond the exact DF product:
// faithful (<= ~0.5 ULP). Selected per lane for |x| < 2^-30, where the
// dropped quadratic/cubic terms are < 2^-31 relative (fit c1 tracks the
// slope to the segment's relative minimax error). NaN compares false and
// keeps the cascade value.
// ---------------------------------------------------------------------------
// GATE FIX (rollout): the lane models a LINEAR zero (out ~= DF(c1)*x). An op
// whose zero has order >= 2 (tanhshrink: x^3/3) stores c1hi == 0 in its
// zero-through segment — the lane would return c1*x = 0 where the reference
// is a tiny NORMAL number (up to 2^23 ULP off). Such tables keep the full
// CompHorner path at small |x| (correct under the contract's output-FTZ
// clause). Lane requires c1hi != 0.
constexpr bool rvkc_has_zero_segment() {
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        const uint32_t co = (NUM_SEGMENTS + 1u) + s * (POLY_DEGREE + 1u);
        if (LUT_DATA[co + 0] == 0.0f && LUT_DATA[co + 1] == 0.0f && LUT_DATA[co + 2] == 0.0f &&
            LUT_DATA[co + RVKC_FIELD_OF_C(1)] != 0.0f) {
            return true;
        }
    }
    return false;
}
// RVKC_SMALLX_CUBIC (define): the CUBIC analog for ops whose zero has order
// 3 (tanhshrink, f ~ x^3/3). Below |x| ~ 2^-33 the CompHorner correction
// chain e ~ c3lo*t^3 falls under 2^-126 and FTZ erases the DF4 low half
// (measured: 1.28 ULP at x ~ -2^-34). The lane rescales by an EXACT 2^40,
// cubes with TwoProd EFTs (xt^3 = p2 + e2 + e1*xt exactly), applies the
// segment's DF c3, and rescales by an exact 2^-120. Selected for
// |x| < 2^-13 (dropped x^5 term < 2^-27 relative there). Requires DF4.
#if defined(RVKC_SMALLX_CUBIC) && !defined(RVKC_DF4)
#error "RVKC_SMALLX_CUBIC requires RVKC_DF4 (c3 must be double-float)"
#endif
constexpr bool RVKC_SMALLX = rvkc_has_zero_segment();
static_assert(
    !RVKC_SMALLX || RVKC_DREAL >= 2,
    "RVK_COMPENSATED: the small-x lane captures c1 at unroll step j==1, which a "
    "DREAL<2 table never executes — pad the fit to D_real >= 2 (rollout trap, prelu)");
constexpr float RVKC_SMALLX_THR = 0x1p-30f;
constexpr float RVKC_SMALLX_UP = 0x1p60f;
constexpr float RVKC_SMALLX_DOWN = 0x1p-60f;

// Kink-corrected index (uniform mode): collect flagged boundaries.
constexpr uint32_t rvkc_kink_count() {
    if (!RVKC_UNIFORM) {
        return 0;
    }
    uint32_t n = 0;
    for (uint32_t k = 1; k < NUM_SEGMENTS; k++) {
        if (rvkc_kink_at(k)) {
            n++;
        }
    }
    return n;
}
constexpr uint32_t RVKC_NKINK = rvkc_kink_count();
static_assert(
    RVKC_NKINK <= 4,
    "RVK_COMPENSATED: more than 4 kink boundaries — refusing (each costs an exact "
    "compare per chunk; refit with fewer kinks or extend the budget deliberately)");
constexpr uint32_t rvkc_kink_kth(uint32_t i) {
    uint32_t n = 0;
    for (uint32_t k = 1; RVKC_UNIFORM && k < NUM_SEGMENTS; k++) {
        if (rvkc_kink_at(k)) {
            if (n == i) {
                return k;
            }
            n++;
        }
    }
    return 0;
}
constexpr float RVKC_KX[4] = {
    RVKC_NKINK > 0 ? LUT_DATA[rvkc_kink_kth(0)] : 0.0f,
    RVKC_NKINK > 1 ? LUT_DATA[rvkc_kink_kth(1)] : 0.0f,
    RVKC_NKINK > 2 ? LUT_DATA[rvkc_kink_kth(2)] : 0.0f,
    RVKC_NKINK > 3 ? LUT_DATA[rvkc_kink_kth(3)] : 0.0f,
};
constexpr int32_t RVKC_KS[4] = {
    (int32_t)rvkc_kink_kth(0),
    (int32_t)rvkc_kink_kth(1),
    (int32_t)rvkc_kink_kth(2),
    (int32_t)rvkc_kink_kth(3),
};

constexpr float RVKC_W = RVKC_UNIFORM ? (LUT_DATA[2] - LUT_DATA[1]) : 1.0f;
constexpr float RVKC_BASE_F = RVKC_UNIFORM ? (LUT_DATA[1] - RVKC_W) : 0.0f;
constexpr float RVKC_INV_W = RVKC_UNIFORM ? (1.0f / RVKC_W) : 0.0f;
constexpr uint32_t RVKC_BITS0 = RVKC_LOG2 ? rvkc_bits(rvkc_u_of_boundary(0)) : 0u;
constexpr uint32_t RVKC_BIT_SHIFT =
    RVKC_LOG2 ? rvkc_log2u(rvkc_bits(rvkc_u_of_boundary(1)) - rvkc_bits(rvkc_u_of_boundary(0))) : 0u;

// ---------------------------------------------------------------------------
// Exact-centering proof: for every segment, x - x0 is exact for every fp32 x
// the index map can send there (its own boundaries, INCLUSIVE of both edges —
// closure covers the one-float uniform fuzz in either direction).
// Sterbenz: x/2 <= x0 <= 2x (same sign), or x0 == 0.
// ---------------------------------------------------------------------------
constexpr bool rvkc_centering_exact() {
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        const uint32_t co = (NUM_SEGMENTS + 1u) + s * (POLY_DEGREE + 1u);
        const float x0 = LUT_DATA[co + 0];
        const float lo = LUT_DATA[s];
        const float hi = LUT_DATA[s + 1];
        if (x0 == 0.0f) {
            continue;  // t = x - 0 always exact
        }
        if (x0 > 0.0f) {
            if (!(lo >= 0.5f * x0 && hi <= 2.0f * x0)) {
                return false;
            }
        } else {
            if (!(hi <= 0.5f * x0 && lo >= 2.0f * x0)) {
                return false;
            }
        }
    }
    return true;
}
static_assert(
    rvkc_centering_exact(),
    "RVK_COMPENSATED: some segment's center x0 violates the Sterbenz window — "
    "x - x0 would round and the EFT contract breaks; fix the fit CSV");

// ============================================================================
// Evaluator (vector code below expects vl = 8 at e32m2 and the staged AoS
// records at RVK_COEFF_BASE with RVK_REC_SHIFT strides — the base kernel's
// staging is reused unchanged; only the record FIELDS are reinterpreted).
// ============================================================================

// Per-chain index computation: off = record byte offset vector from x.
#if defined(RVK_COMPENSATED_DOC_ONLY)
// (documentation stub — real code below)
#endif

// NOTE: the mode select is `if constexpr` on the C++ constexpr bool
// RVKC_UNIFORM — NEVER `#if RVKC_UNIFORM` (a constexpr is not a preprocessor
// symbol; the preprocessor silently evaluates it as 0 and always picks the
// log2 branch — silicon-caught bug, first bring-up run on chip 2).
#define RVKC_OFF(off, xv)                                                                                 \
    vuint32m2_t off;                                                                                      \
    do {                                                                                                  \
        vint32m2_t oi_;                                                                                   \
        if constexpr (RVKC_UNIFORM) {                                                                     \
            oi_ = __riscv_vfcvt_rtz_x_f_v_i32m2(                                                          \
                __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(xv, RVKC_BASE_F, vl), RVKC_INV_W, vl), vl); \
            _Pragma("GCC unroll 4") for (uint32_t ki_ = 0; ki_ < RVKC_NKINK; ki_++) {                     \
                vbool16_t mk_ = __riscv_vmfge_vf_f32m2_b16(xv, RVKC_KX[ki_], vl);                         \
                oi_ = __riscv_vmerge_vvm_i32m2(                                                           \
                    __riscv_vmin_vx_i32m2(oi_, RVKC_KS[ki_] - 1, vl),                                     \
                    __riscv_vmax_vx_i32m2(oi_, RVKC_KS[ki_], vl),                                         \
                    mk_,                                                                                  \
                    vl);                                                                                  \
            }                                                                                             \
        } else {                                                                                          \
            vfloat32m2_t uv_ = xv;                                                                        \
            if constexpr (RVKC_SHIFT_C != 0.0f) {                                                         \
                uv_ = __riscv_vfsub_vf_f32m2(xv, RVKC_SHIFT_C, vl);                                       \
            }                                                                                             \
            if constexpr (RVKC_REFLECT_C != 0.0f) {                                                       \
                uv_ = __riscv_vfrsub_vf_f32m2(xv, RVKC_REFLECT_C, vl); /* C - |x| */                      \
            }                                                                                             \
            oi_ = __riscv_vsra_vx_i32m2(                                                                  \
                __riscv_vsub_vx_i32m2(                                                                    \
                    __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vreinterpret_v_f32m2_u32m2(uv_)),          \
                    (int32_t)RVKC_BITS0,                                                                  \
                    vl),                                                                                  \
                RVKC_BIT_SHIFT,                                                                           \
                vl);                                                                                      \
            if constexpr (RVKC_REFLECT_C != 0.0f) {                                                       \
                oi_ = __riscv_vrsub_vx_i32m2(oi_, (int)(NUM_SEGMENTS - 1), vl); /* reverse order */       \
            }                                                                                             \
        }                                                                                                 \
        oi_ = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(oi_, 0, vl), (int)(NUM_SEGMENTS - 1), vl);      \
        off = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(oi_), RVK_REC_SHIFT, vl);          \
    } while (0)

#if defined(RVKC_SMALLX_CUBIC)
#define RVKC_CAPTURE3(j, c, clo, c3hi, c3lo) \
    if (j == 3) {                            \
        c3hi = c;                            \
        c3lo = clo;                          \
    }
#define RVKC_CUBIC_LANE(acc, xv, c3hi, c3lo)                                                            \
    do {                                                                                                \
        vfloat32m2_t xt_ = __riscv_vfmul_vf_f32m2(xv, 0x1p40f, vl); /* exact */                         \
        vfloat32m2_t p1_ = __riscv_vfmul_vv_f32m2(xt_, xt_, vl);                                        \
        vfloat32m2_t e1_ = __riscv_vfmsac_vv_f32m2(p1_, xt_, xt_, vl); /* exact */                      \
        vfloat32m2_t p2_ = __riscv_vfmul_vv_f32m2(p1_, xt_, vl);                                        \
        vfloat32m2_t e2_ = __riscv_vfmsac_vv_f32m2(p2_, p1_, xt_, vl); /* exact */                      \
        vfloat32m2_t lo_ = __riscv_vfmacc_vv_f32m2(e2_, e1_, xt_, vl); /* e2 + e1*xt */                 \
        vfloat32m2_t h3_ = __riscv_vfmul_vv_f32m2(c3hi, p2_, vl);                                       \
        vfloat32m2_t r3_ = __riscv_vfmsac_vv_f32m2(h3_, c3hi, p2_, vl); /* exact */                     \
        vfloat32m2_t cr_ = __riscv_vfmacc_vv_f32m2(r3_, c3hi, lo_, vl);                                 \
        cr_ = __riscv_vfmacc_vv_f32m2(cr_, c3lo, p2_, vl);                                              \
        vfloat32m2_t yc_ = __riscv_vfadd_vv_f32m2(h3_, cr_, vl);                                        \
        yc_ = __riscv_vfmul_vf_f32m2(yc_, 0x1p-120f, vl); /* exact */                                   \
        vbool16_t mc_ = __riscv_vmflt_vf_f32m2_b16(__riscv_vfsgnjx_vv_f32m2(xv, xv, vl), 0x1p-13f, vl); \
        acc = __riscv_vmerge_vvm_f32m2(acc, yc_, mc_, vl);                                              \
    } while (0)
#else
#define RVKC_CAPTURE3(j, c, clo, c3hi, c3lo) (void)0
#define RVKC_CUBIC_LANE(acc, xv, c3hi, c3lo) (void)0
#endif

// Final-add FastTwoSum fold: on for DF4 layouts (tanhshrink, |e| can reach
// ~2 ulp(s) off the D=7 chain) and for RVKC_FOLD-tagged tables (irrational-
// root ops sin/cos/digamma, where a single silicon tie-flip near the root
// costs a full ULP). Exact algebra; one pass shrinks the final residual 2^24x.
#if defined(RVKC_FOLD)
constexpr bool RVKC_DO_FOLD = true;
#else
constexpr bool RVKC_DO_FOLD = (RVKC_NDF == 4);
#endif

// One full compensated evaluation of a chunk: x (loaded), off (indexed) -> acc.
// s/e recurrence exactly as documented in the header block. The j-loop is
// unrolled with constant bounds so every field index folds at compile time.
#define RVKC_EVAL(acc, xv, off)                                                                                     \
    vfloat32m2_t acc;                                                                                               \
    do {                                                                                                            \
        vfloat32m2_t x0_ = __riscv_vluxei32_v_f32m2(ctab + 0, off, vl);                                             \
        vfloat32m2_t t_ = __riscv_vfsub_vv_f32m2(xv, x0_, vl);                                                      \
        vfloat32m2_t s_ = __riscv_vluxei32_v_f32m2(ctab + RVKC_FIELD_OF_C(RVKC_DREAL), off, vl);                    \
        vfloat32m2_t e_ = __riscv_vfmv_v_f_f32m2(0.0f, vl);                                                         \
        vfloat32m2_t c1hi_ = e_, c1lo_ = e_; /* captured at j==1 (small-x lane) */                                  \
        vfloat32m2_t c3hi_ = e_, c3lo_ = e_; /* captured at j==3 (cubic lane)   */                                  \
        (void)c3hi_;                                                                                                \
        (void)c3lo_;                                                                                                \
        _Pragma("GCC unroll 17") for (int j = (int)RVKC_DREAL - 1; j >= 0; j--) {                                   \
            const uint32_t fhi_ = RVKC_FIELD_OF_C((uint32_t)j);                                                     \
            vfloat32m2_t c_ = __riscv_vluxei32_v_f32m2(ctab + fhi_, off, vl);                                       \
            vfloat32m2_t p_ = __riscv_vfmul_vv_f32m2(s_, t_, vl);                                                   \
            vfloat32m2_t pi_ = __riscv_vfmsac_vv_f32m2(p_, s_, t_, vl); /* s*t - p (exact) */                       \
            vfloat32m2_t sn_ = __riscv_vfadd_vv_f32m2(p_, c_, vl);                                                  \
            vfloat32m2_t bh_ = __riscv_vfsub_vv_f32m2(sn_, p_, vl);                                                 \
            vfloat32m2_t ah_ = __riscv_vfsub_vv_f32m2(sn_, bh_, vl);                                                \
            vfloat32m2_t db_ = __riscv_vfsub_vv_f32m2(c_, bh_, vl);                                                 \
            vfloat32m2_t da_ = __riscv_vfsub_vv_f32m2(p_, ah_, vl);                                                 \
            vfloat32m2_t w_ = __riscv_vfadd_vv_f32m2(pi_, __riscv_vfadd_vv_f32m2(da_, db_, vl), vl);                \
            if (j < (int)RVKC_NDF) {                                                                                \
                vfloat32m2_t clo_ = __riscv_vluxei32_v_f32m2(ctab + RVKC_FIELD_OF_CLO((uint32_t)j), off, vl);       \
                w_ = __riscv_vfadd_vv_f32m2(w_, clo_, vl);                                                          \
                if (RVKC_SMALLX && j == 1) {                                                                        \
                    c1hi_ = c_;                                                                                     \
                    c1lo_ = clo_;                                                                                   \
                }                                                                                                   \
                RVKC_CAPTURE3(j, c_, clo_, c3hi_, c3lo_);                                                           \
            }                                                                                                       \
            e_ = __riscv_vfmadd_vv_f32m2(e_, t_, w_, vl);                                                           \
            s_ = sn_;                                                                                               \
        }                                                                                                           \
        acc = __riscv_vfadd_vv_f32m2(s_, e_, vl);                                                                   \
        if constexpr (RVKC_DO_FOLD) {                                                                               \
            /* DF4 final-add fold (silicon mitigation, tanhshrink): the D=7    */                                   \
            /* correction chain can leave |e| ~ 1-2 ulp(s); the vector adder's */                                   \
            /* operand-alignment truncation then costs up to 2 lsb (chip-2     */                                   \
            /* finding, seg-51 tie cluster). One FastTwoSum pass shrinks the   */                                   \
            /* residual by 2^24: d is Sterbenz-exact, the re-add's operand gap */                                   \
            /* exceeds the truncation window's harm range.                     */                                   \
            vfloat32m2_t d_ = __riscv_vfsub_vv_f32m2(acc, s_, vl);                                                  \
            vfloat32m2_t e2_ = __riscv_vfsub_vv_f32m2(e_, d_, vl);                                                  \
            acc = __riscv_vfadd_vv_f32m2(acc, e2_, vl);                                                             \
        }                                                                                                           \
        RVKC_CUBIC_LANE(acc, xv, c3hi_, c3lo_);                                                                     \
        if (RVKC_SMALLX) { /* constexpr; folded out for f(0) != 0 tables */                                         \
            vfloat32m2_t xt_ = __riscv_vfmul_vf_f32m2(xv, RVKC_SMALLX_UP, vl); /* exact 2^60 */                     \
            vfloat32m2_t ps_ = __riscv_vfmul_vv_f32m2(c1hi_, xt_, vl);                                              \
            vfloat32m2_t pis_ = __riscv_vfmsac_vv_f32m2(ps_, c1hi_, xt_, vl); /* exact residual */                  \
            vfloat32m2_t ys_ = __riscv_vfmadd_vv_f32m2(c1lo_, xt_, pis_, vl); /* c1lo*xt + pi */                    \
            ys_ = __riscv_vfadd_vv_f32m2(ys_, ps_, vl);                                                             \
            ys_ = __riscv_vfmul_vf_f32m2(ys_, RVKC_SMALLX_DOWN, vl); /* exact 2^-60 */                              \
            vbool16_t msx_ = __riscv_vmflt_vf_f32m2_b16(__riscv_vfsgnjx_vv_f32m2(xv, xv, vl), RVKC_SMALLX_THR, vl); \
            acc = __riscv_vmerge_vvm_f32m2(acc, ys_, msx_, vl);                                                     \
        }                                                                                                           \
    } while (0)

static inline void rvkc_eval_tile(const float* x, float* yout) {
    const float* ctab = (const float*)RVK_COEFF_BASE;
    // Pin the dynamic rounding mode to RNE (defensive; costs one CSR write).
    // SILICON FINDING (chip 2 bring-up, 2026-08-22): the vector FP datapath
    // deviates from the IEEE host replica by +/-1 ulp on 167/450560 sampled
    // inputs — every one within 0.008 ulp of an exact rounding TIE of the
    // polynomial value. Not the tie rule (RNA models mismatch; this frm
    // write changes nothing) and not the final add alone (routing s+e
    // through vfmadd changes nothing): the best-fitting model is operand-
    // alignment truncation at ~26 bits in add/FMA (global K=26 pre-align
    // reproduces 100/167 exactly and the 0.008-ulp flip window = 2^-26).
    // Base kernels can never see this — EFT evaluation is the first
    // consumer precise enough. Impact is bounded: flips occur only inside
    // the tie window, so measured accuracy stays <= ~0.52 ULP; the replay
    // gate accepts them as counted, ulp-bounded deviations.
    asm volatile("csrwi frm, 0");
    size_t vl = __riscv_vsetvl_e32m2(8);
    // RVKC_CBRT_EXP3: exponent-mod-3 range reduction (rollout, cbrt).
    //   |x| = m * 2^(3q+r), m in [1,2), r in {0,1,2}  ->  table arg m' =
    //   m*2^r in [1,8), result = eval(m') * 2^q with the 2^q multiply EXACT
    //   (power of two; q in [-42,2] for normal inputs so no under/overflow).
    //   Integer-exact: t = biased_exp-1 in [0,252]; q0 = floor(t/3) via
    //   (t*2731)>>13 (exact for t <= 8190); q = q0-42. Subnormal inputs are
    //   only +/-0 under the DAZ contract clause and take the ODD zero merge.
#if defined(RVKC_CBRT_EXP3)
#if !defined(RVKC_ODD)
#error "RVKC_CBRT_EXP3 requires RVKC_ODD"
#endif
#define RVKC_RRED(mv, sc, av)                                                                                        \
    vfloat32m2_t mv, sc;                                                                                             \
    do {                                                                                                             \
        vuint32m2_t b_ = __riscv_vreinterpret_v_f32m2_u32m2(av);                                                     \
        vuint32m2_t t_ = __riscv_vsub_vx_u32m2(__riscv_vsrl_vx_u32m2(b_, 23, vl), 1u, vl);                           \
        vuint32m2_t q_ = __riscv_vsrl_vx_u32m2(__riscv_vmul_vx_u32m2(t_, 2731u, vl), 13u, vl);                       \
        vuint32m2_t r_ = __riscv_vsub_vv_u32m2(t_, __riscv_vmul_vx_u32m2(q_, 3u, vl), vl);                           \
        vuint32m2_t mb_ = __riscv_vor_vv_u32m2(                                                                      \
            __riscv_vand_vx_u32m2(b_, 0x007FFFFFu, vl),                                                              \
            __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(r_, 127u, vl), 23u, vl),                                     \
            vl);                                                                                                     \
        mv = __riscv_vreinterpret_v_u32m2_f32m2(mb_);                                                                \
        sc = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(q_, 85u, vl), 23u, vl)); \
    } while (0)
#define RVKC_SCALE(acc, sc) acc = __riscv_vfmul_vv_f32m2(acc, sc, vl)
#else
#define RVKC_RRED(mv, sc, av) \
    vfloat32m2_t mv = av;     \
    vfloat32m2_t sc = av
#define RVKC_SCALE(acc, sc) (void)(sc)
#endif
    // RVKC_ODD: index/evaluate at |x|, copysign the result from x; x == +/-0
    // merges to x itself (exact +/-0 — the log2 grids never contain 0).
#if defined(RVKC_ODD)
#define RVKC_ABSFOLD(ax, xv) vfloat32m2_t ax = __riscv_vfsgnjx_vv_f32m2(xv, xv, vl)
#define RVKC_ABSUNFOLD(acc, xv, ax)                               \
    do {                                                          \
        acc = __riscv_vfsgnj_vv_f32m2(acc, xv, vl);               \
        vbool16_t mz_ = __riscv_vmfeq_vf_f32m2_b16(ax, 0.0f, vl); \
        acc = __riscv_vmerge_vvm_f32m2(acc, xv, mz_, vl);         \
    } while (0)
#else
#define RVKC_ABSFOLD(ax, xv) vfloat32m2_t ax = xv
#define RVKC_ABSUNFOLD(acc, xv, ax) (void)0
#endif
#if defined(RVKC_SERIAL)
    for (int c = 0; c < 128; c++) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        RVKC_ABSFOLD(aA, xA);
        RVKC_RRED(mA, scA, aA);
        RVKC_OFF(offA, mA);
        RVKC_EVAL(accA, mA, offA);
        RVKC_SCALE(accA, scA);
        RVKC_ABSUNFOLD(accA, xA, aA);
        accA = tt_rvv_finalize_domain_actions(xA, accA, vl);
        accA = tt_rvv_finalize_special_values(xA, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
    }
#else
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        RVKC_ABSFOLD(aA, xA);
        RVKC_ABSFOLD(aB, xB);
        RVKC_RRED(mA, scA, aA);
        RVKC_RRED(mB, scB, aB);
        RVKC_OFF(offA, mA);
        RVKC_OFF(offB, mB);
        RVKC_EVAL(accA, mA, offA);
        RVKC_EVAL(accB, mB, offB);
        RVKC_SCALE(accA, scA);
        RVKC_SCALE(accB, scB);
        RVKC_ABSUNFOLD(accA, xA, aA);
        RVKC_ABSUNFOLD(accB, xB, aB);
        accA = tt_rvv_finalize_domain_actions(xA, accA, vl);
        accB = tt_rvv_finalize_domain_actions(xB, accB, vl);
        accA = tt_rvv_finalize_special_values(xA, accA, vl);
        accB = tt_rvv_finalize_special_values(xB, accB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, accB, vl);
    }
#endif
}
