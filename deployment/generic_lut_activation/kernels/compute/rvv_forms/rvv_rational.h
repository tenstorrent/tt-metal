// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// rvv_forms/rvv_rational.h — RVV-only RATIONAL-CASCADE evaluator (fp32)
// =============================================================================
// Include contract: this header is included by piecewise_rvv.cpp, INSIDE its
// `#ifdef TRISC_PACK` region, AFTER the scratch-map constants (RVK_SCRATCH,
// RVK_CELL_OFF, RVK_BND_HI_OFF, RVK_BND_LO_OFF, RVK_COEFF_BASE, RVK_COEFF_CAP,
// RVK_ERR_INDEX_RESOLUTION) and AFTER the text-pinned LUT copy RVK_LUT_L1.
// It is included ONLY when the generated adhoc is a rational artifact
// (EVAL_METHOD_RATIONAL_CASCADE, EVAL_METHOD_ABS_DENOMINATOR_RATIONAL,
// EVAL_METHOD_SQUARED_ABS_DENOMINATOR_RATIONAL, or the REDUCE_TAN rational
// cascade: TT_ACT_RATIONAL_LUT + EVAL_METHOD_REDUCED_POLY + REDUCE_TAN),
// i.e. the adhoc defines NUM_DEGREE / DEN_DEGREE (and no POLY_DEGREE).
//
// PRODUCTION SEMANTICS REPRODUCED (piecewise_rational.cpp, fp32 / EMBEDDED_LUT):
//   * LUT layout: [b0..bS] boundaries, then per segment
//     [n0..n_ND, d0..d_DD] (CPS = (ND+1)+(DD+1) floats, numerator first).
//   * Segment select: the production cascade is v_if(x >= lut[SEG]) for
//     SEG = 1..S-1 with segment 0 as the default — i.e. segment =
//     max{ k : x >= b_k } clamped to [0, S-1]; a breakpoint belongs to its
//     RIGHT segment (>=), x < b1 uses segment 0 (b0 is never consulted),
//     x >= b_{S-1} uses the last segment (b_S is never consulted). Both index
//     paths below (uniform clamp / 256-cell LUT + one-step fix-up with >=/<
//     against the true breakpoints) implement exactly this convention.
//     KNOWN DIVERGENCE (documented, output-equivalent): a NaN input selects
//     segment 0 in production (all compares false) but clamps to the LAST
//     segment here (vfcvt_rtz(NaN) saturates positive). Either way both
//     Horner chains produce NaN and num * recip(NaN) = NaN * (+/-0) = NaN,
//     so the OUTPUT is NaN in both engines.
//   * Evaluation ORDER (this implementation — gates are ULP-vs-golden):
//     plain full-degree Horner, high->low, on the RAW x for BOTH chains:
//        num = ((n_ND*x + n_{ND-1})*x + ...)*x + n0
//        den = ((d_DD*x + d_{DD-1})*x + ...)*x + d0
//     every step one single-rounding vfmadd. For non-parity shapes this is
//     the SAME recurrence as production eval_rational_interleaved_numer_denom
//     (identical op DAG; engine rounding differs — RVV vfmadd is IEEE-RNE,
//     BH SFPMAD is the semi-sticky FMA of ttpoly/precision/fma.py — so byte
//     identity is NOT claimed, same stance as the poly paths in this kernel).
//     When RATIONAL_NUM_PARITY_ODD/RATIONAL_DEN_PARITY_EVEN are defined the
//     production kernel runs the x^2-Horner instead; here the zero
//     coefficients are simply evaluated in the plain chain (acc*x + 0.0 —
//     exact contribution, but a different rounding ORDER than the x^2 chain).
//     Value-equivalent in exact arithmetic; the harness ULP report arbitrates.
//   * ONE deferred reciprocal after segment selection (exactly like
//     piecewise_rational_specialized.cpp), then y = num * recip.
//   * Postcompose, in production apply_output_postcompose order:
//       (1) ASYMPTOTIC_FACTOR_QUADRATIC with ASYMPTOTIC_QUAD_ROOT_A/B:
//           y = y * (x-A) * (x-B) (left-assoc MULs), then optional
//           y = y * ASYMPTOTIC_SCALE. (x-A) is computed with the SAME exact
//           Sterbenz property (fl(A-A)=0) that makes the interior root exact.
//       (2) POSTCOMPOSE_AFFINE_Y:              y = fma(y, B, A)
//       (3) POSTCOMPOSE_AFFINE_Y_TIMES_INPUT:  y = x * fma(y, B, A)
//   * ABS_DENOMINATOR_RATIONAL (softsign form): production ignores the LUT
//     entirely: den = |x| + 1.0f (one add), y = x * recip(den), postcompose.
//     Reproduced verbatim (vfsgnjx for |x|, vfadd, recip, vfmul); the
//     numerator is literally x, so y(-0.0) = -0.0 like silicon.
//   * SQUARED_ABS_DENOMINATOR_RATIONAL (softsign_bw form): same den = |x|+1
//     reciprocal chain, then y = recip(den) * recip(den) (production r*r; the
//     numerator is the constant 1, so the input sign never reaches y).
//     Exponent-FF inputs (inf AND NaN: den = |x|+1 is exponent-FF for both)
//     egress +0 exactly like the production certificate: the seed's
//     mag >= 2^126 arm returns a signed zero and the NaN first residual
//     fails the t<0 Newton guard, so the squared seed is exactly +0.
//
// -----------------------------------------------------------------------------
// RECIPROCAL WITHOUT FDIV (Zve32f has no vfdiv here; scalar F has no fdiv):
// bit-exact SFPARECIP seed emulation + the production Newton chain.
//
//   Production (BH ckernel_sfpu_recip.h::sfpu_reciprocal_iter<N>):
//     y  = approx_recip(x)           // 8-bit hardware seed
//     t  = x*y - 2.0                 // SFPMAD (negated Newton residual)
//     y1 = y*(-t) - 0.0              // SFPMAD (unconditional)
//     if (t < 0):                    // NaN guard: t=NaN => t>=0 => keep seed
//       t  = x*y1 - 2.0
//       y  = y1*(-t) - 0.0
//
//   Here: the SEED is reproduced BIT-EXACTLY with vector integer math + a
//   128-entry table (the craq-sim approx_recip seed LUT, validated in
//   ttpoly/precision/reciprocal.py):
//     mag < 0x00800000          -> 0x7F800000            (zero/denormal -> inf;
//                                                         both engines are DAZ)
//     mag < 0x7E800000          -> ((253-exp)<<23) | (SEED_LUT[man>>16 & 0x7F])
//     else (x >= 2^126/inf/NaN) -> 0
//   then OR the sign back. The Newton chain is the SAME operation sequence
//   with vfmadd (t = y*den - 2.0; y1 = y*(-t) + 0.0 with -t formed by exact
//   vfsgnjn) and the SAME t<0 guard via vmflt (false for NaN, exactly like
//   the sfpi sign-check trick — RVV canonicalizes to +qNaN, and t = -0 is
//   impossible since t = fl(x*y) - 2 = +0 when x*y == 2).
//   ITERATION COUNT matches production selection:
//     default                          -> 2 Newton steps (fp32 always: the
//                                         one/zero-iter CSV overrides are
//                                         bf16-only in run_csv.sh, and this
//                                         kernel is fp32-only)
//     RATIONAL_RECIPROCAL_ONE_ITER     -> 1 step (kept for exactness)
//     RATIONAL_RECIPROCAL_ZERO_ITER    -> 0 steps (raw seed)
//
//   ZERO-DENOMINATOR CONVENTION (the fitter charter item; production
//   behavior reproduced by construction): den = +/-0 (or +/-denormal under
//   DAZ) -> seed = +/-inf -> t = 0*inf - 2 = NaN -> guard skips every Newton
//   step -> recip = +/-inf -> y = num * (+/-inf) = +/-inf (sign =
//   sign(num) XOR sign(den)), or NaN when num == 0. den = NaN -> seed = +/-0
//   -> t = NaN -> y = +/-0 -> num * 0 (NaN num -> NaN out). |den| >= 2^126
//   -> seed +/-0, Newton proceeds and returns +/-0 (true result is
//   subnormal -> 0 under FTZ). The fitter's has_pole_in_domain /
//   reciprocal_unsafe gates guarantee in-domain denominators never hit the
//   zero branch; the convention only shapes out-of-domain behavior.
//
//   ULP CLASS (measured host-side with the exact seed table + IEEE fp32 FMA
//   over 2e5 random normal denominators):
//     seed:    max rel err 5.58e-3  (2^-7.5)   == SFPARECIP class
//     1 step:  max rel err 3.11e-5  (2^-15.0)
//     2 steps: max rel err 1.18e-7  (2^-23.0), max 1 ULP vs RN(1/x)
//   i.e. the default 2-step reciprocal lands in the SAME <=1-fp32-ULP class
//   as production's declared budget (ttpoly RECIPROCAL_BUDGET["bh","fp32"]
//   max_ulp = 1.0). Individual results may still differ from silicon by
//   ~1 ULP because the Newton FMAs round on different engines (IEEE-RNE here
//   vs the BH semi-sticky SFPMAD); the final division error class is
//   unchanged: |y*den - 1| <= ~2^-23, plus <=0.5 ULP from the num*recip
//   multiply -> total division error <= ~1.5 ULP, identical to production.
// -----------------------------------------------------------------------------
// L1 budget: coefficient records live in the same 32KB window at
// RVK_COEFF_BASE; the 512B reciprocal seed table is pinned into kernel .text
// exactly like RVK_LUT_L1 (TRISC2 LDM is ~1.7KB — nothing new lands there).
// Hot loop: 2-way chunk interleave at e32m2 (the proven no-spill shape of the
// generic poly path; the rational body carries two Horner chains + the
// reciprocal, so the 3-way poly interleave would spill).
// =============================================================================

#ifndef TRISC_PACK
#error "rvv_rational.h must only be included inside piecewise_rvv.cpp's TRISC_PACK region"
#endif

// ---------------------------------------------------------------------------
// Support matrix (rational-specific). The including file already refuses
// USE_BF16, missing EMBEDDED_LUT, FUSE_GRAD_MUL, RANGE_REDUCTION_*,
// PRECOMPOSE_INPUT_AFFINE and ALL asymptotic factors (including QUADRATIC —
// the quad-root arm below is kept as a faithful twin of production's
// apply_output_postcompose guard, which under current codegen is equally
// unreachable: run_csv.sh emits ASYMPTOTIC_QUAD_ROOT_A/B as constexpr floats,
// so `defined(ASYMPTOTIC_QUAD_ROOT_A)` is false in BOTH kernels).
// ---------------------------------------------------------------------------
#if defined(POSTCOMPOSE_AFFINE_Y) && defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT)
#error "rvv_rational: POSTCOMPOSE_AFFINE_Y and POSTCOMPOSE_AFFINE_Y_TIMES_INPUT are exclusive"
#endif

// Squared abs-denominator variant (softsign_bw): y = (1/(1 + |x|))^2 — the
// derivative of the linear quotient below. Same denominator, same reciprocal
// chain, one squaring SFPMUL-equivalent instead of the x multiply; the LUT is
// equally unused. Production reference: piecewise_rational.cpp
// squared_abs_denominator_rational_eval (the plain body — the BH TTI replay
// overlay is target-gated off for the RVV lane by codegen).
#if defined(SQUARED_ABS_DENOMINATOR_RATIONAL) || (TT_ACT_EVAL_KIND == TT_ACT_EVAL_SQUARED_ABS_DENOMINATOR_RATIONAL)
#define RVKR_SQUARED_ABS_DENOMINATOR 1
#else
#define RVKR_SQUARED_ABS_DENOMINATOR 0
#endif

#if defined(ABS_DENOMINATOR_RATIONAL) || (TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_DENOMINATOR_RATIONAL) || \
    RVKR_SQUARED_ABS_DENOMINATOR
#define RVKR_ABS_DENOMINATOR 1
#else
#define RVKR_ABS_DENOMINATOR 0
#endif

#if defined(RATIONAL_RECIPROCAL_ZERO_ITER)
#define RVKR_RECIP_ITERS 0
#elif defined(RATIONAL_RECIPROCAL_ONE_ITER)
#define RVKR_RECIP_ITERS 1
#else
#define RVKR_RECIP_ITERS 2  // fp32 production default (sfpu_reciprocal_iter<2>)
#endif

// ---------------------------------------------------------------------------
// Layout constants. Both chains live in ONE AoS record per segment:
//   rec = [n0..n_ND, d0..d_DD, 0-pad to the next power-of-two bytes]
// so record offset = seg << RVKR_REC_SHIFT (vector shift, no multiply), the
// numerator coefficient j gathers from ctab + j and the denominator
// coefficient j from ctab + (ND+1) + j.
// ---------------------------------------------------------------------------
static_assert(NUM_SEGMENTS >= 1 && NUM_SEGMENTS <= 1024, "rvv_rational: NUM_SEGMENTS must be 1..1024");
static_assert(NUM_DEGREE <= 16 && DEN_DEGREE <= 16, "rvv_rational: NUM/DEN degree must be <= 16");

constexpr uint32_t RVKR_NUM_COEFFS = NUM_DEGREE + 1u;
constexpr uint32_t RVKR_DEN_COEFFS = DEN_DEGREE + 1u;
constexpr uint32_t RVKR_CPS = RVKR_NUM_COEFFS + RVKR_DEN_COEFFS;  // coeffs per segment
constexpr uint32_t RVKR_CO = NUM_SEGMENTS + 1u;                   // coeff base in LUT

static_assert(LUT_SIZE == RVKR_CO + NUM_SEGMENTS * RVKR_CPS, "rvv_rational: unexpected rational LUT layout");

constexpr uint32_t rvkr_rec_bytes_pow2() {
    uint32_t need = RVKR_CPS * 4u;
    uint32_t p = 4u;
    while (p < need) {
        p <<= 1;
    }
    return p;
}
constexpr uint32_t RVKR_REC_BYTES = rvkr_rec_bytes_pow2();
constexpr uint32_t rvkr_log2u(uint32_t v) {
    uint32_t s = 0;
    while ((1u << s) < v) {
        s++;
    }
    return s;
}
constexpr uint32_t RVKR_REC_SHIFT = rvkr_log2u(RVKR_REC_BYTES);
constexpr uint32_t RVKR_TABLE_BYTES = RVKR_ABS_DENOMINATOR ? 0u : (NUM_SEGMENTS << RVKR_REC_SHIFT);
static_assert(
    RVKR_TABLE_BYTES <= RVK_COEFF_CAP,
    "rvv_rational: (ND+DD+2) floats x NUM_SEGMENTS exceed the 32KB coefficient window");

// Header word for hdr[15] (POLY_DEGREE slot): (NUM_DEGREE << 8) | DEN_DEGREE.
constexpr uint32_t RVKR_HDR_DEGREE = (NUM_DEGREE << 8) | DEN_DEGREE;

// ---------------------------------------------------------------------------
// Compile-time uniformity detection — same interior-boundary rule as the poly
// path (segment 0's lo and the last segment's hi are never consulted by the
// production cascade, so uniformity is defined over b1..b_{S-1} only).
// ---------------------------------------------------------------------------
constexpr bool rvkr_uniform_detect() {
    if (NUM_SEGMENTS < 3) {
        return false;  // no interior width to define; tiny S -> generic path
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
// Uniform-map soundness (mirror of the poly path's rvk_uniform_map_sound,
// STRICT version): the affine index map must reproduce the production
// `x >= b_k -> right segment` convention exactly at every interior boundary
// b_k AND its DAZ predecessor — no continuity tolerance here (a rational
// continuity classifier would need constexpr division; every committed
// rational winner is S<=2 and already generic, so strictness costs nothing
// and can never silently misindex a future S>=3 rational kink fit).
constexpr float rvkr_pred_daz(float x) {
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
constexpr bool rvkr_uniform_map_sound() {
    if (!rvkr_uniform_detect()) {
        return false;
    }
    const float w = LUT_DATA[2] - LUT_DATA[1];
    const float base = LUT_DATA[1] - w;
    const float inv_w = 1.0f / w;
    for (uint32_t k = 1; k < NUM_SEGMENTS; k++) {
        const float bk = LUT_DATA[k];
        if (bk != 0.0f && bk > -0x1p-126f && bk < 0x1p-126f) {
            return false;
        }
        const float vb = (bk - base) * inv_w;
        const float vp = (rvkr_pred_daz(bk) - base) * inv_w;
        const bool exact = (vb >= (float)k) && (k + 1 >= NUM_SEGMENTS || vb < (float)(k + 1)) && (vp < (float)k) &&
                           (k <= 1 || vp >= (float)(k - 1));
        if (!exact) {
            return false;
        }
    }
    return true;
}
constexpr bool RVKR_UNIFORM = RVKR_ABS_DENOMINATOR ? true : rvkr_uniform_map_sound();
// (abs-denominator needs no index at all; report "uniform" so the runner's
// fallback-metric checks stay green.)

constexpr float RVKR_UNIF_W = (!RVKR_ABS_DENOMINATOR && RVKR_UNIFORM) ? (LUT_DATA[2] - LUT_DATA[1]) : 1.0f;
constexpr float RVKR_UNIF_BASE = (!RVKR_ABS_DENOMINATOR && RVKR_UNIFORM) ? (LUT_DATA[1] - RVKR_UNIF_W) : 0.0f;
constexpr float RVKR_UNIF_INV_W = (!RVKR_ABS_DENOMINATOR && RVKR_UNIFORM) ? (1.0f / RVKR_UNIF_W) : 0.0f;

constexpr float RVKR_GRID_LO = LUT_DATA[0];
constexpr float RVKR_GRID_HI = LUT_DATA[NUM_SEGMENTS];
constexpr float RVKR_GRID_STEP = (RVKR_GRID_HI - RVKR_GRID_LO) / 256.0f;
constexpr float RVKR_GRID_INV_STEP = 256.0f / (RVKR_GRID_HI - RVKR_GRID_LO);
static_assert(RVKR_GRID_HI > RVKR_GRID_LO, "rvv_rational: degenerate boundary domain");

// Compile-time index-LUT resolution metric + fix-up depth (constexpr twin of
// the rvkr_build_tables cell walk — bit-identical float expressions; see the
// poly path's rvk_grid_metric for the full rationale).
constexpr uint32_t rvkr_grid_metric() {
    uint32_t cell[256] = {};
    uint32_t seg = 0;
    for (uint32_t k = 0; k < 256; k++) {
        float left = RVKR_GRID_LO + (float)(int)k * RVKR_GRID_STEP;
        while (seg + 1 < NUM_SEGMENTS && left >= LUT_DATA[seg + 1]) {
            seg++;
        }
        cell[k] = seg;
    }
    uint32_t metric = 0;
    for (uint32_t k = 0; k + 2 < 256; k++) {
        uint32_t d = cell[k + 2] - cell[k];
        if (d > metric) {
            metric = d;
        }
    }
    uint32_t top = (NUM_SEGMENTS - 1) - cell[254];
    if (top > metric) {
        metric = top;
    }
    return metric;
}
constexpr uint32_t rvkr_fixup_steps() {
    uint32_t m = (RVKR_ABS_DENOMINATOR || RVKR_UNIFORM) ? 0u : rvkr_grid_metric();
    return (m < 1u) ? 1u : m;
}
constexpr uint32_t RVKR_FIXUP_STEPS = rvkr_fixup_steps();
static_assert(
    RVKR_ABS_DENOMINATOR || RVKR_UNIFORM || RVKR_FIXUP_STEPS <= 8u,
    "rvv_rational: index-LUT resolution metric exceeds the supported fix-up depth");

// hdr[17..19] payloads (fp32 bits of index base / scale / boundary hi).
constexpr float RVKR_IDX_BASE_F = RVKR_UNIFORM ? RVKR_UNIF_BASE : RVKR_GRID_LO;
constexpr float RVKR_IDX_SCALE_F = RVKR_UNIFORM ? RVKR_UNIF_INV_W : RVKR_GRID_INV_STEP;
constexpr float RVKR_GRID_HI_F = RVKR_GRID_HI;

// ---------------------------------------------------------------------------
// SFPARECIP seed table, pinned into kernel .text like RVK_LUT_L1 (runtime
// vector gathers force materialization; LDM has no room, L1 text does).
// Entries are the craq-sim approx_recip 8-bit mantissa seeds PRE-SHIFTED into
// bit position [23:16] (seed mantissa top byte), one uint32 per entry so a
// vluxei32 gather with index = ((mag >> 16) & 0x7F) << 2 lands directly.
// Source of truth: ttpoly/precision/reciprocal.py::_RECIP_SEED_LUT (bit-exact
// port of craq-sim tensix.cpp:approx_recip, silicon-validated).
// ---------------------------------------------------------------------------
__attribute__((section(".text.rvkr_recip_lut"), aligned(16))) static constexpr std::array<uint32_t, 128>
    RVKR_RECIP_SEED_L1 = {{
        0x007F0000u, 0x007D0000u, 0x007B0000u, 0x00790000u, 0x00770000u, 0x00750000u, 0x00740000u, 0x00720000u,
        0x00700000u, 0x006E0000u, 0x006D0000u, 0x006B0000u, 0x00690000u, 0x00680000u, 0x00660000u, 0x00640000u,
        0x00630000u, 0x00610000u, 0x00600000u, 0x005E0000u, 0x005D0000u, 0x005B0000u, 0x005A0000u, 0x00580000u,
        0x00570000u, 0x00550000u, 0x00540000u, 0x00530000u, 0x00510000u, 0x00500000u, 0x004F0000u, 0x004D0000u,
        0x004C0000u, 0x004B0000u, 0x004A0000u, 0x00480000u, 0x00470000u, 0x00460000u, 0x00450000u, 0x00440000u,
        0x00420000u, 0x00410000u, 0x00400000u, 0x003F0000u, 0x003E0000u, 0x003D0000u, 0x003C0000u, 0x003B0000u,
        0x003A0000u, 0x00390000u, 0x00380000u, 0x00370000u, 0x00360000u, 0x00350000u, 0x00340000u, 0x00330000u,
        0x00320000u, 0x00310000u, 0x00300000u, 0x002F0000u, 0x002E0000u, 0x002D0000u, 0x002C0000u, 0x002B0000u,
        0x002A0000u, 0x00290000u, 0x00280000u, 0x00280000u, 0x00270000u, 0x00260000u, 0x00250000u, 0x00240000u,
        0x00230000u, 0x00230000u, 0x00220000u, 0x00210000u, 0x00200000u, 0x001F0000u, 0x001F0000u, 0x001E0000u,
        0x001D0000u, 0x001C0000u, 0x001C0000u, 0x001B0000u, 0x001A0000u, 0x00190000u, 0x00190000u, 0x00180000u,
        0x00170000u, 0x00170000u, 0x00160000u, 0x00150000u, 0x00150000u, 0x00140000u, 0x00130000u, 0x00130000u,
        0x00120000u, 0x00110000u, 0x00110000u, 0x00100000u, 0x000F0000u, 0x000F0000u, 0x000E0000u, 0x000E0000u,
        0x000D0000u, 0x000C0000u, 0x000C0000u, 0x000B0000u, 0x000B0000u, 0x000A0000u, 0x00090000u, 0x00090000u,
        0x00080000u, 0x00080000u, 0x00070000u, 0x00070000u, 0x00060000u, 0x00050000u, 0x00050000u, 0x00040000u,
        0x00040000u, 0x00030000u, 0x00030000u, 0x00020000u, 0x00020000u, 0x00010000u, 0x00010000u, 0x00000000u,
    }};

// ---------------------------------------------------------------------------
// init: stage the AoS coefficient records (+ generic-path index/fix-up
// tables), untimed. No C float->int casts (scalar fcvt is RNE-locked).
// Returns the generic-path resolution metric (0 on uniform / abs-den paths).
// ---------------------------------------------------------------------------
static inline uint32_t rvkr_build_tables() {
    if (RVKR_ABS_DENOMINATOR) {
        return 0;  // production ignores the LUT for this form; nothing to stage
    }
    // Opaque pointer: without it GCC fully unrolls the staging loops for
    // small-S LUTs, constant-folds every RVK_LUT_L1 read, DISCARDS the
    // text-pinned array and rematerializes the coefficients as scalar
    // .srodata constants — i.e. in local data memory, whose ~1.7KB budget a
    // big-S rational LUT would overflow at link time. The barrier keeps every
    // read against the L1 .text copy (measured: RVK_LUT_L1 absent from the
    // uniform-build object without this, present with it).
    const float* lut = &RVK_LUT_L1[0];
    asm volatile("" : "+r"(lut));

    // (a) AoS records: [num coeffs, den coeffs, 0-pad]. The LUT stores num
    //     then den contiguously per segment, so the copy is contiguous.
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        float* rec = (float*)(RVK_COEFF_BASE + (s << RVKR_REC_SHIFT));
        for (uint32_t j = 0; j < RVKR_CPS; j++) {
            rec[j] = lut[RVKR_CO + s * RVKR_CPS + j];
        }
        for (uint32_t j = RVKR_CPS; j < (RVKR_REC_BYTES / 4); j++) {
            rec[j] = 0.0f;
        }
    }
    if (RVKR_UNIFORM) {
        return 0;
    }

    // (b)(c)(d): verbatim boundary-table/cell-table/metric logic of the poly
    // path's rvk_build_tables (the index machinery is coefficient-agnostic).
    float* bnd_hi = (float*)(RVK_SCRATCH + RVK_BND_HI_OFF);
    float* bnd_lo = (float*)(RVK_SCRATCH + RVK_BND_LO_OFF);
    uint32_t* cell_tab = (uint32_t*)(RVK_SCRATCH + RVK_CELL_OFF);

    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        if (s + 1 < NUM_SEGMENTS) {
            bnd_hi[s] = lut[s + 1];
        } else {
            ((uint32_t*)bnd_hi)[s] = 0x7FC00000u;  // qNaN sticky top edge
        }
        if (s > 0) {
            bnd_lo[s] = lut[s];
        } else {
            ((uint32_t*)bnd_lo)[s] = 0xFF800000u;  // -inf sticky bottom edge
        }
    }
    for (uint32_t s = NUM_SEGMENTS; s < 1024; s++) {
        ((uint32_t*)bnd_hi)[s] = 0x7FC00000u;
        ((uint32_t*)bnd_lo)[s] = 0xFF800000u;
    }

    uint32_t seg = 0;
    for (uint32_t k = 0; k < 256; k++) {
        float left = RVKR_GRID_LO + (float)(int)k * RVKR_GRID_STEP;
        while (seg + 1 < NUM_SEGMENTS && left >= lut[seg + 1]) {
            seg++;
        }
        cell_tab[k] = seg;
    }

    uint32_t metric = 0;
    for (uint32_t k = 0; k + 2 < 256; k++) {
        uint32_t d = cell_tab[k + 2] - cell_tab[k];
        if (d > metric) {
            metric = d;
        }
    }
    uint32_t top = (NUM_SEGMENTS - 1) - cell_tab[254];
    if (top > metric) {
        metric = top;
    }
    return metric;
}

// ---------------------------------------------------------------------------
// Vector reciprocal: bit-exact SFPARECIP seed + production Newton chain (see
// the header block for the full semantics/ULP analysis). `vneg2`/`vzero` are
// loop-invariant broadcasts (-2.0f / +0.0f) hoisted by the caller.
// ---------------------------------------------------------------------------
static inline vfloat32m2_t rvkr_recip(vfloat32m2_t den, vfloat32m2_t vneg2, vfloat32m2_t vzero, size_t vl) {
    const uint32_t* seed_tab = &RVKR_RECIP_SEED_L1[0];

    // ---- seed: sign(den) | approx_recip(|den|), bit-exact ----
    vuint32m2_t xb = __riscv_vreinterpret_v_f32m2_u32m2(den);
    vuint32m2_t sign = __riscv_vand_vx_u32m2(xb, 0x80000000u, vl);
    vuint32m2_t mag = __riscv_vand_vx_u32m2(xb, 0x7FFFFFFFu, vl);
    // mid-branch value (garbage for the small/big lanes, merged away below;
    // the 253-e underflow wraps harmlessly in unsigned arithmetic)
    vuint32m2_t sexp =
        __riscv_vsll_vx_u32m2(__riscv_vrsub_vx_u32m2(__riscv_vsrl_vx_u32m2(mag, 23, vl), 253u, vl), 23, vl);
    vuint32m2_t sidx =
        __riscv_vsll_vx_u32m2(__riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(mag, 16, vl), 0x7Fu, vl), 2, vl);
    vuint32m2_t sman = __riscv_vluxei32_v_u32m2(seed_tab, sidx, vl);
    vuint32m2_t seed = __riscv_vor_vv_u32m2(sexp, sman, vl);
    vbool16_t m_small = __riscv_vmsltu_vx_u32m2_b16(mag, 0x00800000u, vl);  // zero/denorm -> inf
    vbool16_t m_big = __riscv_vmsgeu_vx_u32m2_b16(mag, 0x7E800000u, vl);    // >= 2^126 -> 0
    seed = __riscv_vmerge_vxm_u32m2(seed, 0x7F800000u, m_small, vl);
    seed = __riscv_vmerge_vxm_u32m2(seed, 0u, m_big, vl);
    seed = __riscv_vor_vv_u32m2(seed, sign, vl);
    vfloat32m2_t y = __riscv_vreinterpret_v_u32m2_f32m2(seed);

#if RVKR_RECIP_ITERS >= 1
    // t = den*y - 2.0 (single-rounding FMA; NaN when den*y is 0*inf). The
    // intrinsics are value-semantic: the compiler materializes any register
    // copy the destructive vfmadd encoding needs.
    vfloat32m2_t t = __riscv_vfmadd_vv_f32m2(y, den, vneg2, vl);
    vfloat32m2_t tn = __riscv_vfsgnjn_vv_f32m2(t, t, vl);  // -t, exact sign flip
    // y1 = y*(-t) + 0.0 (production SFPMAD shape, computed unconditionally)
    vfloat32m2_t y1 = __riscv_vfmadd_vv_f32m2(y, tn, vzero, vl);
    // guard: v_if(t < 0) — false for the +qNaN degenerate lanes -> keep seed
    vbool16_t m_ok = __riscv_vmflt_vf_f32m2_b16(t, 0.0f, vl);
#if RVKR_RECIP_ITERS >= 2
    vfloat32m2_t t2 = __riscv_vfmadd_vv_f32m2(y1, den, vneg2, vl);
    vfloat32m2_t t2n = __riscv_vfsgnjn_vv_f32m2(t2, t2, vl);
    vfloat32m2_t y2 = __riscv_vfmadd_vv_f32m2(y1, t2n, vzero, vl);
    y = __riscv_vmerge_vvm_f32m2(y, y2, m_ok, vl);
#else
    y = __riscv_vmerge_vvm_f32m2(y, y1, m_ok, vl);
#endif
#endif
    return y;
}

// ---------------------------------------------------------------------------
// Postcompose (production apply_output_postcompose order; see header block).
// `vpostA` is the loop-invariant POSTCOMPOSE_A broadcast when an affine
// postcompose is configured.
// ---------------------------------------------------------------------------
#if defined(POSTCOMPOSE_AFFINE_Y) || defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT)
#define RVKR_DECL_POST_CONSTS() vfloat32m2_t vpostA = __riscv_vfmv_v_f_f32m2(POSTCOMPOSE_A, vl)
#else
#define RVKR_DECL_POST_CONSTS() (void)0
#endif

#if defined(ASYMPTOTIC_FACTOR_QUADRATIC) && defined(ASYMPTOTIC_QUAD_ROOT_A) && defined(ASYMPTOTIC_QUAD_ROOT_B)
#if defined(ASYMPTOTIC_SCALE)
#define RVKR_EP_QUADROOT(y, xo)                                                                    \
    do {                                                                                           \
        y = __riscv_vfmul_vv_f32m2(y, __riscv_vfsub_vf_f32m2(xo, ASYMPTOTIC_QUAD_ROOT_A, vl), vl); \
        y = __riscv_vfmul_vv_f32m2(y, __riscv_vfsub_vf_f32m2(xo, ASYMPTOTIC_QUAD_ROOT_B, vl), vl); \
        y = __riscv_vfmul_vf_f32m2(y, ASYMPTOTIC_SCALE, vl);                                       \
    } while (0)
#else
#define RVKR_EP_QUADROOT(y, xo)                                                                    \
    do {                                                                                           \
        y = __riscv_vfmul_vv_f32m2(y, __riscv_vfsub_vf_f32m2(xo, ASYMPTOTIC_QUAD_ROOT_A, vl), vl); \
        y = __riscv_vfmul_vv_f32m2(y, __riscv_vfsub_vf_f32m2(xo, ASYMPTOTIC_QUAD_ROOT_B, vl), vl); \
    } while (0)
#endif
#else
#define RVKR_EP_QUADROOT(y, xo) (void)0
#endif

#if defined(POSTCOMPOSE_AFFINE_Y)
#define RVKR_EP_AFFINE_Y(y, xo) y = __riscv_vfmadd_vf_f32m2(y, POSTCOMPOSE_B, vpostA, vl)
#elif defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT)
#define RVKR_EP_AFFINE_Y(y, xo)                                    \
    do {                                                           \
        y = __riscv_vfmadd_vf_f32m2(y, POSTCOMPOSE_B, vpostA, vl); \
        y = __riscv_vfmul_vv_f32m2(y, xo, vl);                     \
    } while (0)
#else
#define RVKR_EP_AFFINE_Y(y, xo) (void)0
#endif

#define RVKR_POSTCOMPOSE(y, xo)  \
    do {                         \
        RVKR_EP_QUADROOT(y, xo); \
        RVKR_EP_AFFINE_Y(y, xo); \
    } while (0)

// ---------------------------------------------------------------------------
// Per-chunk segment offset (byte offset of the AoS record). Uniform:
// off = clamp(rtz((x - base) * inv_w), 0, S-1) << REC_SHIFT — vector rtz
// only, matching the poly fast path. Generic: 256-cell LUT + bidirectional
// one-step fix-up against the true breakpoints (sticky sentinels staged by
// rvkr_build_tables). Both implement the production >=/right-segment rule.
// ---------------------------------------------------------------------------
#define RVKR_UNIF_OFF(off, x)                                                                            \
    vint32m2_t off##_i = __riscv_vfcvt_rtz_x_f_v_i32m2(                                                  \
        __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(x, RVKR_UNIF_BASE, vl), RVKR_UNIF_INV_W, vl), vl); \
    off##_i = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(off##_i, 0, vl), (int)(NUM_SEGMENTS - 1), vl); \
    vuint32m2_t off = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(off##_i), RVKR_REC_SHIFT, vl)

// Fix-up depth: RVKR_FIXUP_STEPS unrolled one-segment steps (compile-time
// metric twin — exact for any cell hint; one step == the historical body).
#define RVKR_GENERIC_OFF(off, x)                                                                              \
    vint32m2_t off##_i = __riscv_vfcvt_rtz_x_f_v_i32m2(                                                       \
        __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(x, RVKR_GRID_LO, vl), RVKR_GRID_INV_STEP, vl), vl);     \
    off##_i = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(off##_i, 0, vl), 255, vl);                          \
    vuint32m2_t off##_cb = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(off##_i), 2, vl);         \
    vuint32m2_t off##_seg = __riscv_vluxei32_v_u32m2(cell_tab, off##_cb, vl);                                 \
    _Pragma("GCC unroll 8") for (uint32_t off##_fs = 0; off##_fs < RVKR_FIXUP_STEPS; off##_fs++) {            \
        vuint32m2_t off##_sb = __riscv_vsll_vx_u32m2(off##_seg, 2, vl);                                       \
        vfloat32m2_t off##_bh = __riscv_vluxei32_v_f32m2(bnd_hi, off##_sb, vl);                               \
        vfloat32m2_t off##_bl = __riscv_vluxei32_v_f32m2(bnd_lo, off##_sb, vl);                               \
        vbool16_t off##_up = __riscv_vmfge_vv_f32m2_b16(x, off##_bh, vl);                                     \
        vbool16_t off##_dn = __riscv_vmflt_vv_f32m2_b16(x, off##_bl, vl);                                     \
        off##_seg = __riscv_vadd_vv_u32m2(off##_seg, __riscv_vmerge_vxm_u32m2(vzero_u, 1, off##_up, vl), vl); \
        off##_seg = __riscv_vsub_vv_u32m2(off##_seg, __riscv_vmerge_vxm_u32m2(vzero_u, 1, off##_dn, vl), vl); \
    }                                                                                                         \
    vuint32m2_t off = __riscv_vsll_vx_u32m2(off##_seg, RVKR_REC_SHIFT, vl)

// ---------------------------------------------------------------------------
// REDUCE_TAN (the bf16 tan winner: rational cascade + tan range reduction).
// Reduction is the production tan_reduce twin already silicon-proven in
// rvv_forms/rvv_rr.h's poly tan cascade: j = RNE-round(x * 2/pi) via the
// 1.5*2^23 magic, quadrant parity from the biased float's LSB, two-term
// Cody-Waite a = x - j*hi - j*lo (fused mads, production constants). The
// cascade then selects and evaluates on the REDUCED argument a.
// EXPAND — DOCUMENTED CONVENTION (no production rational-tan reference
// exists: piecewise_rational.cpp implements no RANGE_REDUCTION_TAN body and
// the current-toolchain SFPU evaluates this fit UNREDUCED): for odd j,
// tan(x) = -1/R(a) = -Q(a)/P(a), computed in the SWAP form
//     y_odd = -(den * recip(num))
// (one extra bit-exact SFPARECIP-seed reciprocal on the numerator chain)
// rather than recip(recip)-compounding -1/(num*recip(den)). Merged by the
// parity mask; inactive-lane inf/NaN merge away, no traps on RVV.
// ---------------------------------------------------------------------------
#if defined(RANGE_REDUCTION_TAN)
#define RVKR_TAN 1
#define RVKR_TAN_REDUCE(a, x)                                                                                         \
    vfloat32m2_t a##_z = __riscv_vfmul_vf_f32m2(x, 0.6366197723675814f, vl);                                          \
    vfloat32m2_t a##_t = __riscv_vfadd_vf_f32m2(a##_z, 12582912.0f, vl);                                              \
    vbool16_t a##_odd =                                                                                               \
        __riscv_vmsne_vx_u32m2_b16(__riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(a##_t), 1u, vl), 0u, vl); \
    vfloat32m2_t a##_j = __riscv_vfsub_vf_f32m2(a##_t, 12582912.0f, vl);                                              \
    vfloat32m2_t a = __riscv_vfmacc_vf_f32m2(x, -1.5703125f, a##_j, vl);                                              \
    a = __riscv_vfmacc_vf_f32m2(a, -0.0004837512969970703f, a##_j, vl)
#define RVKR_TAN_EXPAND(y, num, den, a)                                                           \
    do {                                                                                          \
        vfloat32m2_t y##_sw = __riscv_vfmul_vv_f32m2(den, rvkr_recip(num, vneg2, vzero, vl), vl); \
        y##_sw = __riscv_vfsgnjn_vv_f32m2(y##_sw, y##_sw, vl);                                    \
        y = __riscv_vmerge_vvm_f32m2(y, y##_sw, a##_odd, vl);                                     \
    } while (0)
#else
#define RVKR_TAN 0
#define RVKR_TAN_EXPAND(y, num, den, a) (void)0
#endif

// ---------------------------------------------------------------------------
// Dual Horner + reciprocal + tan-expand + postcompose for one chunk pair.
// Both chains are plain full-degree Horner (high->low) at the eval argument
// xA/xB (the RAW input, or the REDUCED argument under REDUCE_TAN),
// interleaved A/B and num/den for issue-latency hiding (4 independent FMA
// chains). xoA/xoB is the ORIGINAL input fed to the postcompose (identical
// to xA/xB when no reduction is active).
// ---------------------------------------------------------------------------
#define RVKR_RATIONAL_CORE(xA, xB, xoA, xoB, offA, offB, outA, outB)                             \
    vfloat32m2_t numA = __riscv_vluxei32_v_f32m2(ctab + NUM_DEGREE, offA, vl);                   \
    vfloat32m2_t numB = __riscv_vluxei32_v_f32m2(ctab + NUM_DEGREE, offB, vl);                   \
    vfloat32m2_t denA = __riscv_vluxei32_v_f32m2(ctab + RVKR_NUM_COEFFS + DEN_DEGREE, offA, vl); \
    vfloat32m2_t denB = __riscv_vluxei32_v_f32m2(ctab + RVKR_NUM_COEFFS + DEN_DEGREE, offB, vl); \
    _Pragma("GCC unroll 17") for (int j = (int)NUM_DEGREE - 1; j >= 0; j--) {                    \
        vfloat32m2_t cA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);                          \
        vfloat32m2_t cB = __riscv_vluxei32_v_f32m2(ctab + j, offB, vl);                          \
        numA = __riscv_vfmadd_vv_f32m2(numA, xA, cA, vl);                                        \
        numB = __riscv_vfmadd_vv_f32m2(numB, xB, cB, vl);                                        \
    }                                                                                            \
    _Pragma("GCC unroll 17") for (int j = (int)DEN_DEGREE - 1; j >= 0; j--) {                    \
        vfloat32m2_t cA = __riscv_vluxei32_v_f32m2(ctab + RVKR_NUM_COEFFS + j, offA, vl);        \
        vfloat32m2_t cB = __riscv_vluxei32_v_f32m2(ctab + RVKR_NUM_COEFFS + j, offB, vl);        \
        denA = __riscv_vfmadd_vv_f32m2(denA, xA, cA, vl);                                        \
        denB = __riscv_vfmadd_vv_f32m2(denB, xB, cB, vl);                                        \
    }                                                                                            \
    vfloat32m2_t outA = __riscv_vfmul_vv_f32m2(numA, rvkr_recip(denA, vneg2, vzero, vl), vl);    \
    vfloat32m2_t outB = __riscv_vfmul_vv_f32m2(numB, rvkr_recip(denB, vneg2, vzero, vl), vl);    \
    RVKR_TAN_EXPAND(outA, numA, denA, xA);                                                       \
    RVKR_TAN_EXPAND(outB, numB, denB, xB);                                                       \
    RVKR_POSTCOMPOSE(outA, xoA);                                                                 \
    RVKR_POSTCOMPOSE(outB, xoB);                                                                 \
    outA = tt_rvv_finalize_domain_actions(xoA, outA, vl);                                        \
    outB = tt_rvv_finalize_domain_actions(xoB, outB, vl)

// Single-chunk core for the RVK_INTERLEAVE_1 serial reference (A/B knob in
// piecewise_rvv.cpp): the num/den chains of ONE chunk, same per-element op
// order as one lane of RVKR_RATIONAL_CORE -> bit-exact by construction.
#if defined(TT_RATIONAL_COORDINATE_BOUND)
#define RVKR_BOUND_COORDINATE(x)                                      \
    x = __riscv_vfmax_vf_f32m2(x, -TT_RATIONAL_COORDINATE_BOUND, vl); \
    x = __riscv_vfmin_vf_f32m2(x, TT_RATIONAL_COORDINATE_BOUND, vl)
#else
#define RVKR_BOUND_COORDINATE(x)
#endif

#define RVKR_RATIONAL_CORE1(xA, xoA, offA, outA)                                                 \
    vfloat32m2_t numA = __riscv_vluxei32_v_f32m2(ctab + NUM_DEGREE, offA, vl);                   \
    vfloat32m2_t denA = __riscv_vluxei32_v_f32m2(ctab + RVKR_NUM_COEFFS + DEN_DEGREE, offA, vl); \
    _Pragma("GCC unroll 17") for (int j = (int)NUM_DEGREE - 1; j >= 0; j--) {                    \
        vfloat32m2_t cA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);                          \
        numA = __riscv_vfmadd_vv_f32m2(numA, xA, cA, vl);                                        \
    }                                                                                            \
    _Pragma("GCC unroll 17") for (int j = (int)DEN_DEGREE - 1; j >= 0; j--) {                    \
        vfloat32m2_t cA = __riscv_vluxei32_v_f32m2(ctab + RVKR_NUM_COEFFS + j, offA, vl);        \
        denA = __riscv_vfmadd_vv_f32m2(denA, xA, cA, vl);                                        \
    }                                                                                            \
    vfloat32m2_t outA = __riscv_vfmul_vv_f32m2(numA, rvkr_recip(denA, vneg2, vzero, vl), vl);    \
    RVKR_TAN_EXPAND(outA, numA, denA, xA);                                                       \
    RVKR_POSTCOMPOSE(outA, xoA);                                                                 \
    outA = tt_rvv_finalize_domain_actions(xoA, outA, vl)

// ---- tile evaluators: 1024 elements, 128 chunks @ e32m2 (vl=8), 2-way ------
#if RVKR_ABS_DENOMINATOR
// Per-lane combine after the shared den = |x| + 1.0f reciprocal chain:
//   linear (softsign):      y = x * recip(den)
//   squared (softsign_bw):  y = recip(den) * recip(den)   (production: r*r)
#if RVKR_SQUARED_ABS_DENOMINATOR
#define RVKR_ABS_DEN_COMBINE(y, xv, dv)                    \
    vfloat32m2_t y##_r = rvkr_recip(dv, vneg2, vzero, vl); \
    vfloat32m2_t y = __riscv_vfmul_vv_f32m2(y##_r, y##_r, vl)
#else
#define RVKR_ABS_DEN_COMBINE(y, xv, dv) \
    vfloat32m2_t y = __riscv_vfmul_vv_f32m2(xv, rvkr_recip(dv, vneg2, vzero, vl), vl)
#endif

// abs-denominator forms: den = |x| + 1.0f, then the combine above. LUT unused.
static inline void rvkr_eval_tile(const float* x, float* yout) {
    size_t vl = __riscv_vsetvl_e32m2(8);
    vfloat32m2_t vneg2 = __riscv_vfmv_v_f_f32m2(-2.0f, vl);
    vfloat32m2_t vzero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    RVKR_DECL_POST_CONSTS();

#if RVK_ILV == 1
    // Serial reference (A/B knob): one chunk per iteration, bit-exact.
    for (int c = 0; c < 128; c++) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t dA = __riscv_vfadd_vf_f32m2(__riscv_vfsgnjx_vv_f32m2(xA, xA, vl), 1.0f, vl);
        RVKR_ABS_DEN_COMBINE(yA, xA, dA);
        RVKR_POSTCOMPOSE(yA, xA);
        yA = tt_rvv_finalize_domain_actions(xA, yA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
    }
#else
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        vfloat32m2_t dA = __riscv_vfadd_vf_f32m2(__riscv_vfsgnjx_vv_f32m2(xA, xA, vl), 1.0f, vl);
        vfloat32m2_t dB = __riscv_vfadd_vf_f32m2(__riscv_vfsgnjx_vv_f32m2(xB, xB, vl), 1.0f, vl);
        RVKR_ABS_DEN_COMBINE(yA, xA, dA);
        RVKR_ABS_DEN_COMBINE(yB, xB, dB);
        RVKR_POSTCOMPOSE(yA, xA);
        RVKR_POSTCOMPOSE(yB, xB);
        yA = tt_rvv_finalize_domain_actions(xA, yA, vl);
        yB = tt_rvv_finalize_domain_actions(xB, yB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, yB, vl);
    }
#endif  // RVK_ILV
}
#else
static inline void rvkr_eval_tile(const float* x, float* yout) {
    const float* ctab = (const float*)RVK_COEFF_BASE;
    const uint32_t* cell_tab = (const uint32_t*)(RVK_SCRATCH + RVK_CELL_OFF);
    const float* bnd_hi = (const float*)(RVK_SCRATCH + RVK_BND_HI_OFF);
    const float* bnd_lo = (const float*)(RVK_SCRATCH + RVK_BND_LO_OFF);
    (void)cell_tab;
    (void)bnd_hi;
    (void)bnd_lo;

    size_t vl = __riscv_vsetvl_e32m2(8);
    vfloat32m2_t vneg2 = __riscv_vfmv_v_f_f32m2(-2.0f, vl);
    vfloat32m2_t vzero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    [[maybe_unused]] vuint32m2_t vzero_u = __riscv_vmv_v_x_u32m2(0, vl);
    RVKR_DECL_POST_CONSTS();

#if RVK_ILV == 1
    // Serial reference (A/B knob): one chunk per iteration, bit-exact. The
    // num/den chains inside the chunk stay interleaved (inherent to the form).
    for (int c = 0; c < 128; c++) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
#if RVKR_TAN
        RVKR_TAN_REDUCE(aA, xA);  // select + eval on the REDUCED argument
#else
        vfloat32m2_t aA = xA;
#endif
        RVKR_BOUND_COORDINATE(aA);
        if constexpr (RVKR_UNIFORM) {
            RVKR_UNIF_OFF(offA, aA);
            RVKR_RATIONAL_CORE1(aA, xA, offA, yA);
            __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
        } else {
            RVKR_GENERIC_OFF(offA, aA);
            RVKR_RATIONAL_CORE1(aA, xA, offA, yA);
            __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
        }
    }
#else
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
#if RVKR_TAN
        RVKR_TAN_REDUCE(aA, xA);  // select + eval on the REDUCED argument
        RVKR_TAN_REDUCE(aB, xB);
#else
        vfloat32m2_t aA = xA;
        vfloat32m2_t aB = xB;
#endif
        RVKR_BOUND_COORDINATE(aA);
        RVKR_BOUND_COORDINATE(aB);
        if constexpr (RVKR_UNIFORM) {
            RVKR_UNIF_OFF(offA, aA);
            RVKR_UNIF_OFF(offB, aB);
            RVKR_RATIONAL_CORE(aA, aB, xA, xB, offA, offB, yA, yB);
            __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
            __riscv_vse32_v_f32m2(yout + c * 8 + 8, yB, vl);
        } else {
            RVKR_GENERIC_OFF(offA, aA);
            RVKR_GENERIC_OFF(offB, aB);
            RVKR_RATIONAL_CORE(aA, aB, xA, xB, offA, offB, yA, yB);
            __riscv_vse32_v_f32m2(yout + c * 8, yA, vl);
            __riscv_vse32_v_f32m2(yout + c * 8 + 8, yB, vl);
        }
    }
#endif  // RVK_ILV
}
#endif  // RVKR_ABS_DENOMINATOR
