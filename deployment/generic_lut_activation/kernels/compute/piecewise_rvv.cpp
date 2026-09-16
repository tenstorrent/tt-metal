// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// PIECEWISE RVV — RVV-ONLY variant of the embedded-LUT activation kernel
// (magic 0xC0FFEE30). TRISC2 computes ALL tiles CB->CB.
// =============================================================================
// This file is NOT compiled standalone: the typed ``--architecture rvv``
// compile route emits the usual adhoc[N].cpp defines block
// (EMBEDDED_LUT, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA, INPUT_MIN/MAX,
// BASIS_*, optional SEGMENT_DEGREES / POLY_PARITY_*) and then includes THIS
// file instead of piecewise_generic.cpp (or instead of piecewise_rational.cpp
// for rational CSVs).
//
// FORMS. Besides the base poly cascade, typed extension forms plug into this
// kernel, each keyed off the codegen-emitted defines (see the form-dispatch
// block after the includes): the rational cascade (rvv_forms/rvv_rational.h),
// the range-reduced standalone forms (rvv_forms/rvv_rr.h), and the algebraic
// whole-function lowerings (rvv_forms/rvv_lowering.h), plus closed structural
// total forms (rvv_forms/rvv_closed_structural.h). A plain poly-cascade
// adhoc activates none of them and compiles the base kernel unchanged
// (byte-identical codegen, verified at merge time).
//
// DATAFLOW (standard single-stream, NO dual CBs, NO reader/writer changes):
//   reader (BRISC)  -> c_0  -> pack thread (TRISC2, Zve32f) -> c_16 -> writer.
//   The pack thread is the SOLE consumer of c_0 and SOLE producer of c_16,
//   driving both with the raw stream-MMIO-register protocol proven in
//   piecewise_hybrid.cpp (poll received / direct-store acked with a local
//   mirror; local-mirror reserve / fenced direct-store push). The unpack and
//   math kernel_main bodies are COMPLETE NO-OPS that never touch the c_0/c_16
//   counters (the compute-side llk cb_wait_front/cb_pop_front live on the
//   unpack thread and are never called), so nothing races the raw protocol.
//   The host needs NOTHING new: in non-hybrid mode the compute kernel already
//   receives {tiles_this_core} as runtime arg 0 on every TRISC, and the
//   standard reader/writer push/pop c_0/c_16 exactly as before.
//
// -----------------------------------------------------------------------------
// EVALUATOR (pack thread only, fp32 only)
// -----------------------------------------------------------------------------
// Works for any embedded polynomial-cascade LUT with NUM_SEGMENTS <= 1024 and
// POLY_DEGREE <= 16 whose AoS coefficient table fits the 32KB L1 window (the
// static_asserts below refuse anything else at compile time).
//
// Coefficient records: one per segment, (POLY_DEGREE+1) fp32 zero-padded to
// the next power-of-two bytes so record offset = seg << RVK_REC_SHIFT
// (vector shift, no multiply). Zero-padding is EXACT for Horner
// (acc=0 -> fma(0,x,c) == c bit-exactly), so per-segment adaptive degrees
// (SEGMENT_DEGREES) come out bit-identical to full-degree evaluation.
//
// UNIFORM FAST PATH (the RVV thesis: segment count is nearly free).
//   Detected AT COMPILE TIME from the constexpr LUT_DATA boundaries: the fits
//   this backend targets (ttpoly uniform / basis-uniform) force segment 0's lo
//   to the domain edge and widen the LAST segment's hi to the emit domain, so
//   uniformity is defined over the INTERIOR boundaries b1..b_{S-1} only:
//       w = b2 - b1;  uniform <=> every fp32 diff b_{k+1}-b_k == w (k=1..S-2)
//   (verified: ttpoly uniform grids have EXACTLY one unique fp32 width).
//   Index math (per element, vector only — no index-LUT, no fix-up):
//       idx = clamp(vfcvt_rtz((x_eval - base) * inv_w), 0, S-1)
//   with base = b1 - w and inv_w = 1/w, both COMPILE-TIME constants. x below
//   b1 lands in [0,1) (or negative -> rtz toward 0 / clamped) -> widened
//   segment 0; x >= b_{S-1} clamps to S-1 -> widened last segment: the clamp
//   handles both widened edge segments for free.
//   SOUNDNESS PROOF (compile time, rvk_uniform_map_sound): the fast path is
//   taken ONLY when the affine index map provably reproduces the production
//   `x >= b_k -> right segment` convention. For every interior boundary b_k
//   the constexpr replica of the vector math (per-op IEEE fp32, same as the
//   silicon RNE ops) must map b_k into [k, k+1) and pred(b_k) — the largest
//   DAZ-representable float below b_k — into [k-1, k); by monotonicity of the
//   map this proves EVERY fp32 input selects exactly the production segment.
//   A one-step near-boundary rounding fuzz is tolerated ONLY where the fit is
//   CONTINUOUS at b_k (constexpr Horner of both segments agrees to <= 2^-4
//   relative — the documented "fit is continuous there, ULP report is the
//   arbiter" stance, now enforced instead of assumed). Kink fits whose
//   boundaries are deliberately one-ULP-shifted (hardshrink/hardsigmoid/
//   hardtanh/softshrink _bw, selu_bw: S=3/S=8 grids where the shifted kink
//   does not lie on the derived affine grid — the S=3 interior-width test is
//   VACUOUS) fail the proof and route to the generic path, whose fix-up
//   against the true breakpoints is exact by construction.
//   Hot loop @ e32m2 (vl=8): 128 chunks/tile, 3-WAY chunk interleave (A/B/C
//   chains hide the vluxei/vfmadd latency; measured: no chaining, dep-issue
//   ~12.7 cyc, need 3+ independent streams), 42 x 3 chunks + one 2-way tail.
//
// GENERIC FALLBACK (non-uniform boundaries): the piecewise_hybrid generic
//   evaluator — 256-cell index LUT over [LUT_DATA[0], LUT_DATA[NUM_SEGMENTS]],
//   bidirectional fix-up against the real breakpoints (sticky-edge sentinels
//   bnd_hi[last]=NaN, bnd_lo[0]=-inf), 2-way interleave — with the boundary
//   tables sized for 1024 segments. FIX-UP DEPTH is now a COMPILE-TIME
//   constant RVK_FIXUP_STEPS = max(1, resolution metric), where the metric
//   (max segments spanned by any 2-cell window, + clamped-top fold) is
//   computed constexpr from LUT_DATA with the bit-identical float expressions
//   the runtime staging uses. Segments narrower than a grid cell (lgamma /
//   polygamma hand-tiered boundaries, metric 2) therefore get exactly enough
//   fix-up steps to be EXACT — no misindexing, deterministic results. A
//   static_assert refuses metric > RVK_FIXUP_STEPS_MAX (4) at COMPILE time
//   (deterministic loud UNSUPPORTED, never garbage-then-error). The runtime
//   metric check is kept as a defensive cross-check: metric > RVK_FIXUP_STEPS
//   writes error word 0xE0000001 + the metric to scratch and the runner must
//   FAIL the run; tiles still stream so the pipeline never deadlocks.
//
// Basis reconstruction / tails mirror piecewise_generic.cpp production order
// exactly as in piecewise_hybrid.cpp: (1) BASIS_MUL_ABS_X_BEFORE_POST;
// (2) BASIS_MUL_SQRT_1_MINUS_ABS; (3) BASIS_AFFINE_EVEN; (4) BASIS_CLAMP_MAX;
// (5) BASIS_POST_SIGN_X; (6) BASIS_POST_REFLECT_PI; (7) BASIS_LEFT_TAIL_ZERO;
// (8) BASIS_RIGHT_TAIL_IDENTITY. The RVV engine
// rounds independently of the SFPU (NOTE 2026-08-24: vfmadd is NOT a true
// single-rounding FMA -- it is the same partially-fused 28-bit-truncated MAD as
// the SFPU, per tt_rv_mad.sv; what still differs is the EVALUATION ORDER), so
// SFPU byte identity is NOT claimed; the harness ULP report is the arbiter.
// Both engines are DAZ+FTZ (subnormal coefficients behave as +0.0).
//
// DST_COEFF_STORE / DST_COEFF_DISABLE / POLY_TTI_DISABLE defines emitted by
// codegen gate code that lives only in piecewise_generic.cpp; nothing here
// consumes them and they are deliberately IGNORED (no DEST, no Tensix math).
//
// -----------------------------------------------------------------------------
// L1 SCRATCH MAP (BH MEM_L1_SIZE = 0x180000; CBs sit far below at the
// kernel-config/unreserved base; this program allocates no top-down L1
// buffers, and the region was silicon-proven free by the hybrid kernel).
// Header block at 0x160000 — uint32 words, fully inside the 16KB
// RVV_SCRATCH_CSV dump window (host dumps 0x160000..0x163FFF, core (0,0)):
//   hdr[0]  = 0xC0FFEE30 magic (written LAST)
//   hdr[1]  = t_first lo   -- wall clock AFTER table staging, BEFORE tile 0
//   hdr[2]  = t_first hi      (init is deliberately untimed)
//   hdr[3]  = t_last lo    -- wall clock AFTER the last c_16 push
//   hdr[4]  = t_last hi
//   hdr[5]  = tiles_done          (per-tile, progress-visible)
//   hdr[6]  = n_tiles
//   hdr[7]  = uniformity flag: 1 = uniform fast path, 0 = generic fallback
//   hdr[8]  = coeff table bytes staged at 0x164000 (NUM_SEGMENTS << REC_SHIFT)
//   hdr[9]  = ERROR word: 0 = OK; 0xE0000001 = index-LUT resolution violation
//             (generic fallback only)
//   hdr[10] = resolution metric (generic fallback; 0 on the uniform path)
//   hdr[11] = breadcrumb stage: 0x10 pack entry, 0x20 tables staged,
//             0x30 in loop, 0x40 loop done, 0x50 header complete
//   hdr[12] = breadcrumb tile (local index in flight)
//   hdr[13] = elapsed cycles (t_last - t_first), low 32 bits
//   hdr[14] = NUM_SEGMENTS   hdr[15] = POLY_DEGREE   hdr[16] = RVK_REC_SHIFT
//   hdr[17] = fp32 bits of index base (uniform: b1-w; generic: LUT_DATA[0])
//   hdr[20] = bw-io feature bits (rvv_forms/rvv_bw_io.h; written only when
//             active): bit0 bf16 I/O, bit1 fused grad, bit2 asymptotic
//   hdr[18] = fp32 bits of index scale (uniform: inv_w; generic: 256/(hi-lo))
//   hdr[19] = fp32 bits of boundary-domain hi (LUT_DATA[NUM_SEGMENTS])
//   [+0x100]  cell    uint32[256]  (generic fallback index LUT)
//   [+0x800]  bnd_hi  float[1024]  (boundaries[s+1]; NaN sentinel for last)
//   [+0x1800] bnd_lo  float[1024]  (boundaries[s];  -inf sentinel for seg 0)
//   [+0x2800..0x3FFF] spare (still inside the dump window)
// Coefficient table (NOT in the dump window — plan: 0x160000+16KB and up):
//   0x164000 .. 0x16BFFF  AoS records, capacity 32KB
//   (0x16C000 .. 0x17FFFF remain free; static_asserts pin all of this)
// =============================================================================

#include <cstdint>
#include "api/compute/common.h"
#include "eval_method.h"

// ---------------------------------------------------------------------------
// Software-pipelining A/B knob (measurement-only; default OFF = the tuned
// production shapes: 3-way poly / 2-way rational / 2-way lowering). Injected
// via TT_ACT_KERNEL_EXTRA_DEFINES='RVK_INTERLEAVE_1' (serial reference: ONE
// chunk chain per loop iteration — quantifies on silicon what chunk
// interleaving buys against the TRISC2 no-chaining dependent-issue gap),
// 'RVK_INTERLEAVE_2' (two chains) or 'RVK_INTERLEAVE_4' (four chains,
// poly uniform path only — register-pressure probe). Every variant executes
// the IDENTICAL per-element op sequence per chunk (chunks are independent;
// only cross-chunk instruction scheduling changes), so all values are
// bit-exact against each other by construction. RVK_ILV == 0 compiles the
// kernel byte-identically to a build without this block. The generic
// (non-uniform) fallback path is NOT parameterized (the RVV sweeps never
// emit non-uniform CSVs).
// ---------------------------------------------------------------------------
#if defined(RVK_INTERLEAVE_1)
#define RVK_ILV 1
#elif defined(RVK_INTERLEAVE_2)
#define RVK_ILV 2
#elif defined(RVK_INTERLEAVE_4)
#define RVK_ILV 4
#else
#define RVK_ILV 0  // tuned default shapes
#endif

// bf16 tile I/O + fused-grad (_bw) + asymptotic-factor epilogue — an I/O
// layer AROUND the form dispatch below (poly / rational / rr / lowering all
// stay fp32 CB->CB evaluators; this wraps their input/output). Defines
// RVK_BW_IO_ACTIVE / RVK_IO_BF16 / RVK_FUSE_GRAD / RVK_ASYM_ACTIVE for the
// support-matrix guards and, on the pack thread, the stage-in / finish hooks.
#ifdef TRISC_PACK
#include <riscv_vector.h>
#include "domain_actions_rvv.h"
#endif
#include "rvv_forms/rvv_bw_io.h"

// ---------------------------------------------------------------------------
// RVV form dispatch. The three extension forms are keyed off the codegen
// defines block and are MUTUALLY EXCLUSIVE by construction (codegen emits
// exactly one eval-method family; the #error below makes any unexpected
// overlap loud):
//   RVK_FORM_RATIONAL   — piecewise rational cascade P(x)/Q(x) and the
//                         abs-denominator collapses (softsign x/(1+|x|) and
//                         its squared derivative form (1/(1+|x|))^2). The
//                         evaluator header rvv_forms/rvv_rational.h is
//                         included inside the TRISC_PACK region below (it
//                         needs RVK_LUT_L1 and the scratch-map constants).
//   RVK_FORM_RR         — range-reduced standalone forms (exp/log/log1p ALU,
//                         newton_root, trig_residual, REDUCE_TAN cascade).
//                         rvv_forms/rvv_rr.h defines RVK_RR_ACTIVE when the
//                         requested form is implemented; the REDUCE_TAN
//                         cascade reuses the base kernel's staged AoS records.
//   RVK_LOWERING_ACTIVE — algebraic whole-function lowerings (identity /
//                         affine / clamped_affine / threshold_* / gated /
//                         abs / slope_max), rvv_forms/rvv_lowering.h
//                         (always defined, 0 or 1; bypasses the LUT).
// A plain poly-cascade adhoc activates none of them: both headers then
// contribute nothing and the base kernel compiles byte-identically.
// ---------------------------------------------------------------------------
#include "rvv_forms/rvv_rr.h"                 // mode tags on every TRISC; evaluators only on pack
#include "rvv_forms/rvv_lowering.h"           // ditto; defines RVK_LOWERING_ACTIVE (0/1)
#include "rvv_forms/rvv_logodds.h"            // ditto; defines RVK_FORM_LOGODDS (0/1) — the
                                              // normalized log-odds separable basis (logit class)
#include "rvv_forms/rvv_closed_structural.h"  // typed total closed forms

// The rational form also owns the REDUCE_TAN rational cascade (bf16 tan
// winner tan_n*d*_rational + range_reduction tan): codegen emits
// EVAL_METHOD_REDUCED_POLY + REDUCE_TAN on a rational LUT (marker
// TT_ACT_RATIONAL_LUT from the rational adhoc template) instead of
// EVAL_METHOD_RATIONAL_CASCADE. rvv_forms/rvv_rr.h excludes that combination
// from its poly tan-cascade mode, so the two lanes stay mutually exclusive.
#if defined(EVAL_METHOD_RATIONAL_CASCADE) || defined(EVAL_METHOD_ABS_DENOMINATOR_RATIONAL) || \
    defined(EVAL_METHOD_SQUARED_ABS_DENOMINATOR_RATIONAL) ||                                  \
    (defined(TT_ACT_RATIONAL_LUT) && defined(EVAL_METHOD_REDUCED_POLY) && defined(REDUCE_TAN))
#define RVK_FORM_RATIONAL 1
#else
#define RVK_FORM_RATIONAL 0
#endif
#if defined(RVK_RR_ACTIVE)
#define RVK_FORM_RR 1
#else
#define RVK_FORM_RR 0
#endif
// Compensated-Horner accuracy tier (rvv_forms/rvv_compensated.h). The typed
// s55 schedule owns this form and s60 emits RVK_COMPENSATED for a validated
// packed artifact (record fields are [x0, c0hi, c0lo, c1hi, c1lo, c2..cD]).
#if defined(RVK_COMPENSATED)
#define RVK_FORM_COMP 1
#else
#define RVK_FORM_COMP 0
#endif
#if (                                                                                          \
    RVK_FORM_RATIONAL + RVK_FORM_RR + RVK_LOWERING_ACTIVE + RVK_FORM_COMP + RVK_FORM_LOGODDS + \
    RVK_FORM_CLOSED_STRUCTURAL) > 1
#error "piecewise_rvv: conflicting form selectors are mutually exclusive"
#endif

// ---------------------------------------------------------------------------
// Support matrix — refuse at COMPILE TIME anything this evaluator does not
// reproduce. Silent wrong math is never an option.
// ---------------------------------------------------------------------------
// USE_BF16 is handled by rvv_forms/rvv_bw_io.h (exact widen in, RNE pack
// out; compute stays fp32 internally like the SFPU path). The lowering form
// keeps its own bf16 #error until that lane relaxes it.
#ifndef EMBEDDED_LUT
#error "piecewise_rvv: requires an embedded LUT (generated adhoc defines)"
#endif
// FUSE_GRAD_MUL is handled by rvv_forms/rvv_bw_io.h: the pack thread is the
// sole raw-protocol acker of the grad CB c_1, consumed in lockstep with c_0,
// y = eval(x) * grad post-epilogue (before the bf16 repack).
// The rational form exempts both halves (rational adhocs are non-standalone
// and define no EVAL_METHOD_POLY_CASCADE); the RR form exempts both halves
// (its ALU/newton/trig forms ARE standalone); a lowering exempts only the
// missing-cascade half (codegen suppresses the eval-method macro when the
// declared algebraic macro carries TT_ACT_EVAL_KIND, and lowerings are never
// standalone).
#if !RVK_FORM_RATIONAL && !RVK_FORM_RR && \
    (EVAL_METHOD_IS_STANDALONE ||         \
     (!defined(EVAL_METHOD_POLY_CASCADE) && !RVK_LOWERING_ACTIVE && !RVK_FORM_CLOSED_STRUCTURAL))
#error "piecewise_rvv: requested evaluator form has no RVV implementation"
#endif
// The rational form implements exactly ONE reduction: REDUCE_TAN (the
// reduce/eval-at-a/parity-swap-expand cascade in rvv_forms/rvv_rational.h).
// Any other rational+REDUCE_* combination has no evaluator and stays loud.
#if (                                                                                                      \
    defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || defined(RANGE_REDUCTION_TAN) ||       \
    defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_CBRT) || defined(EVAL_METHOD_REDUCED_POLY)) && \
    !RVK_FORM_RR && !(RVK_FORM_RATIONAL && defined(RANGE_REDUCTION_TAN))
#error "piecewise_rvv: this range-reduced LUT form is not supported"
#endif
// ASYMPTOTIC_FACTOR_{QUADRATIC, EXP_QUADRATIC, EXP_LINEAR, X_EXP_LINEAR, X}
// are handled by rvv_forms/rvv_bw_io.h for the poly-cascade family
// (production epilogue order: cascade -> basis -> asymptotic -> bf16 convert
// / grad multiply); the rational form applies its QUADRATIC postcompose
// internally (rvv_forms/rvv_rational.h) and rvv_bw_io.h stands down there.
// Collapse gate: a lowering exempts the whole check (its adhocs define the
// AFFINE_*/CLAMPED_* payload macros AND a non-cascade TT_ACT_EVAL_KIND); the
// rational and RR forms exempt only their own TT_ACT_EVAL_KIND values.
#if (                                                                                                          \
    defined(AFFINE_IDENTITY) || defined(AFFINE_COLLAPSE) || defined(CLAMPED_AFFINE_COLLAPSE) ||                \
    (defined(TT_ACT_EVAL_KIND) && (TT_ACT_EVAL_KIND != 0) && (TT_ACT_EVAL_KIND != TT_ACT_EVAL_POLY_CASCADE) && \
     (TT_ACT_EVAL_KIND != TT_ACT_EVAL_RATIONAL_CASCADE) &&                                                     \
     (TT_ACT_EVAL_KIND != TT_ACT_EVAL_ABS_DENOMINATOR_RATIONAL) && !RVK_FORM_RR &&                             \
     !RVK_FORM_RATIONAL /* REDUCE_TAN rational cascade: TT_ACT_EVAL_KIND is REDUCED_POLY */)) &&               \
    !RVK_LOWERING_ACTIVE && !RVK_FORM_CLOSED_STRUCTURAL
#error "piecewise_rvv: algebraic whole-function collapses bypass the cascade dispatch"
#endif
// The rational form implements POSTCOMPOSE_AFFINE_Y (and _TIMES_INPUT);
// HAS_CRITICAL_POINT stays unconditional.
#if defined(HAS_CRITICAL_POINT) || (defined(POSTCOMPOSE_AFFINE_Y) && !RVK_FORM_RATIONAL)
#error "piecewise_rvv: critical-point / postcompose epilogues are not implemented"
#endif
// BASIS_MUL_SQRT_1_MINUS_ABS / BASIS_POST_REFLECT_PI (the sqrt_factored acos
// basis) and BASIS_RIGHT_TAIL_ABS_AFFINE (the odd_cubic_factored signed-abs
// tail, e.g. tanhshrink basis: |x| > T -> A*|x| + B) are implemented in the
// epilogue below (RVK_EP_MULSQRT / RVK_EP_REFLECT / RVK_EP_RTAIL_ABS,
// production single-tile order from piecewise_generic_specialized.cpp).
#if defined(PRECOMPOSE_INPUT_AFFINE)
#error "piecewise_rvv: input precompose is not implemented"
#endif
// The compensated form is a bare fp32 poly-cascade evaluator: no basis
// reconstruction, no bf16 I/O / fused grad / asymptotic epilogues, no parity
// re-encoding (the packed record layout owns the coefficient semantics).
#if RVK_FORM_COMP && !defined(EVAL_METHOD_POLY_CASCADE)
#error "RVK_COMPENSATED: requires a plain polynomial-cascade packed CSV"
#endif
#if RVK_FORM_COMP &&                                                                                              \
    (defined(BASIS_INPUT_ABS_X) || defined(BASIS_MUL_ABS_X_BEFORE_POST) || defined(BASIS_MUL_SQRT_1_MINUS_ABS) || \
     defined(BASIS_AFFINE_EVEN) || defined(BASIS_CLAMP_MAX) || defined(BASIS_POST_SIGN_X) ||                      \
     defined(BASIS_POST_REFLECT_PI) || defined(BASIS_LEFT_TAIL_ZERO) || defined(BASIS_RIGHT_TAIL_IDENTITY) ||     \
     defined(BASIS_RIGHT_TAIL_ABS_AFFINE) || defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN))
#error "RVK_COMPENSATED: basis/parity fits are not supported by the compensated form"
#endif
#if RVK_FORM_COMP && (defined(USE_BF16) || defined(FUSE_GRAD_MUL) || defined(ASYMPTOTIC_FACTOR_QUADRATIC) || \
                      defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) ||   \
                      defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X))
#error "RVK_COMPENSATED: fp32 CB->CB only (no bf16 I/O, fused grad, or asymptotic epilogue)"
#endif

#if !RVK_FORM_RATIONAL  // poly-cascade layout constants (rational form: rvv_forms/rvv_rational.h)
static_assert(NUM_SEGMENTS >= 1 && NUM_SEGMENTS <= 1024, "piecewise_rvv: NUM_SEGMENTS must be 1..1024");
static_assert(POLY_DEGREE <= 16, "piecewise_rvv: POLY_DEGREE must be <= 16");
static_assert(
    LUT_SIZE == (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1), "piecewise_rvv: unexpected LUT layout");

// ---------------------------------------------------------------------------
// Compile-time layout constants (all three TUs see these; only pack uses them).
// ---------------------------------------------------------------------------
constexpr uint32_t rvk_rec_bytes_pow2() {
    uint32_t need = (POLY_DEGREE + 1u) * 4u;
    uint32_t p = 4u;
    while (p < need) {
        p <<= 1;
    }
    return p;
}
constexpr uint32_t RVK_REC_BYTES = rvk_rec_bytes_pow2();
constexpr uint32_t rvk_log2u(uint32_t v) {
    uint32_t s = 0;
    while ((1u << s) < v) {
        s++;
    }
    return s;
}
constexpr uint32_t RVK_REC_SHIFT = rvk_log2u(RVK_REC_BYTES);
constexpr uint32_t RVK_TABLE_BYTES = NUM_SEGMENTS << RVK_REC_SHIFT;

// ---------------------------------------------------------------------------
// Compile-time uniformity detection over the INTERIOR boundaries b1..b_{S-1}
// (segment 0's lo and the last segment's hi are widened by the fit emitters;
// the index clamp absorbs both). Constexpr float arithmetic is exact IEEE
// fp32 per operation, so this is bit-faithful to what the vector unit sees.
// ---------------------------------------------------------------------------
constexpr bool rvk_uniform_detect() {
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
// ---------------------------------------------------------------------------
// Uniform-map SOUNDNESS PROOF (see header block). All arithmetic below is
// constexpr fp32, exact IEEE-RNE per operation — bit-faithful to the vector
// unit's vfsub/vfmul in RVK_UNIF_OFF.
// ---------------------------------------------------------------------------
// Largest DAZ-representable float strictly below x (both engines flush
// subnormals, so the engine-visible predecessor of a positive normal at the
// bottom of the normal range is +0.0, and of 0.0 is -FLT_MIN).
constexpr float rvk_pred_daz(float x) {
    if (x == 0.0f) {
        return -0x1p-126f;
    }
    const uint32_t b = __builtin_bit_cast(uint32_t, x);
    const float p = __builtin_bit_cast(float, (x > 0.0f) ? (b - 1u) : (b + 1u));
    if (p > -0x1p-126f && p < 0x1p-126f) {
        return 0.0f;  // subnormal predecessor behaves as +0.0 under DAZ
    }
    return p;
}
// Full-degree Horner of segment s at x (non-FMA constexpr — a continuity
// CLASSIFIER, not the engine recurrence; zero-padded adaptive degrees exact).
constexpr float rvk_seg_poly_at(uint32_t s, float x) {
    const uint32_t co = (NUM_SEGMENTS + 1u) + s * (POLY_DEGREE + 1u);
    float acc = LUT_DATA[co + POLY_DEGREE];
    for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
        acc = acc * x + LUT_DATA[co + (uint32_t)j];
    }
    return acc;
}
// Proof: for every interior boundary b_k the affine map must select the
// production segment for b_k AND for pred(b_k); monotonicity extends that to
// every fp32 input. One-step fuzz is tolerated only at continuous boundaries
// (relative discontinuity <= 2^-4); larger jumps demand exactness or demote.
constexpr bool rvk_uniform_map_sound() {
    if (!rvk_uniform_detect()) {
        return false;
    }
    const float w = LUT_DATA[2] - LUT_DATA[1];
    const float base = LUT_DATA[1] - w;
    const float inv_w = 1.0f / w;
    for (uint32_t k = 1; k < NUM_SEGMENTS; k++) {
        const float bk = LUT_DATA[k];
        if (bk != 0.0f && bk > -0x1p-126f && bk < 0x1p-126f) {
            return false;  // subnormal boundary: engine sees 0, math would lie
        }
        const float vb = (bk - base) * inv_w;                // engine map at b_k
        const float vp = (rvk_pred_daz(bk) - base) * inv_w;  // ... at pred(b_k)
        const bool exact = (vb >= (float)k) && (k + 1 >= NUM_SEGMENTS || vb < (float)(k + 1)) && (vp < (float)k) &&
                           (k <= 1 || vp >= (float)(k - 1));
        if (exact) {
            continue;
        }
        // Misindexing farther than ONE segment can never be tolerated.
        if (!(vb >= (float)(k - 1) && (k + 1 >= NUM_SEGMENTS || vb < (float)(k + 1)))) {
            return false;
        }
        if (!(vp < (float)(k + 1) && (k <= 1 || vp >= (float)(k - 1)))) {
            return false;
        }
        // One-step fuzz across b_k: OK only if segments k-1 and k agree there.
        const float pl = rvk_seg_poly_at(k - 1, bk);
        const float pr = rvk_seg_poly_at(k, bk);
        const float d = (pl >= pr) ? (pl - pr) : (pr - pl);
        const float al = (pl >= 0.0f) ? pl : -pl;
        const float ar = (pr >= 0.0f) ? pr : -pr;
        const float s = (al >= ar) ? al : ar;
        if (!(d <= s * 0x1p-4f)) {
            return false;  // discontinuous kink boundary off the affine grid
        }
    }
    return true;
}
constexpr bool RVK_UNIFORM = rvk_uniform_map_sound();
#if defined(RVK_EXPECT_UNIFORM)
static_assert(RVK_UNIFORM, "piecewise_rvv: RVK_EXPECT_UNIFORM set but boundaries are not uniform");
#endif

constexpr float RVK_UNIF_W = RVK_UNIFORM ? (LUT_DATA[2] - LUT_DATA[1]) : 1.0f;
constexpr float RVK_UNIF_BASE = RVK_UNIFORM ? (LUT_DATA[1] - RVK_UNIF_W) : 0.0f;
constexpr float RVK_UNIF_INV_W = RVK_UNIFORM ? (1.0f / RVK_UNIF_W) : 0.0f;

// Generic-fallback 256-cell grid over the boundary domain (compile-time
// division — no scalar fdiv exists on TRISC2).
constexpr float RVK_GRID_LO = LUT_DATA[0];
constexpr float RVK_GRID_HI = LUT_DATA[NUM_SEGMENTS];
constexpr float RVK_GRID_STEP = (RVK_GRID_HI - RVK_GRID_LO) / 256.0f;
constexpr float RVK_GRID_INV_STEP = 256.0f / (RVK_GRID_HI - RVK_GRID_LO);
static_assert(RVK_GRID_HI > RVK_GRID_LO, "degenerate boundary domain");

// ---------------------------------------------------------------------------
// Compile-time index-LUT resolution metric — constexpr replica of the runtime
// cell-table walk in rvk_build_tables() (BIT-IDENTICAL float expressions, so
// runtime metric == this value always). The generic-path fix-up is unrolled
// RVK_FIXUP_STEPS times: exactly enough one-segment steps to reach the true
// segment from any cell hint (metric = max segments a 2-cell window spans,
// + the clamped-top fold). Segments below single-cell resolution (lgamma /
// polygamma hand tiers, metric 2) are therefore EXACT, not "flagged after
// garbage". Anything beyond RVK_FIXUP_STEPS_MAX refuses at compile time.
// ---------------------------------------------------------------------------
constexpr uint32_t rvk_grid_metric() {
    uint32_t cell[256] = {};
    uint32_t seg = 0;
    for (uint32_t k = 0; k < 256; k++) {
        float left = RVK_GRID_LO + (float)(int)k * RVK_GRID_STEP;
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
constexpr uint32_t RVK_FIXUP_STEPS_MAX = 8;
constexpr uint32_t rvk_fixup_steps() {
    uint32_t m = RVK_UNIFORM ? 0u : rvk_grid_metric();
    return (m < 1u) ? 1u : m;
}
constexpr uint32_t RVK_FIXUP_STEPS = rvk_fixup_steps();
// The compensated form supplies its own compile-time-proven index map
// (uniform-affine or log2 bit-grid) and never touches the generic fallback,
// so its dyadic-geometric boundary sets are exempt from the cell-resolution
// refusal (RVK_FORM_COMP below).
static_assert(
    RVK_UNIFORM || RVK_FORM_COMP || RVK_FIXUP_STEPS <= RVK_FIXUP_STEPS_MAX,
    "piecewise_rvv: index-LUT resolution metric exceeds the supported fix-up depth "
    "(segments far below 1/256 of the boundary domain) — unsupported, refusing loudly");
#endif  // !RVK_FORM_RATIONAL

// ============================================================================
// RVV evaluator (pack TRISC only — stock tt-metal enables Zve32f on TRISC2 via
// ComputeConfig::enable_trisc2_rvv, so every RVV type/intrinsic/include stays
// strictly inside this guard).
// ============================================================================
#ifdef TRISC_PACK
#include <riscv_vector.h>
#include "internal/tt-1xx/risc_common.h"  // get_timestamp(), invalidate_l1_cache()

static constexpr uint32_t RVK_SCRATCH = 0x160000;
static constexpr uint32_t RVK_CELL_OFF = 0x100;       // uint32[256]
static constexpr uint32_t RVK_BND_HI_OFF = 0x800;     // float[1024]
static constexpr uint32_t RVK_BND_LO_OFF = 0x1800;    // float[1024]
static constexpr uint32_t RVK_COEFF_BASE = 0x164000;  // above the 16KB dump window
static constexpr uint32_t RVK_COEFF_CAP = 0x8000;     // 32KB
static constexpr uint32_t RVK_ERR_INDEX_RESOLUTION = 0xE0000001u;
static constexpr uint32_t RVK_MAGIC = 0xC0FFEE30u;

// ---------------------------------------------------------------------------
// Runtime-readable LUT copy, pinned into the kernel TEXT image.
//
// rvk_build_tables() reads the LUT at RUNTIME (non-constant indices), which
// forces the array to be materialized. main.ld places ALL kernel data —
// .rodata included — in the local-data-memory (LDM) image, and TRISC2's free
// LDM is ~1.7KB: any NUM_SEGMENTS beyond toy sizes fails to link
// ("segment[1] overflows region:1 ... reduce the size of thread_local
// variables"; s256 d3 needs 5124B). Text, however, lives in L1 (kernel-config
// ring, tens of KB of headroom) and is freely readable on RISC-V, and the
// XIP loader relocates absolute references into text. So the ONLY
// runtime-readable copy of the LUT lives in .text; every compile-time
// consumer (uniformity detection, RVK_* constants, static_asserts) keeps
// using LUT_DATA, which is never materialized.
// ---------------------------------------------------------------------------
__attribute__((section(".text.rvk_lut"), aligned(16))) static constexpr std::array<float, LUT_SIZE> RVK_LUT_L1 =
    LUT_DATA;

// Address-plan proofs. BH L1 is 0x180000 bytes; the deployment host maps no
// top-down L1 buffers, CBs live at the low unreserved base, and the
// 0x160000.. region was silicon-proven free by the hybrid kernel.
static_assert(RVK_BND_LO_OFF + 1024 * 4 <= 0x4000, "small tables must stay inside the 16KB dump window");
#if !RVK_FORM_RATIONAL
static_assert(
    RVK_TABLE_BYTES <= RVK_COEFF_CAP,
    "piecewise_rvv: (POLY_DEGREE+1) records x NUM_SEGMENTS exceed the 32KB coefficient window");
#endif
static_assert(RVK_COEFF_BASE + RVK_COEFF_CAP <= 0x16C000, "coefficient window must end below 0x16C000");
static_assert(0x16C000 <= 0x180000, "must fit BH MEM_L1_SIZE");

#if RVK_FORM_RATIONAL
// Rational-cascade form: staging (rvkr_build_tables), evaluators
// (rvkr_eval_tile) and its own record-layout constants. Must come AFTER the
// scratch-map constants and the text-pinned RVK_LUT_L1 above.
#include "rvv_forms/rvv_rational.h"
#endif

#if RVK_FORM_COMP
// Compensated-Horner form: EFT evaluator + its index-map/centering proofs.
// Reuses the base kernel's AoS staging (records are the packed CSV rows,
// zero-padded at RVK_REC_SHIFT strides) — must come AFTER the poly layout
// constants and RVK_COEFF_BASE above.
#include "rvv_forms/rvv_compensated.h"
#endif

// ---- raw single-stream CB protocol (verified in piecewise_hybrid.cpp) ------
// All four counters are per-stream NOC-overlay scratch registers (stream N =
// CB N), MMIO-readable/writable from any RISC, NOT behind the BH L1 cache.
// 16-bit wrap math throughout, matching dataflow_api.h / llk_io_pack.h.
// We are the SOLE acker of c_0 and SOLE producer of c_16: the unpack/math
// kernel bodies are empty, so no llk path ever touches these counters on
// this core's compute side.

static inline uint16_t rvk_in_tiles_received(uint32_t cb) {
    return (uint16_t)reg_read((uint32_t)(uintptr_t)get_cb_tiles_received_ptr((int)cb));
}

static inline void rvk_in_wait(uint32_t cb, uint16_t my_acked, uint16_t want) {
    while ((uint16_t)(rvk_in_tiles_received(cb) - my_acked) < want) {
    }
}

// pop n tiles of input CB `cb` — sole acker, so a plain MMIO store is the
// complete pop. `my_acked` is the local mirror (reg zeroed at launch).
static inline void rvk_in_pop(uint32_t cb, uint16_t& my_acked, uint16_t n) {
    my_acked += n;
    get_cb_tiles_acked_ptr((int)cb)[0] = my_acked;
}

// byte address of the t-th tile in CB `cb`. Pack-side CB init sets write=true
// so fifo_wr_ptr == base (16-byte units); we never llk-push these CBs, so it
// stays at base.
static inline uint32_t rvk_tile_addr(uint32_t cb, uint32_t t) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    uint32_t base16B = i.fifo_wr_ptr;
    uint32_t slot = t % i.fifo_num_pages;
    return (base16B + slot * i.fifo_page_size) << 4;  // CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT
}

// wait for n free pages in output CB `cb` — RISC poll against our OWN
// received mirror (mirrors llk_wait_for_free_tiles).
static inline void rvk_out_reserve(uint32_t cb, uint16_t my_received, uint16_t n) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    while ((uint16_t)((uint16_t)i.fifo_num_pages -
                      (uint16_t)(my_received -
                                 (uint16_t)reg_read((uint32_t)(uintptr_t)get_cb_tiles_acked_ptr((int)cb)))) < n) {
    }
}

// publish n tiles of output CB `cb` — sole producer. Fence + last-word
// read-back order the L1 data stores before the MMIO count store.
static inline void rvk_out_push(uint32_t cb, uint16_t& my_received, uint32_t last_tile_addr, uint16_t n) {
    asm volatile("fence" ::: "memory");
    (void)*(volatile uint32_t*)last_tile_addr;
    my_received += n;
    get_cb_tiles_received_ptr((int)cb)[0] = my_received;
}

#if !RVK_FORM_RATIONAL  // poly-cascade staging + evaluators (rational form: rvkr_* in rvv_rational.h)
// ---- init: stage the coefficient records (+ fallback tables), untimed ------
// No C float->int casts anywhere on this thread (scalar fcvt is RNE-locked):
// integer index math + scalar float compares/mults only (all RNE-safe).
// Returns the generic-path resolution metric (0 when the uniform path runs).
static inline uint32_t rvk_build_tables() {
    // (a) AoS coefficient records, zero-padded to the power-of-two record.
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        float* rec = (float*)(RVK_COEFF_BASE + (s << RVK_REC_SHIFT));
        for (uint32_t j = 0; j <= POLY_DEGREE; j++) {
            rec[j] = RVK_LUT_L1[(NUM_SEGMENTS + 1) + s * (POLY_DEGREE + 1) + j];
        }
        for (uint32_t j = POLY_DEGREE + 1; j < (RVK_REC_BYTES / 4); j++) {
            rec[j] = 0.0f;
        }
    }
    if (RVK_UNIFORM) {
        return 0;  // uniform fast path needs no index LUT and no fix-up tables
    }

    float* bnd_hi = (float*)(RVK_SCRATCH + RVK_BND_HI_OFF);
    float* bnd_lo = (float*)(RVK_SCRATCH + RVK_BND_LO_OFF);
    uint32_t* cell_tab = (uint32_t*)(RVK_SCRATCH + RVK_CELL_OFF);

    // (b) fix-up boundary tables with sticky-edge sentinels.
    //     bnd_hi[last] = quiet NaN: `x >= NaN` is false for EVERY x including
    //     +inf, so the last segment never increments past the table.
    //     bnd_lo[0] = -inf: `x < -inf` is false for every x, so segment 0
    //     never decrements. Written as bit patterns.
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        if (s + 1 < NUM_SEGMENTS) {
            bnd_hi[s] = RVK_LUT_L1[s + 1];
        } else {
            ((uint32_t*)bnd_hi)[s] = 0x7FC00000u;  // qNaN
        }
        if (s > 0) {
            bnd_lo[s] = RVK_LUT_L1[s];
        } else {
            ((uint32_t*)bnd_lo)[s] = 0xFF800000u;  // -inf
        }
    }
    for (uint32_t s = NUM_SEGMENTS; s < 1024; s++) {  // defensive fill for OOB seg reads
        ((uint32_t*)bnd_hi)[s] = 0x7FC00000u;
        ((uint32_t*)bnd_lo)[s] = 0xFF800000u;
    }

    // (c) cell table: segment of each cell's LEFT edge (>=: a breakpoint
    //     belongs to its RIGHT segment, same as the production cascade).
    uint32_t seg = 0;
    for (uint32_t k = 0; k < 256; k++) {
        float left = RVK_GRID_LO + (float)(int)k * RVK_GRID_STEP;  // int->float exact for k<=255
        while (seg + 1 < NUM_SEGMENTS && left >= RVK_LUT_L1[seg + 1]) {
            seg++;
        }
        cell_tab[k] = seg;
    }

    // (d) resolution metric over 2-cell windows (absorbs the <=1-cell rounding
    //     fuzz of the vector rtz((x-lo)*inv_step) index vs these left edges),
    //     folding in the clamped top cells which must reach the last segment
    //     within one fix-up step.
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
// Per-lane building blocks. Conditionally-DEFINED macros (never #if inside a
// macro) so the 3-way hot loop and the 2-way tail share one spelling of every
// numeric step. All lanes run at e32m2 (vl = 8).
// ---------------------------------------------------------------------------
#if defined(BASIS_INPUT_ABS_X)
#define RVK_XEVAL(xe, xo) vfloat32m2_t xe = __riscv_vfsgnjx_vv_f32m2(xo, xo, vl)  // |x|
#else
#define RVK_XEVAL(xe, xo) vfloat32m2_t xe = xo
#endif

// uniform fast path index: off = clamp(rtz((xe - base) * inv_w), 0, S-1) << REC_SHIFT
#define RVK_UNIF_OFF(off, xe)                                                                            \
    vint32m2_t off##_i = __riscv_vfcvt_rtz_x_f_v_i32m2(                                                  \
        __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(xe, RVK_UNIF_BASE, vl), RVK_UNIF_INV_W, vl), vl);  \
    off##_i = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(off##_i, 0, vl), (int)(NUM_SEGMENTS - 1), vl); \
    vuint32m2_t off = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(off##_i), RVK_REC_SHIFT, vl)

// basis reconstruction / tails, in production epilogue order (see header)
#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
#define RVK_EP_MULABS(acc, xe) acc = __riscv_vfmul_vv_f32m2(acc, xe, vl)
#else
#define RVK_EP_MULABS(acc, xe) (void)0
#endif
#if defined(BASIS_MUL_SQRT_1_MINUS_ABS)
// sqrt_factored basis: acos(|x|) = sqrt(1-|x|) * P(|x|). Op-for-op RVV mirror
// of production basis_sqrt (piecewise_generic.cpp: magic-seed 0x5f1110a0
// rsqrt estimate from bits(s)>>1, quadratic polynomial refinement of y ~
// 1/sqrt(s), then one sqrt-form Newton step y = xy + 0.5*xy*(1 - s*y*y)).
// Deviations, per the base rounding stance (vfmadd/vfnmsac are true
// the same partially-fused MAD as sfpi's (28-bit truncated product; the earlier
// "single-rounding vs semi-sticky" contrast is withdrawn); ULP report is the
// arbiter): production's addexp(xy,-1) is rendered as an exact *0.5f multiply
// (the rvv_rr.h stance — identical for all normals; a denormal halved input
// is DAZ on both engines). c = -(y*xy) equals production's (-y)*xy bit-exactly
// (IEEE sign symmetry). s <= 0 lanes follow the same bit recipe as production
// (s=0 -> y=0 exactly; s<0 cannot occur for the acos domain |x| <= 1).
#define RVK_EP_MULSQRT(acc, xe)                                                                                    \
    do {                                                                                                           \
        vfloat32m2_t sq_s_ = __riscv_vfrsub_vf_f32m2(xe, 1.0f, vl); /* s = 1 - |x| */                              \
        vuint32m2_t sq_i_ = __riscv_vsrl_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(sq_s_), 1, vl);               \
        vfloat32m2_t sq_y_ = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vrsub_vx_u32m2(sq_i_, 0x5f1110a0u, vl));   \
        vfloat32m2_t sq_xy_ = __riscv_vfmul_vv_f32m2(sq_s_, sq_y_, vl);                                            \
        vfloat32m2_t sq_c_ = __riscv_vfmul_vv_f32m2(sq_y_, sq_xy_, vl);                                            \
        sq_c_ = __riscv_vfsgnjn_vv_f32m2(sq_c_, sq_c_, vl); /* c = -(y*xy) */                                      \
        vfloat32m2_t sq_in_ = __riscv_vfadd_vf_f32m2(sq_c_, 2.2533049f, vl);                                       \
        vfloat32m2_t sq_mid_ = __riscv_vfmadd_vv_f32m2(sq_in_, sq_c_, __riscv_vfmv_v_f_f32m2(2.2825186f, vl), vl); \
        sq_y_ = __riscv_vfmul_vv_f32m2(sq_y_, sq_mid_, vl);                                                        \
        sq_xy_ = __riscv_vfmul_vv_f32m2(sq_s_, sq_y_, vl);                                                         \
        vfloat32m2_t sq_om_ =                                                                                      \
            __riscv_vfnmsac_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), sq_y_, sq_xy_, vl); /* 1 - y*xy */          \
        vfloat32m2_t sq_hx_ = __riscv_vfmul_vf_f32m2(sq_xy_, 0.5f, vl);                                            \
        vfloat32m2_t sq_r_ = __riscv_vfmadd_vv_f32m2(sq_om_, sq_hx_, sq_xy_, vl);                                  \
        acc = __riscv_vfmul_vv_f32m2(acc, sq_r_, vl);                                                              \
    } while (0)
#else
#define RVK_EP_MULSQRT(acc, xe) (void)0
#endif
#if defined(BASIS_AFFINE_EVEN)
// inner = fma(SCALE, x_orig, BIAS); t1 = EVEN_SCALE*|x| (rounded mul, like the
// production sfpu_mad's A operand); acc = fma(acc, t1, inner). `vaffbias` is a
// loop-invariant vector constant captured from the enclosing scope.
#define RVK_EP_AFFINE_EVEN(acc, xo, xe)                                                      \
    do {                                                                                     \
        vfloat32m2_t inner_ = __riscv_vfmacc_vf_f32m2(vaffbias, BASIS_AFFINE_SCALE, xo, vl); \
        vfloat32m2_t t1_ = __riscv_vfmul_vf_f32m2(xe, BASIS_AFFINE_EVEN_SCALE, vl);          \
        acc = __riscv_vfmadd_vv_f32m2(acc, t1_, inner_, vl);                                 \
    } while (0)
#else
#define RVK_EP_AFFINE_EVEN(acc, xo, xe) (void)0
#endif
#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
// abs-space affine right tail (odd_cubic_factored signed-abs basis, e.g.
// tanhshrink: |x| - 1 above the fit domain). Production reference is
// piecewise_generic_specialized.cpp: AFTER the affine_even slot, BEFORE
// clamp/copysgn — v_if(x_eval > THRESHOLD) result = A*x_eval + B. The A*|x|+B
// step is ONE vfmadd here where sfpi issues one semi-sticky (both are the same
// partially-fused MAD; the difference is issue/order, not rounding)
// SFPMAD (documented base-kernel FMA stance; ULP report is the arbiter). A
// NaN lane fails the compare and keeps the cascade value, like the SFPU v_if.
// `vrtailb` is a loop-invariant vector constant from RVK_DECL_EPILOGUE_CONSTS.
#define RVK_EP_RTAIL_ABS(acc, xe)                                                             \
    do {                                                                                      \
        vbool16_t mra_ = __riscv_vmfgt_vf_f32m2_b16(xe, BASIS_RIGHT_TAIL_ABS_THRESHOLD, vl);  \
        vfloat32m2_t tra_ = __riscv_vfmadd_vf_f32m2(xe, BASIS_RIGHT_TAIL_ABS_A, vrtailb, vl); \
        acc = __riscv_vmerge_vvm_f32m2(acc, tra_, mra_, vl);                                  \
    } while (0)
#else
#define RVK_EP_RTAIL_ABS(acc, xe) (void)0
#endif
#if defined(BASIS_CLAMP_MAX)
#define RVK_EP_CLAMP(acc) acc = __riscv_vfmin_vf_f32m2(acc, BASIS_CLAMP_MAX_VALUE, vl)
#else
#define RVK_EP_CLAMP(acc) (void)0
#endif
#if defined(BASIS_POST_SIGN_X)
#define RVK_EP_SIGN(acc, xo) acc = __riscv_vfsgnj_vv_f32m2(acc, xo, vl)
#else
#define RVK_EP_SIGN(acc, xo) (void)0
#endif
#if defined(BASIS_POST_REFLECT_PI)
// sqrt_factored acos negative half: acos(-|x|) = pi - acos(|x|), applied on
// x_orig < 0 exactly like the production v_if (NaN compares false -> the lane
// keeps the unreflected value; x_orig == -0.0f compares false, as on SFPU).
// BASIS_PI_VALUE is the codegen-emitted constexpr float.
#define RVK_EP_REFLECT(acc, xo)                                              \
    do {                                                                     \
        vbool16_t mr_ = __riscv_vmflt_vf_f32m2_b16(xo, 0.0f, vl);            \
        vfloat32m2_t tr_ = __riscv_vfrsub_vf_f32m2(acc, BASIS_PI_VALUE, vl); \
        acc = __riscv_vmerge_vvm_f32m2(acc, tr_, mr_, vl);                   \
    } while (0)
#else
#define RVK_EP_REFLECT(acc, xo) (void)0
#endif
#if defined(BASIS_LEFT_TAIL_ZERO)
#define RVK_EP_LTAIL(acc, xo)                                                               \
    do {                                                                                    \
        vbool16_t mz_ = __riscv_vmflt_vf_f32m2_b16(xo, BASIS_LEFT_TAIL_ZERO_THRESHOLD, vl); \
        acc = __riscv_vfmerge_vfm_f32m2(acc, 0.0f, mz_, vl);                                \
    } while (0)
#else
#define RVK_EP_LTAIL(acc, xo) (void)0
#endif
#if defined(BASIS_RIGHT_TAIL_IDENTITY)
#define RVK_EP_RTAIL(acc, xo)                                                                    \
    do {                                                                                         \
        vbool16_t mi_ = __riscv_vmfgt_vf_f32m2_b16(xo, BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD, vl); \
        acc = __riscv_vmerge_vvm_f32m2(acc, xo, mi_, vl);                                        \
    } while (0)
#else
#define RVK_EP_RTAIL(acc, xo) (void)0
#endif
// The domain-action finalize is a pass-through when TT_DOMAIN_ACTION_PROGRAM is
// absent (domain_actions_rvv.h:61 just returns `result`), but calling it still
// NAMES xo -- and RVK_EP_XO_RELOAD below declares xo only for the BASIS_* /
// TT_DOMAIN_ACTION_PROGRAM set. For a config in neither set (comp_cos_p4c_s808,
// comp_tanhshrink_p7c_s512, comp_leaky_relu_exact) that is a compile error:
//   piecewise_rvv.cpp:1015: error: 'xoA' was not declared in this scope
// which contradicts this file's own comment that "when no epilogue step needs
// x_orig the macro declares nothing and the epilogue ignores its xo argument".
// Wrapping it like every other RVK_EP_* step restores that invariant. Behaviour
// is identical either way: with the program defined it is the same call, without
// it the call was already a no-op.
#if defined(TT_DOMAIN_ACTION_PROGRAM)
#define RVK_EP_DOMAIN(acc, xo) acc = tt_rvv_finalize_domain_actions(xo, acc, vl)
#else
#define RVK_EP_DOMAIN(acc, xo) (void)0
#endif
#define RVK_EPILOGUE(acc, xo, xe)        \
    do {                                 \
        RVK_EP_MULABS(acc, xe);          \
        RVK_EP_MULSQRT(acc, xe);         \
        RVK_EP_AFFINE_EVEN(acc, xo, xe); \
        RVK_EP_RTAIL_ABS(acc, xe);       \
        RVK_EP_CLAMP(acc);               \
        RVK_EP_SIGN(acc, xo);            \
        RVK_EP_REFLECT(acc, xo);         \
        RVK_EP_LTAIL(acc, xo);           \
        RVK_EP_RTAIL(acc, xo);           \
        RVK_EP_DOMAIN(acc, xo);          \
    } while (0)

// The 3-way uniform hot loop cannot afford x_orig live across the Horner
// chain (3 lanes x {xo,xe,off,acc,cj} + the bias constant > 16 m2 registers —
// measured spills). When (and only when) the epilogue consumes x_orig, RELOAD
// it from L1 right before the epilogue (unit-stride vle32 is cheap; the value
// is bit-identical to the original load). When no epilogue step needs x_orig
// the macro declares nothing and the epilogue ignores its xo argument.
#if defined(BASIS_AFFINE_EVEN) || defined(BASIS_POST_SIGN_X) || defined(BASIS_POST_REFLECT_PI) || \
    defined(BASIS_LEFT_TAIL_ZERO) || defined(BASIS_RIGHT_TAIL_IDENTITY) || defined(TT_DOMAIN_ACTION_PROGRAM)
// The empty asm memory clobber pins the reload AFTER the Horner chain: without
// it GCC folds the reload into the top-of-loop load, hoists the epilogue's
// x_orig consumers up, and spills them across the Horner (measured).
#define RVK_EP_XO_RELOAD(xo2, p)   \
    asm volatile("" ::: "memory"); \
    vfloat32m2_t xo2 = __riscv_vle32_v_f32m2(p, vl)
#else
#define RVK_EP_XO_RELOAD(xo2, p) (void)0
#endif

// Declare the loop-invariant epilogue constants (referenced by RVK_EPILOGUE).
#if defined(BASIS_AFFINE_EVEN)
#define RVK_DECL_EPC_AFFINE() vfloat32m2_t vaffbias = __riscv_vfmv_v_f_f32m2(BASIS_AFFINE_BIAS, vl)
#else
#define RVK_DECL_EPC_AFFINE() (void)0
#endif
#if defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
#define RVK_DECL_EPC_RTAILABS() vfloat32m2_t vrtailb = __riscv_vfmv_v_f_f32m2(BASIS_RIGHT_TAIL_ABS_B, vl)
#else
#define RVK_DECL_EPC_RTAILABS() (void)0
#endif
#define RVK_DECL_EPILOGUE_CONSTS() \
    RVK_DECL_EPC_AFFINE();         \
    RVK_DECL_EPC_RTAILABS()

// ---- UNIFORM fast path: 1024 elements, 128 chunks @ e32m2 (vl=8), 3-way
// interleave (42 x {A,B,C}) + one 2-way {A,B} tail. The polynomial is always
// evaluated at the RAW x_eval (never a clamped copy).
static inline void rvk_eval_tile_uniform(const float* x, float* yout) {
    const float* ctab = (const float*)RVK_COEFF_BASE;
    size_t vl = __riscv_vsetvl_e32m2(8);
    RVK_DECL_EPILOGUE_CONSTS();

#if RVK_ILV == 1
    // Serial reference (A/B knob): one chunk chain per iteration. Identical
    // per-chunk op sequence -> bit-exact vs the interleaved shapes.
    for (int c = 0; c < 128; c++) {
        vfloat32m2_t xldA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        RVK_XEVAL(xeA, xldA);
        RVK_UNIF_OFF(offA, xeA);
        vfloat32m2_t accA = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offA, vl);
#pragma GCC unroll 17
        for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
            vfloat32m2_t cjA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, xeA, cjA, vl);
        }
        RVK_EP_XO_RELOAD(xoA, x + c * 8);
        RVK_EPILOGUE(accA, xoA, xeA);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
    }
#elif RVK_ILV == 2
    // 2-way A/B knob variant.
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xldA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xldB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        RVK_XEVAL(xeA, xldA);
        RVK_XEVAL(xeB, xldB);
        RVK_UNIF_OFF(offA, xeA);
        RVK_UNIF_OFF(offB, xeB);
        vfloat32m2_t accA = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offA, vl);
        vfloat32m2_t accB = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offB, vl);
#pragma GCC unroll 17
        for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
            vfloat32m2_t cjA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);
            vfloat32m2_t cjB = __riscv_vluxei32_v_f32m2(ctab + j, offB, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, xeA, cjA, vl);
            accB = __riscv_vfmadd_vv_f32m2(accB, xeB, cjB, vl);
        }
        RVK_EP_XO_RELOAD(xoA, x + c * 8);
        RVK_EP_XO_RELOAD(xoB, x + c * 8 + 8);
        RVK_EPILOGUE(accA, xoA, xeA);
        RVK_EPILOGUE(accB, xoB, xeB);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, accB, vl);
    }
#elif RVK_ILV == 4
    // 4-way A/B knob variant (register-pressure probe: 4 x {xe,off,acc} = 12
    // m2 groups + short-lived cj gathers — audit the disassembly for spills
    // before trusting its numbers).
    for (int c = 0; c < 128; c += 4) {
        vfloat32m2_t xldA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xldB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        vfloat32m2_t xldC = __riscv_vle32_v_f32m2(x + c * 8 + 16, vl);
        vfloat32m2_t xldD = __riscv_vle32_v_f32m2(x + c * 8 + 24, vl);
        RVK_XEVAL(xeA, xldA);
        RVK_XEVAL(xeB, xldB);
        RVK_XEVAL(xeC, xldC);
        RVK_XEVAL(xeD, xldD);
        RVK_UNIF_OFF(offA, xeA);
        RVK_UNIF_OFF(offB, xeB);
        RVK_UNIF_OFF(offC, xeC);
        RVK_UNIF_OFF(offD, xeD);
        vfloat32m2_t accA = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offA, vl);
        vfloat32m2_t accB = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offB, vl);
        vfloat32m2_t accC = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offC, vl);
        vfloat32m2_t accD = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offD, vl);
#pragma GCC unroll 17
        for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
            vfloat32m2_t cjA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);
            vfloat32m2_t cjB = __riscv_vluxei32_v_f32m2(ctab + j, offB, vl);
            vfloat32m2_t cjC = __riscv_vluxei32_v_f32m2(ctab + j, offC, vl);
            vfloat32m2_t cjD = __riscv_vluxei32_v_f32m2(ctab + j, offD, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, xeA, cjA, vl);
            accB = __riscv_vfmadd_vv_f32m2(accB, xeB, cjB, vl);
            accC = __riscv_vfmadd_vv_f32m2(accC, xeC, cjC, vl);
            accD = __riscv_vfmadd_vv_f32m2(accD, xeD, cjD, vl);
        }
        RVK_EP_XO_RELOAD(xoA, x + c * 8);
        RVK_EP_XO_RELOAD(xoB, x + c * 8 + 8);
        RVK_EP_XO_RELOAD(xoC, x + c * 8 + 16);
        RVK_EP_XO_RELOAD(xoD, x + c * 8 + 24);
        RVK_EPILOGUE(accA, xoA, xeA);
        RVK_EPILOGUE(accB, xoB, xeB);
        RVK_EPILOGUE(accC, xoC, xeC);
        RVK_EPILOGUE(accD, xoD, xeD);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, accB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 16, accC, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 24, accD, vl);
    }
#else
    for (int c = 0; c < 126; c += 3) {
        vfloat32m2_t xldA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xldB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        vfloat32m2_t xldC = __riscv_vle32_v_f32m2(x + c * 8 + 16, vl);
        RVK_XEVAL(xeA, xldA);
        RVK_XEVAL(xeB, xldB);
        RVK_XEVAL(xeC, xldC);
        RVK_UNIF_OFF(offA, xeA);
        RVK_UNIF_OFF(offB, xeB);
        RVK_UNIF_OFF(offC, xeC);

        // straight full-degree Horner, high-to-low, fused vfmadd
        vfloat32m2_t accA = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offA, vl);
        vfloat32m2_t accB = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offB, vl);
        vfloat32m2_t accC = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offC, vl);
#pragma GCC unroll 17
        for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
            vfloat32m2_t cjA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);
            vfloat32m2_t cjB = __riscv_vluxei32_v_f32m2(ctab + j, offB, vl);
            vfloat32m2_t cjC = __riscv_vluxei32_v_f32m2(ctab + j, offC, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, xeA, cjA, vl);
            accB = __riscv_vfmadd_vv_f32m2(accB, xeB, cjB, vl);
            accC = __riscv_vfmadd_vv_f32m2(accC, xeC, cjC, vl);
        }

        RVK_EP_XO_RELOAD(xoA, x + c * 8);
        RVK_EP_XO_RELOAD(xoB, x + c * 8 + 8);
        RVK_EP_XO_RELOAD(xoC, x + c * 8 + 16);
        RVK_EPILOGUE(accA, xoA, xeA);
        RVK_EPILOGUE(accB, xoB, xeB);
        RVK_EPILOGUE(accC, xoC, xeC);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, accB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 16, accC, vl);
    }
    {
        constexpr int c = 126;  // 2-way tail: chunks 126, 127
        vfloat32m2_t xldA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xldB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        RVK_XEVAL(xeA, xldA);
        RVK_XEVAL(xeB, xldB);
        RVK_UNIF_OFF(offA, xeA);
        RVK_UNIF_OFF(offB, xeB);
        vfloat32m2_t accA = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offA, vl);
        vfloat32m2_t accB = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offB, vl);
#pragma GCC unroll 17
        for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
            vfloat32m2_t cjA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);
            vfloat32m2_t cjB = __riscv_vluxei32_v_f32m2(ctab + j, offB, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, xeA, cjA, vl);
            accB = __riscv_vfmadd_vv_f32m2(accB, xeB, cjB, vl);
        }
        RVK_EP_XO_RELOAD(xoA, x + c * 8);
        RVK_EP_XO_RELOAD(xoB, x + c * 8 + 8);
        RVK_EPILOGUE(accA, xoA, xeA);
        RVK_EPILOGUE(accB, xoB, xeB);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, accB, vl);
    }
#endif  // RVK_ILV
}

// ---- GENERIC fallback (non-uniform boundaries): the piecewise_hybrid
// evaluator with 1024-segment fix-up tables. 2-way interleave (proven no-spill
// shape); only engaged for non-uniform CSVs, which the RVV sweep never emits.
static inline void rvk_eval_tile_generic(const float* x, float* yout) {
    const float* ctab = (const float*)RVK_COEFF_BASE;
    const uint32_t* cell_tab = (const uint32_t*)(RVK_SCRATCH + RVK_CELL_OFF);
    const float* bnd_hi = (const float*)(RVK_SCRATCH + RVK_BND_HI_OFF);
    const float* bnd_lo = (const float*)(RVK_SCRATCH + RVK_BND_LO_OFF);

    size_t vl = __riscv_vsetvl_e32m2(8);
    vuint32m2_t vzero = __riscv_vmv_v_x_u32m2(0, vl);
    RVK_DECL_EPILOGUE_CONSTS();

    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xoA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xoB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
        RVK_XEVAL(xeA, xoA);
        RVK_XEVAL(xeB, xoB);
        // cell = clamp(rtz((xe - lo) * inv_step), 0, 255) — vector rtz only;
        // the raw xe (NOT a clamped copy) feeds the polynomial below.
        vint32m2_t iA = __riscv_vfcvt_rtz_x_f_v_i32m2(
            __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(xeA, RVK_GRID_LO, vl), RVK_GRID_INV_STEP, vl), vl);
        vint32m2_t iB = __riscv_vfcvt_rtz_x_f_v_i32m2(
            __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(xeB, RVK_GRID_LO, vl), RVK_GRID_INV_STEP, vl), vl);
        iA = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(iA, 0, vl), 255, vl);
        iB = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(iB, 0, vl), 255, vl);
        vuint32m2_t cbA = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(iA), 2, vl);
        vuint32m2_t cbB = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(iB), 2, vl);
        vuint32m2_t segA = __riscv_vluxei32_v_u32m2(cell_tab, cbA, vl);
        vuint32m2_t segB = __riscv_vluxei32_v_u32m2(cell_tab, cbB, vl);

        // Bidirectional fix-up against the REAL breakpoints, unrolled
        // RVK_FIXUP_STEPS times (compile-time metric — see the header block).
        // Each step moves at most one segment toward the containing one and
        // the sticky sentinels pin both edges, so RVK_FIXUP_STEPS steps are
        // EXACT for any cell hint the 256-cell table can produce. One step
        // compiles to the byte-identical sequence this loop body always had.
#pragma GCC unroll 8
        for (uint32_t fs = 0; fs < RVK_FIXUP_STEPS; fs++) {
            vuint32m2_t sbA = __riscv_vsll_vx_u32m2(segA, 2, vl);
            vuint32m2_t sbB = __riscv_vsll_vx_u32m2(segB, 2, vl);
            vfloat32m2_t bhA = __riscv_vluxei32_v_f32m2(bnd_hi, sbA, vl);
            vfloat32m2_t bhB = __riscv_vluxei32_v_f32m2(bnd_hi, sbB, vl);
            vfloat32m2_t blA = __riscv_vluxei32_v_f32m2(bnd_lo, sbA, vl);
            vfloat32m2_t blB = __riscv_vluxei32_v_f32m2(bnd_lo, sbB, vl);
            vbool16_t upA = __riscv_vmfge_vv_f32m2_b16(xeA, bhA, vl);
            vbool16_t upB = __riscv_vmfge_vv_f32m2_b16(xeB, bhB, vl);
            vbool16_t dnA = __riscv_vmflt_vv_f32m2_b16(xeA, blA, vl);
            vbool16_t dnB = __riscv_vmflt_vv_f32m2_b16(xeB, blB, vl);
            segA = __riscv_vadd_vv_u32m2(segA, __riscv_vmerge_vxm_u32m2(vzero, 1, upA, vl), vl);
            segB = __riscv_vadd_vv_u32m2(segB, __riscv_vmerge_vxm_u32m2(vzero, 1, upB, vl), vl);
            segA = __riscv_vsub_vv_u32m2(segA, __riscv_vmerge_vxm_u32m2(vzero, 1, dnA, vl), vl);
            segB = __riscv_vsub_vv_u32m2(segB, __riscv_vmerge_vxm_u32m2(vzero, 1, dnB, vl), vl);
        }
        vuint32m2_t offA = __riscv_vsll_vx_u32m2(segA, RVK_REC_SHIFT, vl);
        vuint32m2_t offB = __riscv_vsll_vx_u32m2(segB, RVK_REC_SHIFT, vl);

        vfloat32m2_t accA = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offA, vl);
        vfloat32m2_t accB = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offB, vl);
#pragma GCC unroll 17
        for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
            vfloat32m2_t cjA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);
            vfloat32m2_t cjB = __riscv_vluxei32_v_f32m2(ctab + j, offB, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, xeA, cjA, vl);
            accB = __riscv_vfmadd_vv_f32m2(accB, xeB, cjB, vl);
        }

        RVK_EPILOGUE(accA, xoA, xeA);
        RVK_EPILOGUE(accB, xoB, xeB);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, accB, vl);
    }
}
#endif  // !RVK_FORM_RATIONAL
#endif  // TRISC_PACK

// ============================================================================
// Kernel entry. Unpack/math threads are COMPLETE NO-OPS (they never touch the
// c_0/c_16 counters, DEST, or any Tensix state — the raw pack-thread protocol
// is unraced by construction). The pack thread computes every tile.
// ============================================================================
#if defined(TRISC_PACK) && RVK_FORM_RATIONAL
// Route kernel_main's poly-named hooks onto the rational form's interface.
// The poly definitions of these names are compiled out above, so plain macro
// aliases are safe; hdr[15] (the POLY_DEGREE slot) reports (ND << 8) | DD.
#define RVK_UNIFORM RVKR_UNIFORM
#define RVK_TABLE_BYTES RVKR_TABLE_BYTES
#define RVK_REC_SHIFT RVKR_REC_SHIFT
#define RVK_FIXUP_STEPS RVKR_FIXUP_STEPS
#define POLY_DEGREE RVKR_HDR_DEGREE
#define RVK_UNIF_BASE RVKR_IDX_BASE_F
#define RVK_UNIF_INV_W RVKR_IDX_SCALE_F
#define RVK_GRID_LO RVKR_IDX_BASE_F
#define RVK_GRID_INV_STEP RVKR_IDX_SCALE_F
#define RVK_GRID_HI RVKR_GRID_HI_F
#define rvk_build_tables rvkr_build_tables
#define rvk_eval_tile_uniform rvkr_eval_tile
#define rvk_eval_tile_generic rvkr_eval_tile
#endif  // TRISC_PACK && RVK_FORM_RATIONAL

#if defined(TRISC_PACK) && RVK_FORM_COMP
// Route kernel_main onto the compensated form. The base staging is reused
// verbatim (packed records ARE plain AoS rows); only the eval dispatch and
// the header-debug floats change. RVK_UNIFORM -> true keeps the generic-path
// error plumbing dead (the compensated index map is compile-time proven and
// needs no fix-up); hdr[7] is rewritten to the compensated path flag below.
#define RVK_UNIFORM true
#define rvk_eval_tile_uniform rvkc_eval_tile
#define RVK_UNIF_BASE (RVKC_UNIFORM ? RVKC_BASE_F : __builtin_bit_cast(float, RVKC_BITS0))
#define RVK_UNIF_INV_W (RVKC_UNIFORM ? RVKC_INV_W : (float)RVKC_BIT_SHIFT)
#endif  // TRISC_PACK && RVK_FORM_COMP

void kernel_main() {
#ifndef TRISC_PACK
    // unpack + math: intentionally empty pass-through.
#else
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
#if RVK_FUSE_GRAD
    constexpr auto cb_grad = tt::CBIndex::c_1;  // one grad tile per input tile (reader.cpp)
#endif

    volatile uint32_t* hdr = (volatile uint32_t*)RVK_SCRATCH;
    hdr[0] = 0;  // clear magic first: a partial header is never mistaken for done
    hdr[5] = 0;
    hdr[6] = n_tiles;
    hdr[7] = RVK_UNIFORM ? 1u : 0u;
    hdr[8] = RVK_TABLE_BYTES;
    hdr[9] = 0;
    hdr[10] = 0;
    hdr[11] = 0x10;  // breadcrumb: pack entry
    hdr[12] = 0xFFFFFFFFu;
    hdr[14] = NUM_SEGMENTS;
    hdr[15] = POLY_DEGREE;
    hdr[16] = RVK_REC_SHIFT;
#if RVK_BW_IO_ACTIVE
    hdr[20] = RVK_BW_FEATURES;  // bit0 bf16, bit1 grad, bit2 asym (only written when active,
                                // keeping plain-fp32 builds byte-identical to the base kernel)
#endif
    if (RVK_UNIFORM) {
        *(volatile float*)(RVK_SCRATCH + 17 * 4) = RVK_UNIF_BASE;
        *(volatile float*)(RVK_SCRATCH + 18 * 4) = RVK_UNIF_INV_W;
    } else {
        *(volatile float*)(RVK_SCRATCH + 17 * 4) = RVK_GRID_LO;
        *(volatile float*)(RVK_SCRATCH + 18 * 4) = RVK_GRID_INV_STEP;
    }
    *(volatile float*)(RVK_SCRATCH + 19 * 4) = RVK_GRID_HI;

#if RVK_LOWERING_ACTIVE
    // Algebraic lowering: the LUT is bypassed — no coefficient staging, no
    // index LUT, no resolution metric (and no false 0xE0000001 from the
    // non-uniform boundaries of s3 lowering CSVs).
    hdr[7] = 2u;  // path flag: 2 = lowering (1 = uniform, 0 = generic)
    hdr[8] = 0;
    hdr[10] = 0;
#elif RVK_FORM_CLOSED_STRUCTURAL
    // Typed total closed form: all coefficients are schedule constants and
    // the ordinary cascade LUT is inert.
    hdr[7] = 5u;
    hdr[8] = 0;
    hdr[10] = 0;
#elif RVK_FORM_LOGODDS
    // Log-odds separable basis: all five coefficients are compile-time
    // constants — no staging, no index LUT, no resolution metric.
    hdr[7] = 4u;  // path flag: 4 = log-odds (1 uniform, 0 generic, 2 lowering, 3 compensated)
    hdr[8] = 0;
    hdr[10] = 0;
#else
    // Poly/rational cascades AND the range-reduced forms: full staging (the
    // REDUCE_TAN RR cascade reuses the staged AoS records; the standalone RR
    // forms keep the staging + resolution contract unchanged — their winner
    // CSVs are uniform, so the metric path never engages).
    uint32_t metric = rvk_build_tables();
    hdr[10] = metric;
    if (!RVK_UNIFORM && metric > RVK_FIXUP_STEPS) {
        // Generic-path index-LUT resolution violation. The fix-up depth is
        // compiled from the constexpr twin of this exact metric computation,
        // so this can only fire on staging corruption — kept as a defensive
        // cross-check. LOUD failure — the runner must abort on this word. We
        // still stream tiles below so the reader/writer never deadlock.
        hdr[9] = RVK_ERR_INDEX_RESOLUTION;
    }
#endif
#if RVK_FORM_COMP
    // Compensated path: index map is compile-time proven (uniform-affine or
    // log2 bit-grid) — the generic-path metric is meaningless here.
    hdr[7] = 3u;  // path flag: 3 = compensated (1 uniform, 0 generic, 2 lowering)
    hdr[10] = 0;
#endif
    hdr[11] = 0x20;  // breadcrumb: tables staged

    uint16_t my_acked_in = 0;  // local mirror of c_0 acked (sole acker; reg zeroed at launch)
#if RVK_FUSE_GRAD
    uint16_t my_acked_grad = 0;  // local mirror of c_1 acked (sole acker, same protocol)
#endif
    uint16_t my_recv_out = 0;  // local mirror of c_16 received (sole producer)
    uint32_t done = 0;

    uint64_t t_first = get_timestamp();
    hdr[1] = (uint32_t)t_first;
    hdr[2] = (uint32_t)(t_first >> 32);
    hdr[11] = 0x30;  // breadcrumb: in loop

    for (uint32_t t = 0; t < n_tiles; t++) {
        hdr[12] = t;
        rvk_in_wait(cb_in, my_acked_in, 1);
#if RVK_FUSE_GRAD
        rvk_in_wait(cb_grad, my_acked_grad, 1);  // reader pushes c_0/c_1 together
#endif
        invalidate_l1_cache();  // BH: L1 data reads after an MMIO count poll
        rvk_out_reserve(cb_out, my_recv_out, 1);
        uint32_t src = rvk_tile_addr(cb_in, t);
        uint32_t dst = rvk_tile_addr(cb_out, t);
#if RVK_BW_IO_ACTIVE
        const uint32_t cb_src = src;  // real CB pages (bf16-sized in bf16 mode)
        const uint32_t cb_dst = dst;
        src = rvk_bw_stage_in(cb_src);  // bf16: exact widen into fp32 XSTAGE
        dst = rvk_bw_eval_dst(cb_dst);  // bf16: fp32 YSTAGE for the evaluator
#endif
#if RVK_LOWERING_ACTIVE
        rvk_eval_tile_lowering((const float*)src, (float*)dst);  // rvv_forms/rvv_lowering.h
#elif RVK_FORM_CLOSED_STRUCTURAL
        rvk_eval_tile_closed_structural((const float*)src, (float*)dst);
#elif RVK_FORM_LOGODDS
        rvk_eval_tile_logodds((const float*)src, (float*)dst);  // rvv_forms/rvv_logodds.h
#elif RVK_FORM_RR
        rvk_rr_eval_tile((const float*)src, (float*)dst);  // rvv_forms/rvv_rr.h
#else
        // poly cascade — or, via the alias shim above, the rational cascade
        if (RVK_UNIFORM) {
            rvk_eval_tile_uniform((const float*)src, (float*)dst);
        } else {
            rvk_eval_tile_generic((const float*)src, (float*)dst);
        }
#endif
#if RVK_BW_IO_ACTIVE && RVK_FUSE_GRAD
        rvk_bw_finish_tile((const float*)src, (float*)dst, cb_dst, rvk_tile_addr(cb_grad, t));
        dst = cb_dst;  // the push fence must read back the REAL output page
#elif RVK_BW_IO_ACTIVE
        rvk_bw_finish_tile((const float*)src, (float*)dst, cb_dst);
        dst = cb_dst;
#endif
        rvk_in_pop(cb_in, my_acked_in, 1);  // after the tile's last RVV load
#if RVK_FUSE_GRAD
        rvk_in_pop(cb_grad, my_acked_grad, 1);  // after finish_tile's last grad load
#endif
        rvk_out_push(cb_out, my_recv_out, dst, 1);
        done++;
        hdr[5] = done;
    }
    hdr[11] = 0x40;  // breadcrumb: loop done

    uint64_t t_last = get_timestamp();
    hdr[3] = (uint32_t)t_last;
    hdr[4] = (uint32_t)(t_last >> 32);
    hdr[13] = (uint32_t)(t_last - t_first);
    hdr[11] = 0x50;      // breadcrumb: header complete
    hdr[0] = RVK_MAGIC;  // magic LAST
#endif  // TRISC_PACK
}
