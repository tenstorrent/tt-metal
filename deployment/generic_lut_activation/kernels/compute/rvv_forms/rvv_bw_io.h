// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// rvv_forms/rvv_bw_io.h — bf16 tile I/O + fused-grad (_bw) + asymptotic-factor
// epilogue for the RVV-only kernel piecewise_rvv.cpp (magic 0xC0FFEE30).
// =============================================================================
//
// WHAT THIS UNLOCKS (the _bw bf16 board): today the base RVV kernel #errors on
// USE_BF16, FUSE_GRAD_MUL and every ASYMPTOTIC_FACTOR_*. This header supplies
// all three as an I/O LAYER around the UNCHANGED fp32 evaluators — the form
// dispatch (uniform / generic cascade, and the parallel lanes' lowering /
// rational / rr forms) is never touched:
//
//   input CB page --[stage-in: bf16 DAZ -> fp32 widen, or pass-through]-->
//     fp32 eval input --[UNCHANGED form evaluator]--> fp32 eval output
//     --[finish: asymptotic apply -> grad multiply -> bf16 RNE pack]-->
//   output CB page
//
// (1) BF16 TILE I/O (USE_BF16). The host runs --precision bf16 with
//     tt::DataFormat::Float16_b CBs: 1024 x 2B = 2KB pages (plain bfloat16,
//     no shared exponents; generic_lut_activation.cpp tile_size_bytes =
//     sizeof(bfloat16) * 1024, set_page_size likewise; USE_BF16 reaches every
//     compute TU via compute_defines). Compute stays fp32 INTERNALLY exactly
//     like the SFPU path (piecewise_generic.cpp evaluates in fp32 registers
//     and only converts at the DST store).
//       in : apply the declared input DAZ at the raw-bf16 boundary (each
//            subnormal becomes a same-sign zero), then zero-extend, << 16,
//            and reinterpret f32. Exact zeros retain their sign.
//       out: fp32 -> bf16 matches the production convert<vFloat16b>(y,
//            RoundMode::Nearest) contract (piecewise_generic.cpp: "SFPSTORE
//            narrows fp32->bf16 by truncation (RTZ) in hardware; rounding here
//            (sfpstochrnd RND_EVEN) makes the already-bf16 value lossless
//            under SFPSTORE") — i.e. the produced bf16 bits are the RNE
//            rounding of the fp32 result. Implemented bit-exactly in integer
//            ops: h = (b + 0x7FFF + ((b >> 16) & 1)) >> 16. The add trick is
//            RNE-correct for every non-NaN input including +/-0 (sign kept),
//            +/-inf (mantissa zero, no carry) and max-finite (carries to inf,
//            which IS the RNE result). It can corrupt a NaN whose top mantissa
//            bits are all ones (carry into the exponent), so NaN lanes are
//            merged to (b >> 16) | 0x0040: sign + payload top bits preserved,
//            quiet bit forced so truncation can never yield inf — NaN stays
//            NaN, as the SFPU convert produces. The declared output FTZ is
//            then applied to the rounded bf16 encoding, again retaining the
//            sign of zero. No C float<->int casts anywhere (vector integer
//            ops only).
//
// (2) FUSED UNARY-BACKWARD MULTIPLY (FUSE_GRAD_MUL). Production contract
//     (reader.cpp / piecewise_generic.cpp): the standard reader streams one
//     grad tile into CB c_1 per input tile (same 2-deep CB, same data format
//     as c_0), the compute epilogue forms y = f'(x) * grad and the compute
//     side pops c_0 and c_1 together. RVV-only version: the pack thread is
//     the SOLE acker of c_1, consumed in lockstep with c_0 through the same
//     raw stream-register protocol (wait -> use -> pop; the unpack/math TUs
//     stay complete no-ops). The multiply happens here, POST-epilogue on the
//     fp32 result, BEFORE the bf16 repack:
//         bf16: y_bf16 = RNE(f'(x)_fp32 * widen(grad_bf16))   [ONE rounding]
//     DOCUMENTED DIVERGENCE from the SFPU path, which double-rounds
//     (bf16 DST rows: bf16(bf16(f'(x)) * g), piecewise_generic.cpp
//     fuse_grad_mul comment). Single rounding is never less accurate; the
//     harness ULP report is the arbiter (base-kernel stance). The grad == 1
//     regression oracle SURVIVES: y * 1.0f is exact in fp32, so a fused dump
//     is still byte-identical to the unfused dump of the same kernel.
//
// (3) ASYMPTOTIC FACTORING (gelu_bw class). run_csv.sh codegen emits
//     ASYMPTOTIC_FACTOR_{EXP_QUADRATIC | EXP_LINEAR | X_EXP_LINEAR | X |
//     QUADRATIC} + ASYMPTOTIC_EXP_ARG_SCALE / ASYMPTOTIC_SCALE /
//     [ASYMPTOTIC_NEGATE_OUTPUT] / ASYMPTOTIC_QUAD_ROOT_A/B and exactly one
//     region tag ASYMPTOTIC_REGION_{LEFT | RIGHT | ALL} (+ ASYMPTOTIC_BOUND).
//     Production semantics (piecewise_generic_specialized.cpp): inside the
//     region predicate on x_orig,
//         EXP_QUADRATIC: y = scale((y * exp((x*x)*ARG_SCALE)))
//         EXP_LINEAR   : y = scale((y * exp(x*ARG_SCALE)))
//         X_EXP_LINEAR : y = scale(((y * x) * exp(x*ARG_SCALE)))
//         X            : y = scale((y * x))
//         QUADRATIC    : y = scale((y * ((x-A)*(x-B))))
//     where scale(v) = v * ASYMPTOTIC_SCALE, then an EXACT sign flip when
//     ASYMPTOTIC_NEGATE_OUTPUT (production XORs the sign bit; vfsgnjn here).
//     Applied AFTER the basis epilogue and BEFORE the bf16 convert / grad
//     multiply — the exact production order (cascade -> BASIS_* ->
//     ASYMPTOTIC -> bf16 convert -> fuse_grad_mul).
//     The exp core mirrors production asymptotic_exp() (piecewise_generic.cpp
//     :2093) step for step: z = arg * INV_LN2; magic-number RNE round to k
//     (0x4B400000 trick — no scalar fcvt, integer subtract for k_int);
//     Cody-Waite r = arg + k*NEG_LN2_HI + k*NEG_LN2_LO; degree-5 Taylor
//     Horner; scale by 2^k via exponent-field integer add; and the SAME
//     underflow guard (biased_exp(p) + k <= 0 -> 0) that fixed the silicon
//     erf_bw wrap at |x| >= 9.4375. No overflow guard, exactly like
//     production ("the asymptotic region has known bounded input range").
//     NOTE(duplication audit, at integration time): rvv_forms/rvv_rr.h's exp
//     core is the EXPONENT-ALU exp2 mirror (production exp_hw_eval) — it is
//     parameterized by the EXP_HW_DEGREE/EXP_HW_* codegen macros of a
//     standalone exp CSV, which asymptotic adhocs do not define, and it
//     reproduces a DIFFERENT production reference than the asymptotic path
//     needs (production asymptotic tiles call asymptotic_exp(), not
//     exp_hw_eval). So this self-contained mirror of asymptotic_exp() is not
//     redundant with rvv_rr.h; if a shared general-purpose Cody-Waite exp
//     ever lands in rvv_forms/, fold this one into it and re-run the gelu_bw
//     A/B.
//
// ROUNDING STANCE (identical to the base kernel): the RVV engine's vfmadd /
// vfmul are IEEE-RNE single-rounding; SFPU byte identity is NOT claimed — the
// harness pure-ULP report vs golden is the arbiter, and the _bw board's gate
// is pure-ULP parity vs ttnn (tier_silicon_summary_bw.csv).
//
// L1 BUDGET: two 4KB fp32 staging tiles at 0x16C000 / 0x16D000 — directly
// above the coefficient window, whose end the base kernel pins with
// static_assert(RVK_COEFF_BASE + RVK_COEFF_CAP <= 0x16C000). Nothing new in
// LDM (~1.7KB free on TRISC2): all tables/constants are immediates or .text.
// Staging is only touched in bf16 mode; fp32 mode evaluates CB->CB as today.
//
// PERF NOTE (honest, unmeasured): stage-in + finish add two extra linear
// passes per tile (~2x 1024 elems of 1-3 op vector work + the asymptotic
// chain on asymptotic fits). Correctness-first unlock; fusing the conversions
// into the 3-way hot loop is follow-up work once the ULP gate is green.
//
// This header is self-contained: preprocessor detection is visible to ALL
// three TUs; vector code stays strictly inside TRISC_PACK (Zve32f exists only
// on the pack RISC). Include it from piecewise_rvv.cpp BEFORE the support-
// matrix #errors (same slot style as rvv_forms/rvv_lowering.h).
// =============================================================================

#ifndef RVV_FORMS_RVV_BW_IO_H_
#define RVV_FORMS_RVV_BW_IO_H_

// ---------------------------------------------------------------------------
// Feature detection (preprocessor, all TUs).
// ---------------------------------------------------------------------------
#if defined(USE_BF16)
#define RVK_IO_BF16 1
#else
#define RVK_IO_BF16 0
#endif

#if defined(FUSE_GRAD_MUL)
#define RVK_FUSE_GRAD 1
#else
#define RVK_FUSE_GRAD 0
#endif

// Asymptotic factoring is a POLY-CASCADE-family epilogue here. Rational
// adhocs can also carry ASYMPTOTIC_FACTOR_QUADRATIC (lgamma; run_csv.sh emits
// {asymptotic_macro} into the rational template too), but the rational form
// applies it INSIDE rvkr_eval_tile as its postcompose (rvv_forms/
// rvv_rational.h) — gating on the rational eval methods prevents a double
// apply. Range-reduced forms never co-emit asymptotic (production: "mutually
// exclusive with range reduction").
#if (                                                                                    \
    defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X) ||           \
    defined(ASYMPTOTIC_FACTOR_QUADRATIC)) &&                                             \
    !defined(EVAL_METHOD_RATIONAL_CASCADE) && !defined(EVAL_METHOD_ABS_DENOMINATOR_RATIONAL)
#define RVK_ASYM_ACTIVE 1
// Region sanity — mirror of piecewise_generic_specialized.cpp: codegen must
// have declared exactly one active region.
#if (defined(ASYMPTOTIC_REGION_LEFT) + defined(ASYMPTOTIC_REGION_RIGHT) + defined(ASYMPTOTIC_REGION_ALL)) != 1
#error "rvv_bw_io: asymptotic factor requires exactly one LEFT/RIGHT/ALL region"
#endif
#else
#define RVK_ASYM_ACTIVE 0
#endif

#if RVK_IO_BF16 || RVK_FUSE_GRAD || RVK_ASYM_ACTIVE
#define RVK_BW_IO_ACTIVE 1
#else
#define RVK_BW_IO_ACTIVE 0
#endif

// Debug feature word for the scratch header (hdr[20]); 0 when inactive.
#define RVK_BW_FEATURES ((RVK_IO_BF16 * 1u) | (RVK_FUSE_GRAD * 2u) | (RVK_ASYM_ACTIVE * 4u))

// Interleave depth for the two bf16 staging loops (stage_in / finish_tile).
// 1 = one 8-element chunk per iteration (the original, fully serial dependency
// chain: each vse32 waits on the widen right in front of it). 2 = two
// independent chunks issued per iteration so the chains overlap. Purely an
// issue-order change: every element sees the identical ops on identical data,
// so any RVK_BW_ILV is bit-exact with any other. Not a numerics knob.
#ifndef RVK_BW_ILV
#define RVK_BW_ILV 4
#endif
#if RVK_BW_ILV != 1 && RVK_BW_ILV != 2 && RVK_BW_ILV != 4
#error "RVK_BW_ILV must be 1, 2 or 4"
#endif

// bf16 RNE pack shape. 0 = the original mixed-SEW form (e32 rounding ops and
// e16 narrow/or/merge interleaved, which costs a vsetvli at every SEW switch).
// 1 = do the NaN select at e32 and narrow once, so the pack has a single e16
// instruction. Bit-identical (exhaustively proven), one fewer vector op.
#ifndef RVK_BW_PACK_E32
#define RVK_BW_PACK_E32 1
#endif

// ---------------------------------------------------------------------------
// Pack-thread implementation.
// ---------------------------------------------------------------------------
#if RVK_BW_IO_ACTIVE && defined(TRISC_PACK)

#include <riscv_vector.h>

// fp32 staging tiles (bf16 mode only). The base kernel's static_asserts pin
// the coefficient window to [0x164000, 0x16C000); BH MEM_L1_SIZE is 0x180000
// and the deployment host maps no top-down L1 buffers (same argument as the
// silicon-proven 0x160000 scratch block).
static constexpr uint32_t RVK_BW_XSTAGE = 0x16C000;  // fp32 input tile (4KB)
static constexpr uint32_t RVK_BW_YSTAGE = 0x16D000;  // fp32 eval output tile (4KB)
static_assert(RVK_BW_YSTAGE + 4096 <= 0x180000, "staging tiles must fit BH MEM_L1_SIZE");

// The declared RVV bf16 target applies DAZ before the evaluator and FTZ after
// the final RNE conversion.  Do both transformations while the bf16 encoding
// is still explicit so that the sign of zero is never lost.
static inline vuint16m1_t rvk_bw_flush_bf16_subnormals(vuint16m1_t h, size_t vl) {
    vuint16m1_t exponent = __riscv_vand_vx_u16m1(h, 0x7F80u, vl);
    vuint16m1_t fraction = __riscv_vand_vx_u16m1(h, 0x007Fu, vl);
    vbool16_t subnormal = __riscv_vmand_mm_b16(
        __riscv_vmseq_vx_u16m1_b16(exponent, 0u, vl), __riscv_vmsne_vx_u16m1_b16(fraction, 0u, vl), vl);
#if defined(TT_RVV_PRESERVE_POS_SUBNORMAL)
    subnormal =
        __riscv_vmand_mm_b16(subnormal, __riscv_vmsne_vx_u16m1_b16(__riscv_vand_vx_u16m1(h, 0x8000u, vl), 0u, vl), vl);
#endif
#if defined(TT_RVV_PRESERVE_NEG_SUBNORMAL)
    subnormal =
        __riscv_vmand_mm_b16(subnormal, __riscv_vmseq_vx_u16m1_b16(__riscv_vand_vx_u16m1(h, 0x8000u, vl), 0u, vl), vl);
#endif
    vuint16m1_t signed_zero = __riscv_vand_vx_u16m1(h, 0x8000u, vl);
    return __riscv_vmerge_vvm_u16m1(h, signed_zero, subnormal, vl);
}

// ---- bf16 -> fp32 widen after target-declared input DAZ -------------------
static inline vfloat32m2_t rvk_bw_widen_bf16(const uint16_t* p, size_t vl) {
    vuint16m1_t h = __riscv_vle16_v_u16m1(p, vl);
    h = rvk_bw_flush_bf16_subnormals(h, vl);
    vuint32m2_t w = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(h, vl), 16, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(w);
}

// ---- fp32 -> bf16 RNE followed by target-declared output FTZ ---------------
static inline vuint16m1_t rvk_bw_pack_bf16(vfloat32m2_t y, size_t vl) {
    vuint32m2_t b = __riscv_vreinterpret_v_f32m2_u32m2(y);
    // RNE: b + 0x7FFF + ((b >> 16) & 1), then take the high half.
    vuint32m2_t lsb = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(b, 16, vl), 1u, vl);
    vuint32m2_t rnd = __riscv_vadd_vv_u32m2(__riscv_vadd_vx_u32m2(b, 0x7FFFu, vl), lsb, vl);
    // NaN lanes: truncate + force the quiet bit. vmfne(y,y) is true exactly for NaN.
    vbool16_t mn = __riscv_vmfne_vv_f32m2_b16(y, y, vl);
#if RVK_BW_PACK_E32
    // Same arithmetic, but the NaN select happens at e32 and only ONE op is
    // narrowing, so the e16 region is a single instruction instead of four
    // interleaved ones. The quiet bit is set at its e32 position (0x0040 << 16)
    // so the final high-half extract lands it in bit 6 of the bf16, exactly as
    // the mixed-SEW form did. Proven bit-identical to the form below over all
    // 2^32 fp32 patterns (scripts/prove_pack_bf16_identity.c, 0 mismatches).
    vuint32m2_t nanv = __riscv_vor_vx_u32m2(b, 0x00400000u, vl);
    vuint16m1_t h = __riscv_vnsrl_wx_u16m1(__riscv_vmerge_vvm_u32m2(rnd, nanv, mn, vl), 16, vl);
#else
    vuint16m1_t h = __riscv_vnsrl_wx_u16m1(rnd, 16, vl);
    vuint16m1_t hn = __riscv_vor_vx_u16m1(__riscv_vnsrl_wx_u16m1(b, 16, vl), 0x0040u, vl);
    h = __riscv_vmerge_vvm_u16m1(h, hn, mn, vl);
#endif
    return rvk_bw_flush_bf16_subnormals(h, vl);
}

#if RVK_ASYM_ACTIVE
// ---- production asymptotic_exp(), RVV mirror (see header for provenance) ---
static inline vfloat32m2_t rvk_bw_asym_exp(vfloat32m2_t arg, size_t vl) {
    constexpr float RVK_INV_LN2 = 1.4426950408889634f;
    constexpr float RVK_EXP_MAGIC = 12582912.0f;  // 0x4B400000 = 1.5 * 2^23
    vfloat32m2_t z = __riscv_vfmul_vf_f32m2(arg, RVK_INV_LN2, vl);
    // Branch-free RNE round-to-int (magic-number technique, no fcvt):
    vfloat32m2_t tmp = __riscv_vfadd_vf_f32m2(z, RVK_EXP_MAGIC, vl);
    vfloat32m2_t k = __riscv_vfsub_vf_f32m2(tmp, RVK_EXP_MAGIC, vl);
    vint32m2_t k_int = __riscv_vsub_vx_i32m2(__riscv_vreinterpret_v_f32m2_i32m2(tmp), 0x4B400000, vl);
    // Cody-Waite extended precision: r = arg - k*ln2 (two fused steps).
    constexpr float RVK_NEG_LN2_HI = -0.6931152343750000f;
    constexpr float RVK_NEG_LN2_LO = -3.19461832987e-05f;
    vfloat32m2_t r = __riscv_vfmacc_vf_f32m2(arg, RVK_NEG_LN2_HI, k, vl);
    r = __riscv_vfmacc_vf_f32m2(r, RVK_NEG_LN2_LO, k, vl);
    // Degree-5 Taylor for exp(r), |r| < ln(2)/2 (production coefficients).
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(1.0f / 120.0f, vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f / 24.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f / 6.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(0.5f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, one, vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, one, vl);
    // Scale by 2^k: exponent-field integer add (p > 0 here, so the biased
    // exponent is bits >> 23). UNDERFLOW GUARD exactly like production:
    // biased_exp(p) + k <= 0 would wrap the exponent field — clamp to 0.
    vint32m2_t pb = __riscv_vreinterpret_v_f32m2_i32m2(p);
    vint32m2_t new_exp = __riscv_vadd_vv_i32m2(__riscv_vsra_vx_i32m2(pb, 23, vl), k_int, vl);
    vfloat32m2_t out =
        __riscv_vreinterpret_v_i32m2_f32m2(__riscv_vadd_vv_i32m2(pb, __riscv_vsll_vx_i32m2(k_int, 23, vl), vl));
    vbool16_t uf = __riscv_vmsle_vx_i32m2_b16(new_exp, 0, vl);
    return __riscv_vfmerge_vfm_f32m2(out, 0.0f, uf, vl);
}

#ifndef ASYMPTOTIC_SCALE
#define ASYMPTOTIC_SCALE 1.0f  // production default (piecewise_generic.cpp QUADRATIC arm)
#endif

// ---- one 8-element chunk: y' = region(x) ? scale(dominant(x) * y) : y ------
static inline vfloat32m2_t rvk_bw_asym_apply(vfloat32m2_t y, vfloat32m2_t xo, size_t vl) {
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
    vfloat32m2_t t = __riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(xo, xo, vl), ASYMPTOTIC_EXP_ARG_SCALE, vl);
    vfloat32m2_t ya = __riscv_vfmul_vv_f32m2(y, rvk_bw_asym_exp(t, vl), vl);
#elif defined(ASYMPTOTIC_FACTOR_EXP_LINEAR)
    vfloat32m2_t t = __riscv_vfmul_vf_f32m2(xo, ASYMPTOTIC_EXP_ARG_SCALE, vl);
    vfloat32m2_t ya = __riscv_vfmul_vv_f32m2(y, rvk_bw_asym_exp(t, vl), vl);
#elif defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
    vfloat32m2_t t = __riscv_vfmul_vf_f32m2(xo, ASYMPTOTIC_EXP_ARG_SCALE, vl);
    vfloat32m2_t ya = __riscv_vfmul_vv_f32m2(__riscv_vfmul_vv_f32m2(y, xo, vl), rvk_bw_asym_exp(t, vl), vl);
#elif defined(ASYMPTOTIC_FACTOR_X)
    vfloat32m2_t ya = __riscv_vfmul_vv_f32m2(y, xo, vl);
#elif defined(ASYMPTOTIC_FACTOR_QUADRATIC)
    // (x-A)*(x-B) first, then y*q — the poly-cascade production order
    // (piecewise_generic.cpp:3341; the specialized file associates the other
    // way — value-equivalent, rounding arbitration is the ULP report).
    vfloat32m2_t q = __riscv_vfmul_vv_f32m2(
        __riscv_vfsub_vf_f32m2(xo, ASYMPTOTIC_QUAD_ROOT_A, vl),
        __riscv_vfsub_vf_f32m2(xo, ASYMPTOTIC_QUAD_ROOT_B, vl),
        vl);
    vfloat32m2_t ya = __riscv_vfmul_vv_f32m2(y, q, vl);
#endif
    ya = __riscv_vfmul_vf_f32m2(ya, ASYMPTOTIC_SCALE, vl);
#if defined(ASYMPTOTIC_NEGATE_OUTPUT)
    // Exact IEEE sign flip (production XORs the sign bit; never *-1).
    ya = __riscv_vfsgnjn_vv_f32m2(ya, ya, vl);
#endif
#if defined(ASYMPTOTIC_REGION_LEFT)
    // x < BOUND selects the asymptotic value; a failed compare (NaN) keeps the
    // cascade value, exactly like the SFPU v_if.
    vbool16_t m = __riscv_vmflt_vf_f32m2_b16(xo, ASYMPTOTIC_BOUND, vl);
    return __riscv_vmerge_vvm_f32m2(y, ya, m, vl);
#elif defined(ASYMPTOTIC_REGION_RIGHT)
    vbool16_t m = __riscv_vmfge_vf_f32m2_b16(xo, ASYMPTOTIC_BOUND, vl);
    return __riscv_vmerge_vvm_f32m2(y, ya, m, vl);
#else  // ASYMPTOTIC_REGION_ALL
    return ya;
#endif
}
#endif  // RVK_ASYM_ACTIVE

// ---------------------------------------------------------------------------
// Per-tile hooks called from piecewise_rvv.cpp kernel_main (see the hook
// patch): stage-in BEFORE the form dispatch, finish AFTER it. All loops run
// at e32m2 (vl = 8), 128 chunks per 1024-element tile, single stream — every
// step is 1-4 element-wise ops (plus the exp chain on asymptotic fits), and
// none of this sits inside the latency-critical gather/Horner loop.
// ---------------------------------------------------------------------------

// Returns the fp32 eval-input address for this tile. bf16: exact widen of the
// 2KB CB page into XSTAGE; fp32: the CB page itself (pass-through, no copy).
static inline uint32_t rvk_bw_stage_in(uint32_t cb_src) {
#if RVK_IO_BF16
    const uint16_t* src = (const uint16_t*)cb_src;
    float* dst = (float*)RVK_BW_XSTAGE;
    size_t vl = __riscv_vsetvl_e32m2(8);
#if RVK_BW_ILV == 4
    for (int c = 0; c < 128; c += 4) {
        vfloat32m2_t wA = rvk_bw_widen_bf16(src + c * 8, vl);
        vfloat32m2_t wB = rvk_bw_widen_bf16(src + (c + 1) * 8, vl);
        vfloat32m2_t wC = rvk_bw_widen_bf16(src + (c + 2) * 8, vl);
        vfloat32m2_t wD = rvk_bw_widen_bf16(src + (c + 3) * 8, vl);
        __riscv_vse32_v_f32m2(dst + c * 8, wA, vl);
        __riscv_vse32_v_f32m2(dst + (c + 1) * 8, wB, vl);
        __riscv_vse32_v_f32m2(dst + (c + 2) * 8, wC, vl);
        __riscv_vse32_v_f32m2(dst + (c + 3) * 8, wD, vl);
    }
#elif RVK_BW_ILV == 2
    // Both widens issue before either store, so chunk B's load/shift/store
    // chain fills the slots chunk A's store is stalled in.
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t wA = rvk_bw_widen_bf16(src + c * 8, vl);
        vfloat32m2_t wB = rvk_bw_widen_bf16(src + (c + 1) * 8, vl);
        __riscv_vse32_v_f32m2(dst + c * 8, wA, vl);
        __riscv_vse32_v_f32m2(dst + (c + 1) * 8, wB, vl);
    }
#else
    for (int c = 0; c < 128; c++) {
        __riscv_vse32_v_f32m2(dst + c * 8, rvk_bw_widen_bf16(src + c * 8, vl), vl);
    }
#endif
    return RVK_BW_XSTAGE;
#else
    return cb_src;
#endif
}

// Returns where the form evaluator writes its fp32 result. bf16: YSTAGE (the
// CB page is 2KB bf16); fp32: the output CB page directly (as today).
static inline uint32_t rvk_bw_eval_dst(uint32_t cb_dst) {
#if RVK_IO_BF16
    (void)cb_dst;
    return RVK_BW_YSTAGE;
#else
    return cb_dst;
#endif
}

// Post-dispatch epilogue: asymptotic apply -> domain actions -> typed special
// policy -> grad multiply -> bf16 RNE/FTZ pack.
//   x      : fp32 eval input of this tile (XSTAGE or the fp32 input CB page)
//   y      : fp32 eval output (YSTAGE, or the fp32 output CB page — then the
//            asym/grad steps run in place and the store is a plain vse32)
//   cb_dst : the real output CB page (bf16 pack target; unused in fp32 mode)
//   grad   : the c_1 grad tile page (bf16 or fp32 per USE_BF16), grad builds only
#if RVK_FUSE_GRAD
static inline void rvk_bw_finish_tile(const float* x, float* y, uint32_t cb_dst, uint32_t grad)
#else
static inline void rvk_bw_finish_tile(const float* x, float* y, uint32_t cb_dst)
#endif
{
    (void)x;
    (void)cb_dst;
    size_t vl = __riscv_vsetvl_e32m2(8);
#if RVK_IO_BF16
    uint16_t* out16 = (uint16_t*)cb_dst;
#endif
#if RVK_FUSE_GRAD && RVK_IO_BF16
    const uint16_t* g16 = (const uint16_t*)grad;
#elif RVK_FUSE_GRAD
    const float* g32 = (const float*)grad;
#endif
#if RVK_BW_ILV >= 2
    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t yA = __riscv_vle32_v_f32m2(y + c * 8, vl);
        vfloat32m2_t yB = __riscv_vle32_v_f32m2(y + (c + 1) * 8, vl);
#if RVK_ASYM_ACTIVE || defined(TT_DOMAIN_ACTION_PROGRAM) || defined(TT_SPECIAL_VALUE_POLICY)
        {
            vfloat32m2_t xA = __riscv_vle32_v_f32m2(x + c * 8, vl);
            vfloat32m2_t xB = __riscv_vle32_v_f32m2(x + (c + 1) * 8, vl);
#if RVK_ASYM_ACTIVE
            yA = rvk_bw_asym_apply(yA, xA, vl);
            yB = rvk_bw_asym_apply(yB, xB, vl);
#endif
            yA = tt_rvv_finalize_domain_actions(xA, yA, vl);
            yB = tt_rvv_finalize_domain_actions(xB, yB, vl);
            yA = tt_rvv_finalize_special_values(xA, yA, vl);
            yB = tt_rvv_finalize_special_values(xB, yB, vl);
        }
#endif
#if RVK_FUSE_GRAD && RVK_IO_BF16
        yA = __riscv_vfmul_vv_f32m2(yA, rvk_bw_widen_bf16(g16 + c * 8, vl), vl);
        yB = __riscv_vfmul_vv_f32m2(yB, rvk_bw_widen_bf16(g16 + (c + 1) * 8, vl), vl);
#elif RVK_FUSE_GRAD
        yA = __riscv_vfmul_vv_f32m2(yA, __riscv_vle32_v_f32m2(g32 + c * 8, vl), vl);
        yB = __riscv_vfmul_vv_f32m2(yB, __riscv_vle32_v_f32m2(g32 + (c + 1) * 8, vl), vl);
#endif
#if RVK_IO_BF16
        __riscv_vse16_v_u16m1(out16 + c * 8, rvk_bw_pack_bf16(yA, vl), vl);
        __riscv_vse16_v_u16m1(out16 + (c + 1) * 8, rvk_bw_pack_bf16(yB, vl), vl);
#else
        __riscv_vse32_v_f32m2(y + c * 8, yA, vl);
        __riscv_vse32_v_f32m2(y + (c + 1) * 8, yB, vl);
#endif
    }
#else
    for (int c = 0; c < 128; c++) {
        vfloat32m2_t yv = __riscv_vle32_v_f32m2(y + c * 8, vl);
#if RVK_ASYM_ACTIVE || defined(TT_DOMAIN_ACTION_PROGRAM) || defined(TT_SPECIAL_VALUE_POLICY)
        {
            vfloat32m2_t xo = __riscv_vle32_v_f32m2(x + c * 8, vl);
#if RVK_ASYM_ACTIVE
            yv = rvk_bw_asym_apply(yv, xo, vl);
#endif
            yv = tt_rvv_finalize_domain_actions(xo, yv, vl);
            yv = tt_rvv_finalize_special_values(xo, yv, vl);
        }
#endif
#if RVK_FUSE_GRAD && RVK_IO_BF16
        yv = __riscv_vfmul_vv_f32m2(yv, rvk_bw_widen_bf16(g16 + c * 8, vl), vl);
#elif RVK_FUSE_GRAD
        yv = __riscv_vfmul_vv_f32m2(yv, __riscv_vle32_v_f32m2(g32 + c * 8, vl), vl);
#endif
#if RVK_IO_BF16
        __riscv_vse16_v_u16m1(out16 + c * 8, rvk_bw_pack_bf16(yv, vl), vl);
#else
        __riscv_vse32_v_f32m2(y + c * 8, yv, vl);
#endif
    }
#endif
}

#endif  // RVK_BW_IO_ACTIVE && TRISC_PACK
#endif  // RVV_FORMS_RVV_BW_IO_H_
