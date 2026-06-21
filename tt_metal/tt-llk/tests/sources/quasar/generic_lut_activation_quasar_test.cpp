// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Generic piecewise-polynomial LUT activation on Quasar.
//
// Models the eltwise_unary_sfpu_quasar recipe (UNPACK -> SrcA -> FPU datacopy
// (MOVA2D) -> Dest -> SFPU -> PACK), but replaces the canned unary SFPU op with
// an embedded piecewise-polynomial LUT evaluated in the SFPU using the sfpi DSL.
//
// The LUT is a generic piecewise polynomial of arbitrary degree N (POLY_DEGREE)
// and arbitrary segment count S (NUM_SEGMENTS). Layout matches the BH/WH custom
// test:
//   [b0..bS  (NUM_SEGMENTS+1 ascending boundaries),
//    then per segment (POLY_DEGREE+1) Horner coeffs c0..cN]
// No range reduction. The coefficients/boundaries are baked in at build time.
// They DEFAULT to the proven sigmoid LUT (deg-2, 4-seg) below, but the python
// test overrides them via the build header (the GENERIC_LUT_DATA template
// parameter emits LUT_POLY_DEGREE / LUT_NUM_SEGMENTS / LUT_DATA_INIT). The
// Python golden replicates these EXACT values so the PCC isolates kernel
// correctness.
//
// Key Quasar-LLK differences vs the BH/WH custom test:
//   * Includes: Quasar MATH-thread headers (cfg_defines.h, cmath_common.h,
//     llk_math_common.h, llk_math_eltwise_unary_datacopy.h,
//     llk_math_eltwise_unary_sfpu.h) + the eltwise_unary recipe's UNPACK/PACK
//     headers, NOT the WH/BH llk_unpack_A.h / llk_pack.h set.
//   * Iteration model: the SFPU iterates 2 Dest rows per iteration
//     (SFP_ROWS == 2) over SFPU_ITERATIONS, advancing the Dest window with
//     ckernel::math::_incr_counters_<...SFP_ROWS...>() — NOT a 0..31 dst_reg
//     loop.
//   * Polynomial body uses sfpi dst_reg[0] / vFloat / v_if and float literal
//     constants (Converter::as_float does not exist on Quasar).

#include <array>
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

// The embedded piecewise-polynomial LUT (degree N, segment count S) is defined
// inside the LLK_TRISC_MATH block below — that is the only thread that evaluates
// it, and it is the only place where the auto-generated build header (params.h
// -> build.h, carrying the python test's LUT overrides) is in scope.

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id = 0;
    const std::uint32_t num_tiles   = params.TILE_CNT;

    // FPU path: UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim       = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim       = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim       = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val);

    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles);
    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(0, ckernel::DEFAULT_TENSOR_SHAPE);
}

#endif

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "sfpi.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// =====================================================================
// Embedded piecewise-polynomial LUT. Generic in degree N (POLY_DEGREE) and
// segment count S (NUM_SEGMENTS). Layout: [b0..bS (S+1 ascending boundaries),
// then per segment (N+1) Horner coeffs c0..cN]. Defaults below replicate the
// proven sigmoid LUT (deg-2, 4-seg); the python test overrides them via the
// build header (params.h -> build.h), emitting LUT_POLY_DEGREE /
// LUT_NUM_SEGMENTS / LUT_DATA_INIT. Defined here (after build.h is in scope)
// because the MATH thread is the only consumer.
// =====================================================================
#ifndef LUT_POLY_DEGREE
#define LUT_POLY_DEGREE 2
#endif
#ifndef LUT_NUM_SEGMENTS
#define LUT_NUM_SEGMENTS 4
#endif
#ifndef LUT_DATA_INIT
// clang-format off
#define LUT_DATA_INIT {                       \
    /* boundaries b0..b4 */                   \
    -4.0f, -2.0f, 0.0f, 2.0f, 4.0f,           \
    /* seg0 coeffs: c0, c1, c2 */             \
    0.38296354f, 0.17515847f, 0.02109685f,    \
    /* seg1 */                                \
    0.50329190f, 0.27505103f, 0.04113654f,    \
    /* seg2 */                                \
    0.49670810f, 0.27505103f, -0.04113654f,   \
    /* seg3 */                                \
    0.61703646f, 0.17515847f, -0.02109685f }
// clang-format on
#endif

// ---- Range reduction contract (codegen emits these; default = none) ----
// The python test parses range_reduction_method (+ params) from the fitter CSV
// METADATA and emits these #defines into build.h. The kernel then performs the
// exact reduce+reconstruct from range_reduction.py so the golden can be the TRUE
// activation over the FULL original domain. With no codegen (default sigmoid
// build) LUT_RR_METHOD stays 0 (none) and the kernel is byte-identical to before.
#ifndef LUT_RR_METHOD
#define LUT_RR_METHOD 0 // 0 none, 1 log, 2 exp, 3 cbrt,
                        // 4 expalu_exp2, 5 expalu_log2, 6 expalu_pow
#endif
// legacy log
#ifndef LUT_RR_LOG_LN2
#define LUT_RR_LOG_LN2 1.0f
#endif
// legacy exp
#ifndef LUT_RR_EXP_MULT
#define LUT_RR_EXP_MULT 1.4426950408889634f
#endif
#ifndef LUT_RR_EXP_CONST
#define LUT_RR_EXP_CONST 0.6931471805599453f
#endif
// legacy cbrt + expalu_pow scale table (3 entries; pow n<=3 uses [0..n-1])
#ifndef LUT_RR_SCALE0
#define LUT_RR_SCALE0 1.0f
#endif
#ifndef LUT_RR_SCALE1
#define LUT_RR_SCALE1 1.0f
#endif
#ifndef LUT_RR_SCALE2
#define LUT_RR_SCALE2 1.0f
#endif
// expalu_exp2
#ifndef LUT_RR_EXP2_MULT
#define LUT_RR_EXP2_MULT 1.0f
#endif
#ifndef LUT_RR_COMPOSE
#define LUT_RR_COMPOSE 0 // 0 none, 1 sigmoid, 2 minus_one
#endif
// expalu_log2
#ifndef LUT_RR_LOG2_SCALE
#define LUT_RR_LOG2_SCALE 1.0f
#endif
#ifndef LUT_RR_LOG2_BASIS_MMINUS1
#define LUT_RR_LOG2_BASIS_MMINUS1 0 // 1 -> evaluate Horner on (m-1)
#endif
#ifndef LUT_RR_INPUT_OFFSET
#define LUT_RR_INPUT_OFFSET 0.0f // log1p: x+1 before frexp
#endif
// expalu_pow
#ifndef LUT_RR_POW_N
#define LUT_RR_POW_N 2
#endif
#ifndef LUT_RR_POW_RECIP
#define LUT_RR_POW_RECIP 0
#endif

constexpr std::uint32_t POLY_DEGREE  = LUT_POLY_DEGREE;
constexpr std::uint32_t NUM_SEGMENTS = LUT_NUM_SEGMENTS;
constexpr std::uint32_t LUT_SIZE     = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1);

constexpr std::array<float, LUT_SIZE> LUT_DATA = LUT_DATA_INIT;

namespace
{
// SFPADDI avoidance (FIX 3). The Horner step `acc * x + c` lowers in sfpi to a
// separate SFPMUL + SFPADD. When `c` is a compile-time constant (LUT_DATA is
// constexpr) the backend fuses the SFPADD with the constant's SFPLOADI into a
// single SFPADDI for round values (e.g. 0.5 = 0x3F000000) — and Quasar removed
// SFPADDI (ttsim aborts "tensix_execute_sfpaddi"). Emitting the multiply-add as
// one fused SFPMAD instead means the constant is the MAD's third (addend)
// operand, so there is no standalone SFPADD for the backend to fold into SFPADDI.
sfpi_inline sfpi::vFloat fma_const(sfpi::vFloat a, sfpi::vFloat b, float c)
{
    const sfpi::vFloat cv = sfpi::vFloat(c);
    return __builtin_rvtt_sfpmad(a.get(), b.get(), cv.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// =====================================================================
// Range-reduction primitives (exponent family). Mirror range_reduction.py's
// reduce/reconstruct math using the Quasar SFPU exexp/setexp/convert ops. All
// adds that the SFPADDI-fold rule could trip are expressed as fused SFPMAD
// (fma_const / explicit __builtin_rvtt_sfpmad). int<->float goes through vSMag
// (the only int<->float path on Quasar sfpi: there is NO direct
// convert<vInt>(vFloat) or convert<vFloat>(vInt)).
// =====================================================================
#if LUT_RR_METHOD != 0

// Reduced-domain mantissa m in [1,2): replace x's biased exponent with 127.
// Matches np.frexp adjustment (m=2*m, e=e-1) -> m in [1,2). Proven idiom from
// the BH log kernel (setexp(in,127)). Do NOT use exman (FractionOnly semantics).
sfpi_inline sfpi::vFloat rr_mantissa(sfpi::vFloat in)
{
    return sfpi::setexp(in, 127);
}

// int -> float (vInt -> vSMag -> vFloat); the only int->float path on Quasar.
sfpi_inline sfpi::vFloat rr_int_to_float(sfpi::vInt v)
{
    const auto s = sfpi::convert<sfpi::vSMag>(v);
    return sfpi::convert<sfpi::vFloat>(s, sfpi::RoundMode::Nearest);
}

// True signed exponent e as float (e==0 for x in [1,2)). exexp(Debias) returns
// the frexp-adjusted signed exponent directly (matches BH log kernel pairing).
sfpi_inline sfpi::vFloat rr_exp_float(sfpi::vFloat in)
{
    return rr_int_to_float(sfpi::exexp(in, sfpi::ExponentMode::Debias));
}

// Variable ldexp: result = mant * 2^(e_int). addexp() only takes a compile-time
// int, so synthesize 2^e by writing the biased exponent field (e+127) onto 1.0
// and multiplying. e_int is the TRUE (debiased) integer exponent as a vInt.
// Valid for |e_int| <= 127 (all in-scope configs have |e| <= ~15).
sfpi_inline sfpi::vFloat rr_ldexp(sfpi::vFloat mant, sfpi::vInt e_int)
{
    const sfpi::vInt   e_biased  = e_int + sfpi::vInt(127);
    const sfpi::vFloat two_pow_e = sfpi::setexp(sfpi::vFloat(1.0f), e_biased);
    return mant * two_pow_e;
}

// round-to-nearest float -> vInt. Quasar has no fp32->sm32 SFPCAST (mod 4/5 are
// rejected by the compiler); the working path is SFPSTOCHRND fp32->sm16 (RNE),
// which yields a sign-magnitude (sign at bit31) pattern, then SFPCAST sm32->int32
// (the convert<vInt>(vSMag) = smag_to_int path). The reduced exponent integers
// (|q|,|i|,|e| <= ~127) fit in 16 bits, so this is exact.
sfpi_inline sfpi::vInt rr_round_to_int(sfpi::vFloat t)
{
    const sfpi::vSMag s = sfpi::vSMag(__builtin_rvtt_sfpstochrnd_i(t.get(), 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_SMAG16, sfpi::SFPSTOCHRND_RND_EVEN));
    return sfpi::convert<sfpi::vInt>(s);
}

// floor(t) -> vInt. convert<vSMag> only rounds to nearest, so round then correct
// down when the rounded value overshot the input (toward -inf).
sfpi_inline sfpi::vInt rr_floor_to_int(sfpi::vFloat t)
{
    sfpi::vInt         i  = rr_round_to_int(t);
    const sfpi::vFloat fi = rr_int_to_float(i);
    v_if (fi > t)
    {
        i = i - sfpi::vInt(1);
    }
    v_endif;
    return i;
}

// Floored divmod by compile-time divisor d (>0): e = d*q + r, r in [0,d).
// Matches np.floor(e/d) / (e - d*q). |e| <= 127 is exactly fp32-representable.
// vInt has no multiply on Quasar, so r is formed in the float domain via a fused
// FMA (-d)*q_f + e_f (no SFPADDI) then converted back to int (r in [0,d) is exact).
sfpi_inline void rr_divmod_floor(sfpi::vInt e, int d, sfpi::vInt& q, sfpi::vInt& r)
{
    const sfpi::vFloat ef = rr_int_to_float(e);
    q                     = rr_floor_to_int(ef * sfpi::vFloat(1.0f / (float)d));
    const sfpi::vFloat qf = rr_int_to_float(q);
    const sfpi::vFloat rf = __builtin_rvtt_sfpmad(qf.get(), sfpi::vFloat(-(float)d).get(), ef.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    r                     = rr_round_to_int(rf);
}

// Reciprocal 1/x for the reconstruct post-ops (rsqrt, sigmoid compose). Quasar
// sfpi has no vFloat operator/; use the SFPNONLINEAR recip approximation plus two
// Newton-Raphson steps (y <- y*(2 - x*y)), formed with fused FMA (no SFPADDI).
sfpi_inline sfpi::vFloat rr_recip(sfpi::vFloat x)
{
    sfpi::vFloat y = sfpi::approx_recip(x);
#pragma GCC unroll 1
    for (int it = 0; it < 2; it++)
    {
        // t = x*y - 2 ;  y = -y*t = y*(2 - x*y)
        const sfpi::vFloat t = fma_const(x, y, -2.0f);
        y                    = __builtin_rvtt_sfpmad(y.get(), (-t).get(), sfpi::vFloat(0.0f).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    return y;
}

// Multiply v by scale-table entry C[r], r in {0,1,2}. Constants are compile-time
// folded; top-level v_if (no nesting) per the predicate-stack rule.
sfpi_inline sfpi::vFloat rr_scale_by_r(sfpi::vFloat v, sfpi::vInt r)
{
    sfpi::vFloat out = v * sfpi::vFloat(LUT_RR_SCALE0); // r==0 default
    v_if (r == sfpi::vInt(1))
    {
        out = v * sfpi::vFloat(LUT_RR_SCALE1);
    }
    v_endif;
    v_if (r == sfpi::vInt(2))
    {
        out = v * sfpi::vFloat(LUT_RR_SCALE2);
    }
    v_endif;
    return out;
}

#endif // LUT_RR_METHOD != 0

// Horner step recursion with COMPILE-TIME coefficient indices (FIX 4). Reading
// LUT_DATA only through constexpr indices (never `&LUT_DATA[i]` with a runtime
// index) lets the compiler constant-fold every coefficient into an SFPLOADI
// immediate and ELIDE the LUT_DATA array — so it never lands in .ldm_data and
// cannot overflow the tiny 1792 B private TRISC1_LOCAL_DATA_MEM region (the
// "section .ldm_data not within region" error on large deg/seg configs). The
// folded immediates live in code (16 KiB TRISC1_CODE), which has room. (Placing
// LUT_DATA in an L1 `l1_data` section instead corrupts the SFPU reads on this
// Quasar sim, so folding it away is the robust fix.) Applies coefficients c_K
// down to c_0.
template <std::uint32_t BASE, int K>
sfpi_inline sfpi::vFloat horner_step(sfpi::vFloat acc, sfpi::vFloat x)
{
    if constexpr (K < 0)
    {
        return acc;
    }
    else
    {
        acc = fma_const(acc, x, LUT_DATA[BASE + static_cast<std::uint32_t>(K)]);
        return horner_step<BASE, K - 1>(acc, x);
    }
}

// eval_seg<SEG>: clamp x to [b_SEG, b_{SEG+1}] (top-level v_if) and Horner-eval
// segment SEG with compile-time coefficient indices, so the polynomial is never
// evaluated outside its fit range and LUT_DATA is constant-folded away.
template <std::uint32_t SEG>
sfpi_inline sfpi::vFloat eval_seg(sfpi::vFloat x_clamped)
{
    constexpr std::uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr std::uint32_t COEFF_OFFSET       = NUM_SEGMENTS + 1; // skip boundaries
    constexpr std::uint32_t base               = COEFF_OFFSET + SEG * COEFFS_PER_SEGMENT;

    const sfpi::vFloat seg_lo = sfpi::vFloat(LUT_DATA[SEG]);
    const sfpi::vFloat seg_hi = sfpi::vFloat(LUT_DATA[SEG + 1]);
    sfpi::vFloat       xs     = x_clamped;
    v_if (xs < seg_lo)
    {
        xs = seg_lo;
    }
    v_endif;
    v_if (xs > seg_hi)
    {
        xs = seg_hi;
    }
    v_endif;
    const sfpi::vFloat acc = sfpi::vFloat(LUT_DATA[base + POLY_DEGREE]);
    return horner_step<base, static_cast<int>(POLY_DEGREE) - 1>(acc, xs);
}

// Cumulative segment-override chain via template recursion over compile-time SEG
// (the loop is unrolled because NUM_SEGMENTS is constexpr). Each segment's
// (already-bounded) evaluation is computed BEFORE the select v_if so its clamp
// v_ifs stay at top level — a nested v_if corrupts the Quasar predicate stack.
template <std::uint32_t SEG>
sfpi_inline void select_segment(sfpi::vFloat& result, sfpi::vFloat x_clamped)
{
    if constexpr (SEG < NUM_SEGMENTS)
    {
        const sfpi::vFloat b_seg  = sfpi::vFloat(LUT_DATA[SEG]);
        const sfpi::vFloat segval = eval_seg<SEG>(x_clamped);
        v_if (x_clamped >= b_seg)
        {
            result = segval;
        }
        v_endif;
        select_segment<SEG + 1>(result, x_clamped);
    }
}

// Evaluate the embedded piecewise-polynomial LUT on the current 2-row Dest
// window (sfpi dst_reg[0]). Clamp x to [b0, bN], select segment by boundaries,
// Horner-eval that segment.
//
// CRITICAL (Quasar): sfpi evaluates the RHS of a predicated assignment on ALL
// 32 lanes; the v_if predicate only gates the final conditional MOV. The
// original kernel did `for seg: v_if (x >= b[seg]) result = eval(seg, x_clamped);`,
// i.e. it ran EVERY segment's Horner on every lane's GLOBALLY clamped x. For a
// high-degree poly evaluated far outside its fit interval (e.g. a [5,5.3]-fit
// deg-8 poly at x=10) that RHS overflows to ~1e36/inf. With many segments the
// late cumulative overrides on Quasar fail to fully overwrite those poisoned
// lanes, so the top segments read back ~1e37 (PCC craters; the s32 bug). The
// segment select itself was always correct.
//
// Fix: clamp the Horner argument to EACH segment's own sub-interval
// [b_seg, b_{seg+1}] before evaluating that segment, so no polynomial is ever
// evaluated outside its fit range -> every RHS stays bounded and finite, and the
// cumulative select can no longer be poisoned, independent of segment count.
// The per-segment clamp uses TOP-LEVEL v_if blocks (NOT nested inside the select
// v_if): a nested v_if corrupts the Quasar predicate stack and silently drops
// the override write.
sfpi_inline void piecewise_generic_lut_sfp_rows()
{
    const sfpi::vFloat x_in = sfpi::dst_reg[0];

    // ---- REDUCE: compute the reduced polynomial argument r_arg + the per-method
    // reconstruct state. For LUT_RR_METHOD==0 (none) r_arg is x itself and the
    // reconstruct block is empty, so this is byte-identical to the legacy path.
    sfpi::vFloat r_arg = x_in;
#if LUT_RR_METHOD == 1 // legacy log: x = 2^e * m
    const sfpi::vFloat rr_ef = rr_exp_float(x_in);
    r_arg                    = rr_mantissa(x_in);
#elif LUT_RR_METHOD == 2 // legacy exp: q=round(x*mult); s = x - q*const
    const sfpi::vFloat t_q = x_in * sfpi::vFloat(LUT_RR_EXP_MULT);
    const sfpi::vInt   q_i = rr_round_to_int(t_q);
    const sfpi::vFloat q_f = rr_int_to_float(q_i);
    // s = x - q*const via fused FMA: (-const)*q + x (no SFPADDI)
    r_arg = __builtin_rvtt_sfpmad(q_f.get(), sfpi::vFloat(-(float)(LUT_RR_EXP_CONST)).get(), x_in.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
#elif LUT_RR_METHOD == 3 // legacy cbrt: |x| = 2^e * m, e = 3q + r
    const sfpi::vFloat ax = sfpi::abs(x_in);
    const sfpi::vInt   e_i = sfpi::exexp(ax, sfpi::ExponentMode::Debias);
    sfpi::vInt         q_i, r_i;
    rr_divmod_floor(e_i, 3, q_i, r_i);
    r_arg = rr_mantissa(ax);
#elif LUT_RR_METHOD == 4 // expalu_exp2: t = x*mult; i = floor(t); f = t - i
    const sfpi::vFloat t_e = x_in * sfpi::vFloat(LUT_RR_EXP2_MULT);
    const sfpi::vInt   i_i = rr_floor_to_int(t_e);
    const sfpi::vFloat i_f = rr_int_to_float(i_i);
    r_arg = __builtin_rvtt_sfpmad(sfpi::vFloat(-1.0f).get(), i_f.get(), t_e.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE); // f = t - i
#elif LUT_RR_METHOD == 5 // expalu_log2: (x+off) = 2^e * m
    const sfpi::vFloat xo   = x_in + sfpi::vFloat(LUT_RR_INPUT_OFFSET);
    const sfpi::vFloat rr_ef = rr_exp_float(xo);
    const sfpi::vFloat m_v  = rr_mantissa(xo);
#if LUT_RR_LOG2_BASIS_MMINUS1
    r_arg = m_v - sfpi::vFloat(1.0f); // Horner on u = m-1 (c0 == 0)
#else
    r_arg = m_v;
#endif
#elif LUT_RR_METHOD == 6 // expalu_pow: |x| = 2^e * m, e = n*q + r
    const sfpi::vFloat ax = sfpi::abs(x_in);
    const sfpi::vInt   e_i = sfpi::exexp(ax, sfpi::ExponentMode::Debias);
    sfpi::vInt         q_i, r_i;
    rr_divmod_floor(e_i, LUT_RR_POW_N, q_i, r_i);
    r_arg = rr_mantissa(ax);
#endif

    // ---- Clamp r_arg to [b0, bN] (the REDUCED domain) with explicit v_if
    // (vec_min_max semantics differ from WH/BH, which silently pinned everything
    // to the lower bound).
    const sfpi::vFloat b_lo = sfpi::vFloat(LUT_DATA[0]);
    const sfpi::vFloat b_hi = sfpi::vFloat(LUT_DATA[NUM_SEGMENTS]);

    sfpi::vFloat x_clamped = r_arg;
    v_if (x_clamped < b_lo)
    {
        x_clamped = b_lo;
    }
    v_endif;
    v_if (x_clamped > b_hi)
    {
        x_clamped = b_hi;
    }
    v_endif;

    // Start with segment 0, then override with the correct segment per lane via
    // the template-recursive cumulative chain (see select_segment).
    sfpi::vFloat result = eval_seg<0>(x_clamped);
    select_segment<1>(result, x_clamped);

    // ---- RECONSTRUCT (full activation over the original domain).
#if LUT_RR_METHOD == 1 // log: e*ln2 + P(m)
    result = __builtin_rvtt_sfpmad(rr_ef.get(), sfpi::vFloat(LUT_RR_LOG_LN2).get(), result.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    v_if (x_in == sfpi::vFloat(0.0f))
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
#elif LUT_RR_METHOD == 2 // exp: 2^q * P(s)
    result = rr_ldexp(result, q_i);
#elif LUT_RR_METHOD == 3 // cbrt: sign * C[r] * 2^q * P(m)
    result = rr_scale_by_r(result, r_i);
    result = rr_ldexp(result, q_i);
    result = sfpi::copysgn(result, x_in); // odd function: restore sign
    v_if (sfpi::abs(x_in) == sfpi::vFloat(0.0f))
    {
        result = sfpi::vFloat(0.0f);
    }
    v_endif;
#elif LUT_RR_METHOD == 4 // expalu_exp2: 2^i * P(f) [+ compose]
    result = rr_ldexp(result, i_i);
#if LUT_RR_COMPOSE == 1 // sigmoid: 1/(1+y)
    result = rr_recip(result + sfpi::vFloat(1.0f));
#elif LUT_RR_COMPOSE == 2 // minus_one: y - 1
    result = result - sfpi::vFloat(1.0f);
#endif
#elif LUT_RR_METHOD == 5 // expalu_log2: scale*(e + P(u))
    result = (rr_ef + result) * sfpi::vFloat(LUT_RR_LOG2_SCALE);
#elif LUT_RR_METHOD == 6 // expalu_pow: [1/](sign) * C[r] * 2^q * P(m)
    result = rr_scale_by_r(result, r_i);
    result = rr_ldexp(result, q_i);
#if (LUT_RR_POW_N % 2) == 1
    result = sfpi::copysgn(result, x_in); // odd root: restore sign
#endif
#if LUT_RR_POW_RECIP
    result = rr_recip(result); // rsqrt
#endif
    v_if (sfpi::abs(x_in) == sfpi::vFloat(0.0f))
    {
#if LUT_RR_POW_RECIP
        result = std::numeric_limits<float>::infinity();
#else
        result = sfpi::vFloat(0.0f);
#endif
    }
    v_endif;
#endif

    sfpi::dst_reg[0] = result;
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_generic_lut()
{
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++)
    {
        piecewise_generic_lut_sfp_rows();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // dest_reg++ (advances by 2 rows)
    }
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // FPU path: UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest -> SFPU -> PACK.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    // FPU datacopy SrcA -> Dest (MOVA2D) before SFPU operates on it.
    const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_rows, 1);
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();

    _llk_math_eltwise_sfpu_init_();

    // Apply the embedded piecewise-polynomial LUT in-place on Dest for each tile.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_generic_lut<SFPU_ITERATIONS>, params.DST_INDEX + i);
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = params.TILE_CNT;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = params.buffer_Res[0] / 16;
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX, 0, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
