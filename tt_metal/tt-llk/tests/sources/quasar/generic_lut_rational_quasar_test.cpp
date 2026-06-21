// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Generic piecewise-RATIONAL LUT activation on Quasar.
//
// Sibling of generic_lut_activation_quasar_test.cpp (piecewise-polynomial),
// but each segment evaluates a rational approximation P(x)/Q(x) instead of a
// single polynomial. The numerator P and denominator Q are independent
// polynomials (degrees RAT_NUM_DEGREE / RAT_DEN_DEGREE). The division is done
// in the SFPU as  result = P(x) * (1 / Q(x))  using the sfpi-based iterative
// reciprocal (_sfpu_reciprocal_) from ckernel_sfpu_recip.h.
//
// Reciprocal choice: the iterative sfpi reciprocal (approx_recip + 2 rounds of
// Newton-Raphson) is used instead of the native SFPNONLINEAR RECIP. In ttsim,
// SFPNONLINEAR RECIP is libm-exact and so is optimistic vs real silicon; the
// iterative sfpi path uses the same approx-reciprocal seed + NR refinement that
// runs on hardware, so its PCC is representative of silicon. (See RETURN note.)
//
// Unlike the poly test, the LUT layout (boundaries, per-segment numerator and
// denominator coefficients, degrees, segment count) is NOT hard-coded here.
// It is injected at build time through build.h (pulled in via params.h) by the
// Python test's RATIONAL_LUT template parameter, so the same source serves any
// rational config the tt-polynomial-fitter emits. The Python golden replicates
// the EXACT same coefficients and division semantics so the PCC isolates kernel
// correctness on the fitter's rational coefficients.
//
// build.h provides:
//   constexpr std::uint32_t RAT_NUM_SEGMENTS;
//   constexpr std::uint32_t RAT_NUM_DEGREE;   // numerator P degree
//   constexpr std::uint32_t RAT_DEN_DEGREE;   // denominator Q degree
//   constexpr std::array<float, RAT_NUM_SEGMENTS + 1> RAT_BOUNDARIES;
//   constexpr std::array<float, RAT_NUM_SEGMENTS * (RAT_NUM_DEGREE + 1)> RAT_NUM_COEFFS;
//   constexpr std::array<float, RAT_NUM_SEGMENTS * (RAT_DEN_DEGREE + 1)> RAT_DEN_COEFFS;
// All coefficient arrays are stored low-order-first (c0, c1, c2, ...).
//
// Quasar-LLK specifics match the proven poly test: Quasar semantic LLK headers
// (no llk_unpack_A.h), the 2-rows-per-iteration SFPU model with
// _incr_counters_<...SFP_ROWS...>(), sfpi dst_reg[0]/vFloat/v_if, and float
// literal constants (no Converter::as_float).

#include <array>
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

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
#include "sfpu/ckernel_sfpu_recip.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// ---- Range reduction contract (codegen emits these; default = none) ----
// The python test parses range_reduction_method (+ params) from the fitter CSV
// METADATA and emits these #defines into build.h. The kernel then performs the
// exact reduce+reconstruct from range_reduction.py so the golden can be the TRUE
// activation over the FULL original domain. With no codegen (default sigmoid
// build) LUT_RR_METHOD stays 0 (none) and the kernel is byte-identical to before.
#ifndef LUT_RR_METHOD
#define LUT_RR_METHOD 0 // 0 none, 1 log, 2 exp, 3 cbrt,
                        // 4 expalu_exp2, 5 expalu_log2, 6 expalu_pow,
                        // 7 trig (sin/cos), 8 tan
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
// trig (sin/cos) — method 7
#ifndef LUT_RR_TRIG_INV_PI
#define LUT_RR_TRIG_INV_PI 0.3183098861837907f // 1/pi
#endif
#ifndef LUT_RR_TRIG_PI
#define LUT_RR_TRIG_PI 3.141592653589793f
#endif
// tan — method 8 (Cody-Waite two-stage; constants are fp32-exact and MUST match
// range_reduction.py NEG_PI_2_HI/LO so the numpy fp32 golden is faithful).
#ifndef LUT_RR_TAN_INV_HALFPI
#define LUT_RR_TAN_INV_HALFPI 0.6366197723675814f // 1/(pi/2)
#endif
#ifndef LUT_RR_TAN_PI2_HI
#define LUT_RR_TAN_PI2_HI 1.5703125f // pi/2 high bits
#endif
#ifndef LUT_RR_TAN_PI2_LO
#define LUT_RR_TAN_PI2_LO 0.0004837512969970703f // pi/2 remainder
#endif

namespace
{
// SFPADDI avoidance (ported from the poly kernel). The Horner step `acc * x + c`
// lowers in sfpi to a separate SFPMUL + SFPADD. When `c` is a compile-time
// constant the backend can fuse the SFPADD with the constant's SFPLOADI into a
// single SFPADDI for round values — and Quasar removed SFPADDI (ttsim aborts
// "tensix_execute_sfpaddi"). Emitting the multiply-add as one fused SFPMAD means
// the constant is the MAD's third (addend) operand, so there is no standalone
// SFPADD for the backend to fold into SFPADDI. (The reciprocal's own NR subtract
// is handled separately by -mno-tt-tensix-optimize-combine; see below.)
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

// Parity p in {0,1} of a signed integer kv: p = kv - 2*floor(kv/2). Computed in
// the float domain (no vInt multiply / no signed bit-AND ambiguity on Quasar),
// returned as a vFloat that is exactly 0.0f or 1.0f. |kv| <= ~few is exact in fp32.
sfpi_inline sfpi::vFloat rr_parity_f(sfpi::vInt kv)
{
    const sfpi::vFloat kf = rr_int_to_float(kv);
    const sfpi::vInt   h  = rr_floor_to_int(kf * sfpi::vFloat(0.5f)); // floor(k/2)
    const sfpi::vFloat hf = rr_int_to_float(h);
    // p = k - 2*h  via fused FMA: (-2)*hf + kf (no SFPADDI)
    return __builtin_rvtt_sfpmad(hf.get(), sfpi::vFloat(-2.0f).get(), kf.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

#endif // LUT_RR_METHOD != 0

// Horner step recursion with COMPILE-TIME coefficient indices (the proven poly
// kernel fold). Reading the constexpr coefficient array ONLY through constexpr
// indices (never `&ARR[i]` / `ARR[k]` with a runtime index) lets the compiler
// constant-fold every coefficient into an SFPLOADI immediate and ELIDE the
// coefficient array — so it never lands in .ldm_data and cannot overflow the
// tiny 1792 B private TRISC1_LOCAL_DATA_MEM region (the "section .ldm_data not
// within region" link error on large deg/seg rational configs). The folded
// immediates live in code (16 KiB TRISC1_CODE), which has room. Applies
// coefficients c_K down to c_0 from the given constexpr array reference + BASE.
template <const auto& ARR, std::uint32_t BASE, int K>
sfpi_inline sfpi::vFloat horner_step(sfpi::vFloat acc, sfpi::vFloat x)
{
    if constexpr (K < 0)
    {
        return acc;
    }
    else
    {
        acc = fma_const(acc, x, ARR[BASE + static_cast<std::uint32_t>(K)]);
        return horner_step<ARR, BASE, K - 1>(acc, x);
    }
}

// Horner-eval a polynomial from constexpr array ARR, segment SEG, given degree
// DEG and per-segment stride DEG+1, all with compile-time indices.
template <const auto& ARR, std::uint32_t SEG, std::uint32_t DEG>
sfpi_inline sfpi::vFloat eval_poly_seg(sfpi::vFloat x)
{
    constexpr std::uint32_t base = SEG * (DEG + 1);
    const sfpi::vFloat      acc  = sfpi::vFloat(ARR[base + DEG]);
    return horner_step<ARR, base, static_cast<int>(DEG) - 1>(acc, x);
}

// Cumulative segment-override chain via template recursion over compile-time SEG
// (unrolled because RAT_NUM_SEGMENTS is constexpr). Each segment's numerator and
// denominator are evaluated on the clamped x, then selected per lane by the
// boundary compare. Mirrors the poly kernel's select_segment, carrying both P and
// Q.
template <std::uint32_t SEG>
sfpi_inline void select_segment(sfpi::vFloat& p, sfpi::vFloat& q, sfpi::vFloat x_clamped)
{
    if constexpr (SEG < RAT_NUM_SEGMENTS)
    {
        const sfpi::vFloat b_seg  = sfpi::vFloat(RAT_BOUNDARIES[SEG]);
        const sfpi::vFloat segp   = eval_poly_seg<RAT_NUM_COEFFS, SEG, RAT_NUM_DEGREE>(x_clamped);
        const sfpi::vFloat segq   = eval_poly_seg<RAT_DEN_COEFFS, SEG, RAT_DEN_DEGREE>(x_clamped);
        v_if (x_clamped >= b_seg)
        {
            p = segp;
            q = segq;
        }
        v_endif;
        select_segment<SEG + 1>(p, q, x_clamped);
    }
}

// Evaluate the injected piecewise-rational LUT on the current 2-row Dest window
// (sfpi dst_reg[0]). Clamp x to [b0, bN], select segment by boundaries, then
// compute P(x) / Q(x) = P(x) * reciprocal(Q(x)). All coefficient reads use
// compile-time indices (see horner_step) so the LUT arrays are folded away.
sfpi_inline void piecewise_rational_lut_sfp_rows()
{
    const sfpi::vFloat x_in = sfpi::dst_reg[0];

    // ---- REDUCE: compute reduced argument r_arg + per-method reconstruct state.
    // LUT_RR_METHOD==0 -> r_arg = x_in (byte-identical to the legacy rational path).
    sfpi::vFloat r_arg = x_in;
    // RATIONAL-SPECIFIC REGISTER-PRESSURE NOTE: the Quasar SFPU exposes only 8
    // vector LREGs and CANNOT spill them to memory (the reload pass aborts:
    // "maximum number of generated reload insns per insn"). Unlike the poly
    // kernel, the rational eval keeps BOTH P and Q live AND runs the iterative
    // reciprocal (its own Newton-Raphson temporaries), so holding the exponent
    // /quotient reduce-state live across the P*recip(Q) eval overflows the 8
    // LREGs at deg>=3/multi-seg configs. The reduce-state is a deterministic
    // function of x_in, so the REDUCE block computes ONLY r_arg (+ the cheap
    // parity flags for trig/tan that are needed unchanged), and the RECONSTRUCT
    // block RECOMPUTES the exponent/quotient state from x_in after the eval has
    // freed its registers. Numerically identical to range_reduction.py; only the
    // live-range is shortened. (LUT_RR_METHOD==0 is unaffected -> byte-identical.)
#if LUT_RR_METHOD == 1 // legacy log: x = 2^e * m
    r_arg = rr_mantissa(x_in);
#elif LUT_RR_METHOD == 2 // legacy exp: q=round(x*mult); s = x - q*const
    const sfpi::vFloat t_q = x_in * sfpi::vFloat(LUT_RR_EXP_MULT);
    const sfpi::vFloat q_f = rr_int_to_float(rr_round_to_int(t_q));
    // s = x - q*const via fused FMA: (-const)*q + x (no SFPADDI)
    r_arg = __builtin_rvtt_sfpmad(q_f.get(), sfpi::vFloat(-(float)(LUT_RR_EXP_CONST)).get(), x_in.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
#elif LUT_RR_METHOD == 3 // legacy cbrt: |x| = 2^e * m, e = 3q + r
    r_arg = rr_mantissa(sfpi::abs(x_in));
#elif LUT_RR_METHOD == 4 // expalu_exp2: t = x*mult; i = floor(t); f = t - i
    const sfpi::vFloat t_e = x_in * sfpi::vFloat(LUT_RR_EXP2_MULT);
    const sfpi::vFloat i_f = rr_int_to_float(rr_floor_to_int(t_e));
    r_arg = __builtin_rvtt_sfpmad(sfpi::vFloat(-1.0f).get(), i_f.get(), t_e.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE); // f = t - i
#elif LUT_RR_METHOD == 5 // expalu_log2: (x+off) = 2^e * m
    const sfpi::vFloat xo   = x_in + sfpi::vFloat(LUT_RR_INPUT_OFFSET);
    const sfpi::vFloat m_v  = rr_mantissa(xo);
#if LUT_RR_LOG2_BASIS_MMINUS1
    r_arg = m_v - sfpi::vFloat(1.0f); // Horner on u = m-1 (c0 == 0)
#else
    r_arg = m_v;
#endif
#elif LUT_RR_METHOD == 6 // expalu_pow: |x| = 2^e * m, e = n*q + r
    r_arg = rr_mantissa(sfpi::abs(x_in));
#elif LUT_RR_METHOD == 7 // trig: k=round(x/pi); s = x - k*pi
    const sfpi::vFloat k_f = rr_int_to_float(rr_round_to_int(x_in * sfpi::vFloat(LUT_RR_TRIG_INV_PI)));
    // s = x - k*pi via fused FMA: (-pi)*k + x (no SFPADDI)
    r_arg = __builtin_rvtt_sfpmad(k_f.get(), sfpi::vFloat(-(float)(LUT_RR_TRIG_PI)).get(), x_in.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
#elif LUT_RR_METHOD == 8 // tan: j=round(x/(pi/2)); Cody-Waite a = x - j*(pi/2)
    const sfpi::vFloat j_f  = rr_int_to_float(rr_round_to_int(x_in * sfpi::vFloat(LUT_RR_TAN_INV_HALFPI)));
    // Cody-Waite stage 1: a_hi = (-PI2_HI)*j + x   (fused FMA)
    const sfpi::vFloat a_hi = __builtin_rvtt_sfpmad(j_f.get(), sfpi::vFloat(-(float)(LUT_RR_TAN_PI2_HI)).get(), x_in.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    // Cody-Waite stage 2: a    = (-PI2_LO)*j + a_hi (fused FMA)
    r_arg = __builtin_rvtt_sfpmad(j_f.get(), sfpi::vFloat(-(float)(LUT_RR_TAN_PI2_LO)).get(), a_hi.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
#endif

    // ---- Clamp r_arg to [b0, bN] (the REDUCED domain). For RR configs the
    // RAT_BOUNDARIES are the reduced-domain boundaries from the fitter CSV.
    const sfpi::vFloat b_lo = sfpi::vFloat(RAT_BOUNDARIES[0]);
    const sfpi::vFloat b_hi = sfpi::vFloat(RAT_BOUNDARIES[RAT_NUM_SEGMENTS]);

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

    // Numerator and denominator of segment 0, then override with the correct
    // segment via the template-recursive cumulative chain (see select_segment).
    // BOTH P and Q are evaluated at the SAME reduced arg.
    sfpi::vFloat p = eval_poly_seg<RAT_NUM_COEFFS, 0, RAT_NUM_DEGREE>(x_clamped);
    sfpi::vFloat q = eval_poly_seg<RAT_DEN_COEFFS, 0, RAT_DEN_DEGREE>(x_clamped);
    select_segment<1>(p, q, x_clamped);

    // result = P(x) / Q(x) = P(x) * (1 / Q(x)) using the iterative sfpi
    // reciprocal (silicon-representative; see file header), evaluated at the
    // reduced arg.
    //
    // Note: _sfpu_reciprocal_ does the Newton-Raphson step "x*y - vConstFloatPrgm0"
    // (vConstFloatPrgm0 == 2.0). At -O3 the Tensix instruction-combine pass folds
    // this into an SFPADDI immediate, which the Quasar ttsim build implements only
    // as a throwing stub (MissingSpecification: tensix_execute_sfpaddi); high-degree
    // single-segment configs (e.g. n10d10_s1) emit it and abort. The Python test
    // compiles this source with -mno-tt-tensix-optimize-combine so the subtract
    // stays a register-operand SFPADD instead — same numerics, ttsim-supported.
    sfpi::vFloat inv_q  = _sfpu_reciprocal_<2>(q);
    sfpi::vFloat result = p * inv_q;

    // ---- RECONSTRUCT (full activation over the original domain). The exponent
    // /quotient reduce-state is RECOMPUTED here from x_in (see the REGISTER-
    // PRESSURE NOTE in REDUCE): the P*recip(Q) eval above has freed its LREGs,
    // so recomputing now keeps the simultaneous live-vFloat count within the 8
    // SFPU LREGs. Numerically identical to holding it across the eval.
#if LUT_RR_METHOD == 1 // log: e*ln2 + P(m)
    const sfpi::vFloat rr_ef = rr_exp_float(x_in);
    result = __builtin_rvtt_sfpmad(rr_ef.get(), sfpi::vFloat(LUT_RR_LOG_LN2).get(), result.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    v_if (x_in == sfpi::vFloat(0.0f))
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
#elif LUT_RR_METHOD == 2 // exp: 2^q * P(s)
    const sfpi::vInt q_i = rr_round_to_int(x_in * sfpi::vFloat(LUT_RR_EXP_MULT));
    result               = rr_ldexp(result, q_i);
#elif LUT_RR_METHOD == 3 // cbrt: sign * C[r] * 2^q * P(m)
    const sfpi::vInt e_i = sfpi::exexp(sfpi::abs(x_in), sfpi::ExponentMode::Debias);
    sfpi::vInt       q_i, r_i;
    rr_divmod_floor(e_i, 3, q_i, r_i);
    result = rr_scale_by_r(result, r_i);
    result = rr_ldexp(result, q_i);
    result = sfpi::copysgn(result, x_in); // odd function: restore sign
    v_if (sfpi::abs(x_in) == sfpi::vFloat(0.0f))
    {
        result = sfpi::vFloat(0.0f);
    }
    v_endif;
#elif LUT_RR_METHOD == 4 // expalu_exp2: 2^i * P(f) [+ compose]
    const sfpi::vInt i_i = rr_floor_to_int(x_in * sfpi::vFloat(LUT_RR_EXP2_MULT));
    result               = rr_ldexp(result, i_i);
#if LUT_RR_COMPOSE == 1 // sigmoid: 1/(1+y)
    result = rr_recip(result + sfpi::vFloat(1.0f));
#elif LUT_RR_COMPOSE == 2 // minus_one: y - 1
    result = result - sfpi::vFloat(1.0f);
#endif
#elif LUT_RR_METHOD == 5 // expalu_log2: scale*(e + P(u))
    const sfpi::vFloat rr_ef = rr_exp_float(x_in + sfpi::vFloat(LUT_RR_INPUT_OFFSET));
    result                   = (rr_ef + result) * sfpi::vFloat(LUT_RR_LOG2_SCALE);
#elif LUT_RR_METHOD == 6 // expalu_pow: [1/](sign) * C[r] * 2^q * P(m)
    const sfpi::vInt e_i = sfpi::exexp(sfpi::abs(x_in), sfpi::ExponentMode::Debias);
    sfpi::vInt       q_i, r_i;
    rr_divmod_floor(e_i, LUT_RR_POW_N, q_i, r_i);
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
#elif LUT_RR_METHOD == 7 // trig: result = (-1)^k * P(s)
    // sign_flip = +1 (even k) / -1 (odd k). Recompute k parity from x_in.
    const sfpi::vFloat trig_sign = rr_parity_f(rr_round_to_int(x_in * sfpi::vFloat(LUT_RR_TRIG_INV_PI)));
    v_if (trig_sign == sfpi::vFloat(1.0f))
    {
        result = -result;
    }
    v_endif;
#elif LUT_RR_METHOD == 8 // tan: j even -> P(a); j odd -> -1/P(a)
    const sfpi::vFloat tan_odd   = rr_parity_f(rr_round_to_int(x_in * sfpi::vFloat(LUT_RR_TAN_INV_HALFPI)));
    const sfpi::vFloat tan_recip = -rr_recip(result); // -1/P(a), computed on all lanes
    v_if (tan_odd == sfpi::vFloat(1.0f))
    {
        result = tan_recip;
    }
    v_endif;
#endif

    sfpi::dst_reg[0] = result;
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_rational_lut()
{
// #pragma GCC unroll 1 (not 8): the poly kernel found unroll-8 mis-schedules the
// SFPU iteration loop at high segment counts; keep it at 1 here for the same
// reason.
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++)
    {
        piecewise_rational_lut_sfp_rows();
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

    // Program vConstFloatPrgm0 = 2.0f, used by the Newton-Raphson step in
    // _sfpu_reciprocal_ (APPROXIMATION_MODE = false -> full NR refinement).
    ckernel::sfpu::_init_sfpu_reciprocal_<false>();

    // Apply the embedded piecewise-rational LUT in-place on Dest for each tile.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_rational_lut<SFPU_ITERATIONS>, params.DST_INDEX + i);
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
