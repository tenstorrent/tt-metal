// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// STANDALONE exponent-ALU eval method on Quasar (sim-qsr / ttsim).
//
// This is the DISTINCT eval method EVAL_METHOD_EXPONENT_ALU: a single
// reduced-domain Horner that BYPASSES the segment cascade entirely. It is NOT
// the in-cascade reduce+reconstruct framing that the poly / rational quasar
// tests carry as LUT_RR_METHOD 4/5/6 (those still clamp + select segments).
// Here the whole activation comes from the bit-decompose of the input via the
// hardware exponent ALU (exexp / exman(ImplicitOne) / setexp / shft) and ONE
// degree-DEG Horner over the fitter's reduced-domain coefficients.
//
// It is a faithful Quasar port of the three DEPLOYED Blackhole evaluators from
//   tt_metal/programming_examples/generic_lut_activation_embedded/kernels/
//   compute/piecewise_generic.cpp
// mirrored in the BH craq-sim harnesses
//   sources/generic_lut_{exp2,expalu,log2alu,pow}_bh_test.cpp:
//
//   EXPALU_MODE == 1  exp_hw_eval<DEG>  (exp / exp2; EXPONENT_ALU_EXP2)
//                     xlog2 = x*MULT + 127; branch-free float->int via
//                     exman(ImplicitOne)+shft(Logical); ep=2^i, f in [0,1);
//                     p = Horner(2^f); y = setexp(p, ep + exexp(p) - 127).
//                     Optional COMPOSE: sigmoid 1/(1+y) or expm1 y-1.
//   EXPALU_MODE == 2  log_hw_eval<DEG>  (log / log2 / log10 / log1p;
//                     EXPONENT_ALU_LOG2). x = 2^e * m, m in [1,2):
//                     log_b(x) = SCALE*(e + h(m)), h(m)=log2(m) on [1,2).
//                     Optional INPUT_OFFSET (log1p -> x+1) and m_minus_1 basis.
//   EXPALU_MODE == 3  pow_hw_eval<DEG>  (sqrt / rsqrt / cbrt via root_N;
//                     EXPONENT_ALU_POW). |x| = 2^e*m, e = N*q + r:
//                     root_N(x) = sign * C[r] * 2^q * p(m). Optional RECIPROCAL.
//
// The Blackhole sfpi evaluator bodies port to Quasar essentially verbatim: the
// SFPU intrinsics (SFPEXEXP / SFPEXMAN / SFPSHFT / SFPSETEXP / SFP_STOCH_RND)
// and the sfpi wrappers (exexp / exman / shft / setexp / convert / as<> /
// int32_to_float / reinterpret / vec_min_max) are shared (runtime/sfpi). The
// ONLY Quasar-specific changes are the LLK harness (semantic unpack/pack
// headers, FPU MOVA2D datacopy, 2-rows-per-iteration SFPU model advanced by
// ckernel::math::_incr_counters_<...SFP_ROWS...>(), and the SFPADDI-avoidance
// Horner step that the rational/poly quasar tests already use). See
// sources/quasar/generic_lut_rational_quasar_test.cpp for the BH->Quasar SFPU
// translation reference.
//
// LUT injected at build time via #defines emitted into build.h (-> params.h) by
// the Python driver test_generic_lut_expalu_quasar.py:
//   EXPALU_MODE, EXPALU_DEG, EXPALU_COEFFS_INIT (c0..cDEG, low-order first)
//   mode 1: EXPALU_EXP_MULT [, EXPALU_COMPOSE]
//   mode 2: EXPALU_LOG_SCALE [, EXPALU_LOG_BASIS_MMINUS1] [, EXPALU_LOG_OFFSET]
//   mode 3: EXPALU_POW_N, EXPALU_POW_C0/C1/C2 [, EXPALU_POW_RECIP]

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

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// ---- Injected LUT defaults (overridden by build.h from the Python driver) ----
#ifndef EXPALU_MODE
#define EXPALU_MODE 1 // 1 exp_hw, 2 log_hw, 3 pow_hw
#endif
#ifndef EXPALU_DEG
#define EXPALU_DEG 2
#endif
#ifndef EXPALU_COEFFS_INIT
#define EXPALU_COEFFS_INIT 1.0f, 1.0f, 1.0f
#endif
// mode 1 (exp)
#ifndef EXPALU_EXP_MULT
#define EXPALU_EXP_MULT 1.4426950216293334961f // log2(e), exp(x)
#endif
#ifndef EXPALU_COMPOSE
#define EXPALU_COMPOSE 0 // 0 none, 1 sigmoid 1/(1+y), 2 minus_one y-1
#endif
// mode 2 (log)
#ifndef EXPALU_LOG_SCALE
#define EXPALU_LOG_SCALE 1.0f
#endif
#ifndef EXPALU_LOG_BASIS_MMINUS1
#define EXPALU_LOG_BASIS_MMINUS1 0
#endif
#ifdef EXPALU_LOG_OFFSET
#define EXPALU_LOG_HAS_OFFSET 1
#else
#define EXPALU_LOG_OFFSET 0.0f
#endif
// mode 3 (pow)
#ifndef EXPALU_POW_N
#define EXPALU_POW_N 2
#endif
#ifndef EXPALU_POW_C0
#define EXPALU_POW_C0 1.0f
#endif
#ifndef EXPALU_POW_C1
#define EXPALU_POW_C1 1.0f
#endif
#ifndef EXPALU_POW_C2
#define EXPALU_POW_C2 1.0f
#endif

constexpr std::uint32_t EXPALU_DEGREE                        = EXPALU_DEG;
constexpr std::array<float, EXPALU_DEGREE + 1> EXPALU_COEFFS = {EXPALU_COEFFS_INIT};

namespace
{
// SFPADDI avoidance (mirrors the poly / rational quasar kernels). The Horner
// step `acc * x + c` with a compile-time constant `c` otherwise lowers to a
// standalone SFPADD that the -O3 instruction-combine pass folds into an SFPADDI
// immediate for round constants. Quasar removed SFPADDI (ttsim aborts
// "tensix_execute_sfpaddi"); emitting one fused SFPMAD (c is the MAD addend)
// leaves no standalone SFPADD to fold. The Python driver additionally passes
// -mno-tt-tensix-optimize-combine to neutralise the reciprocal's own NR
// subtract for the sigmoid / rsqrt composes.
sfpi_inline sfpi::vFloat fma_const(sfpi::vFloat a, sfpi::vFloat b, float c)
{
    const sfpi::vFloat cv = sfpi::vFloat(c);
    return __builtin_rvtt_sfpmad(a.get(), b.get(), cv.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// Compile-time-indexed Horner over EXPALU_COEFFS (low-order-first). Reading the
// constexpr array ONLY through constexpr indices lets the compiler fold every
// coefficient into an SFPLOADI immediate and elide the array from .ldm_data
// (the tiny private TRISC1 region) — same fold the poly / rational kernels use.
template <int K>
sfpi_inline sfpi::vFloat horner_step(sfpi::vFloat acc, sfpi::vFloat x)
{
    if constexpr (K < 0)
    {
        return acc;
    }
    else
    {
        acc = fma_const(acc, x, EXPALU_COEFFS[static_cast<std::uint32_t>(K)]);
        return horner_step<K - 1>(acc, x);
    }
}

template <std::uint32_t DEG>
sfpi_inline sfpi::vFloat eval_horner(sfpi::vFloat x)
{
    const sfpi::vFloat acc = sfpi::vFloat(EXPALU_COEFFS[DEG]);
    return horner_step<static_cast<int>(DEG) - 1>(acc, x);
}

// Reciprocal 1/x for the sigmoid / rsqrt composes. Quasar sfpi has no vFloat
// operator/; use SFPNONLINEAR recip approximation + two Newton-Raphson steps
// (y <- y*(2 - x*y)), formed with fused FMA (no SFPADDI). Same idiom as the
// rational kernel's rr_recip.
sfpi_inline sfpi::vFloat expalu_recip(sfpi::vFloat x)
{
    sfpi::vFloat y = sfpi::approx_recip(x);
#pragma GCC unroll 1
    for (int it = 0; it < 2; it++)
    {
        const sfpi::vFloat t = fma_const(x, y, -2.0f);                                                                              // t = x*y - 2
        y                    = __builtin_rvtt_sfpmad(y.get(), (-t).get(), sfpi::vFloat(0.0f).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE); // y*(2 - x*y)
    }
    return y;
}

// int -> float (vInt -> vSMag -> vFloat); the only int->float path on Quasar.
sfpi_inline sfpi::vFloat int_to_float(sfpi::vInt v)
{
    const auto s = sfpi::convert<sfpi::vSMag>(v);
    return sfpi::convert<sfpi::vFloat>(s, sfpi::RoundMode::Nearest);
}

// round-to-nearest float -> vInt. Quasar path: SFPSTOCHRND fp32->sm16 (RNE)
// then SFPCAST sm32->int32. Reduced exponent integers (|q|,|r|,|e| <= ~127) fit
// in 16 bits, so this is exact. (Used by the pow divmod below.)
sfpi_inline sfpi::vInt rr_round(sfpi::vFloat t)
{
    const sfpi::vSMag s = sfpi::vSMag(__builtin_rvtt_sfpstochrnd_i(t.get(), 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_SMAG16, sfpi::SFPSTOCHRND_RND_EVEN));
    return sfpi::convert<sfpi::vInt>(s);
}

#if EXPALU_MODE == 1
// ===== exp_hw_eval<DEG> (EXPONENT_ALU_EXP2) — Quasar port of the BH evaluator. =====
// exp(x) = 2^(x*MULT). t = x*MULT + 127, clamped [0,255]; integer part i and
// fraction f in [0,1); p = Horner(2^f); y = 2^(i-127) * p.
//
// SIM NOTE: the Blackhole evaluator does the float->int decompose with the
// exponent-ALU exman(ImplicitOne)+shft(Logical) + a second exman for the
// fraction. The PINNED craq-sim build (libttsim.so) does NOT implement SFPEXMAN
// (it aborts MissingSpecification: tensix_execute_sfpexman), so this port does
// the SAME decompose with the sim-supported floor/round path proven by the
// rational quasar kernel (SFPSTOCHRND + setexp-based ldexp). The math is
// identical: i = floor(t), f = t - i, 2^(i-127) = setexp(1.0, i). exexp / setexp
// / SFPSTOCHRND all run on this sim; SFPEXMAN does not.
template <std::uint32_t DEG>
sfpi_inline sfpi::vFloat exp_hw_eval(sfpi::vFloat x)
{
    constexpr float MULT = EXPALU_EXP_MULT;

    sfpi::vFloat t = fma_const(x, sfpi::vFloat(MULT), 127.0f); // x*MULT + 127

    // Full-range safety clamp: keep t in [0, 255].
    sfpi::vFloat thr_lo = sfpi::vFloat(0.0f);
    sfpi::vFloat thr_hi = sfpi::vFloat(255.0f);
    sfpi::vec_min_max(thr_lo, t); // t = max(0, t)
    sfpi::vec_min_max(t, thr_hi); // t = min(t, 255)

    // Integer part i = floor(t) (exp-biased), fraction f = t - i in [0,1).
    sfpi::vInt i = rr_round(t);
    v_if (int_to_float(i) > t) // correct round-to-nearest down to floor
    {
        i = i - sfpi::vInt(1);
    }
    v_endif;
    const sfpi::vFloat fi = int_to_float(i);
    // f = t - i via fused FMA (no SFPADDI): (-1)*fi + t
    sfpi::vFloat f = __builtin_rvtt_sfpmad(sfpi::vFloat(-1.0f).get(), fi.get(), t.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    sfpi::vFloat p = eval_horner<DEG>(f); // 2^f over natural [0,1) coeffs

    // y = 2^(i-127) * p. setexp(1.0, i) writes biased exponent i onto 1.0,
    // i.e. 2^(i-127) (i already carries the +127 bias from t = x*MULT + 127).
    const sfpi::vFloat two_pow_i = sfpi::setexp(sfpi::vFloat(1.0f), i);
    sfpi::vFloat y               = p * two_pow_i;

#if EXPALU_COMPOSE == 1 // sigmoid: 1/(1+y)  (y == exp(-x))
    y = expalu_recip(y + sfpi::vFloat(1.0f));
#elif EXPALU_COMPOSE == 2 // minus_one (expm1): y - 1
    y = y - sfpi::vFloat(1.0f);
#endif
    return y;
}
#endif // EXPALU_MODE == 1

#if EXPALU_MODE == 2
// ===== log_hw_eval<DEG> (EXPONENT_ALU_LOG2) — port of the BH evaluator. =====
// x = 2^e * m, m in [1,2): log_b(x) = SCALE*(e + h(m)), h(m)=log2(m) on [1,2).
template <std::uint32_t DEG>
sfpi_inline sfpi::vFloat log_hw_eval(sfpi::vFloat x)
{
#ifdef EXPALU_LOG_HAS_OFFSET
    constexpr float OFFSET = EXPALU_LOG_OFFSET;
    sfpi::vFloat xd        = x + sfpi::vFloat(OFFSET);
#else
    sfpi::vFloat xd = x;
#endif

    // Biased exponent e and mantissa m in [1,2) of the decompose input.
    sfpi::vInt biased = sfpi::exexp(xd, sfpi::ExponentMode::NoDebias);
    sfpi::vInt e_int  = biased - sfpi::vInt(127);
    sfpi::vFloat m    = sfpi::setexp(xd, 127);

#if EXPALU_LOG_BASIS_MMINUS1
    sfpi::vFloat u = m - sfpi::vFloat(1.0f);
#else
    sfpi::vFloat u = m;
#endif
    sfpi::vFloat h = eval_horner<DEG>(u);

    // e + h(m); int->float wants sign-magnitude for negatives.
    v_if (e_int < sfpi::vInt(0))
    {
        e_int = sfpi::setsgn(~e_int + sfpi::vInt(1), 1);
    }
    v_endif;
    sfpi::vFloat e_float = sfpi::int32_to_float(e_int, sfpi::RoundMode::Nearest);
    sfpi::vFloat result  = (e_float + h) * sfpi::vFloat(EXPALU_LOG_SCALE);

    // Special cases on the DECOMPOSE input (x + offset).
    v_if (xd < sfpi::vFloat(0.0f))
    {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif (xd == sfpi::vFloat(0.0f))
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
    return result;
}
#endif // EXPALU_MODE == 2

#if EXPALU_MODE == 3
// ===== pow_hw_eval<DEG> (EXPONENT_ALU_POW) — port of the BH evaluator. =====
// |x| = 2^e * m, e = N*q + r: root_N(x) = sign * C[r] * 2^q * p(m).
// Uses the float-domain magic-number divmod from the BH pow kernel, expressed
// with the Quasar int<->float path (int_to_float).
template <std::uint32_t DEG>
sfpi_inline sfpi::vFloat pow_hw_eval(sfpi::vFloat x)
{
    constexpr int ROOT_N    = EXPALU_POW_N;
    constexpr bool ODD_ROOT = (ROOT_N % 2 == 1);

    sfpi::vInt sign_bits = sfpi::reinterpret<sfpi::vInt>(x) & sfpi::vInt(0x80000000);
    sfpi::vFloat ax      = ODD_ROOT ? sfpi::setsgn(x, 0) : x;

    sfpi::vInt biased = sfpi::exexp(ax, sfpi::ExponentMode::NoDebias);
    sfpi::vInt e_int  = biased - sfpi::vInt(127);
    sfpi::vFloat m    = sfpi::setexp(ax, 127);

    sfpi::vFloat s = eval_horner<DEG>(m); // p(m) ~ root_N(m) on [1,2)

    // ef = (float)e_int with sign-magnitude handling for negatives.
    sfpi::vFloat ef = sfpi::int32_to_float(e_int, sfpi::RoundMode::Nearest);
    v_if (e_int < sfpi::vInt(0))
    {
        const sfpi::vInt mag = ~e_int + sfpi::vInt(1);
        ef                   = -sfpi::int32_to_float(mag, sfpi::RoundMode::Nearest);
    }
    v_endif;

    // Floored divmod e = N*q + r, r in [0,N). |e| <= 127 is exact in fp32.
    const sfpi::vFloat qf = ef * sfpi::vFloat(1.0f / (float)ROOT_N);
    // floor(qf): round-to-nearest then correct down (matches rr_floor_to_int).
    sfpi::vInt q = rr_round(qf);
    v_if (int_to_float(q) > qf)
    {
        q = q - sfpi::vInt(1);
    }
    v_endif;
    // r = e - N*q  (fused FMA in float, no SFPADDI), then back to int.
    const sfpi::vFloat qfb = int_to_float(q);
    const sfpi::vFloat rf  = __builtin_rvtt_sfpmad(qfb.get(), sfpi::vFloat(-(float)ROOT_N).get(), ef.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    sfpi::vInt r           = rr_round(rf);

    // Scale by C[r], r in {0,1,2}.
    v_if (r == sfpi::vInt(1))
    {
        s = s * sfpi::vFloat(EXPALU_POW_C1);
    }
    v_endif;
    if constexpr (ROOT_N >= 3)
    {
        v_if (r == sfpi::vInt(2))
        {
            s = s * sfpi::vFloat(EXPALU_POW_C2);
        }
        v_endif;
    }

    // result = 2^q * s.
    sfpi::vInt s_exp    = sfpi::exexp(s, sfpi::ExponentMode::NoDebias);
    sfpi::vFloat result = sfpi::setexp(s, s_exp + q);

#if defined(EXPALU_POW_RECIP)
    result = expalu_recip(result); // rsqrt
#endif

    if constexpr (ODD_ROOT)
    {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(result) | sign_bits);
        v_if (ax == sfpi::vFloat(0.0f))
        {
            result = sfpi::reinterpret<sfpi::vFloat>(sign_bits);
        }
        v_endif;
    }
    else
    {
        v_if (x < sfpi::vFloat(0.0f))
        {
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif (x == sfpi::vFloat(0.0f))
        {
#if defined(EXPALU_POW_RECIP)
            result = std::numeric_limits<float>::infinity();
#else
            result = sfpi::vFloat(0.0f);
#endif
        }
        v_endif;
    }
    return result;
}
#endif // EXPALU_MODE == 3

// Evaluate the STANDALONE exponent-ALU activation on the current 2-row Dest
// window (sfpi dst_reg[0]). No clamp, no boundaries, no segment select — the
// whole activation comes from the single reduced-domain Horner + bit-decompose.
sfpi_inline void expalu_lut_sfp_rows()
{
    const sfpi::vFloat x = sfpi::dst_reg[0];
#if EXPALU_MODE == 1
    sfpi::dst_reg[0] = exp_hw_eval<EXPALU_DEGREE>(x);
#elif EXPALU_MODE == 2
    sfpi::dst_reg[0] = log_hw_eval<EXPALU_DEGREE>(x);
#elif EXPALU_MODE == 3
    sfpi::dst_reg[0] = pow_hw_eval<EXPALU_DEGREE>(x);
#endif
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_expalu_lut()
{
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++)
    {
        expalu_lut_sfp_rows();
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

    // Apply the standalone exponent-ALU activation in-place on Dest per tile.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_expalu_lut<SFPU_ITERATIONS>, params.DST_INDEX + i);
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
