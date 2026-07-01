// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-24_trigonometry_quasar_6e898d97
//
// Trigonometry SFPU kernel in the sfpi:: value-typed DSL. sine/cosine use a
// Maclaurin polynomial with argument reduction; acosh/asinh/atanh use an inline
// 3rd-order log polynomial. The compiler auto-replays the per-element body.
#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

using namespace sfpi;

namespace trig_const {
constexpr float INV_PI = 0.31830988618f;  // 1/pi
constexpr float PI = 3.14159265359f;
constexpr float HALF = 0.5f;
constexpr float TWO = 2.0f;

// Round-to-nearest snap bias: 1.5 * 2^23 (fp16b/bf16-exact -> single immediate).
constexpr float RNE_BIAS = 12582912.0f;

// Sine Maclaurin: sin(z) = z * (1 + w*(C3 + w*(C5 + w*(C7 [+ w*(C9 + w*C11)]))))  with w = z^2
constexpr float SIN_C3 = -1.0f / 6.0f;
constexpr float SIN_C5 = 1.0f / 120.0f;
constexpr float SIN_C7 = -1.0f / 5040.0f;
constexpr float SIN_C9 = 1.0f / 362880.0f;
constexpr float SIN_C11 = -1.0f / 39916800.0f;

// Cosine Maclaurin: cos(z) = 1 + w*(C2 + w*(C4 + w*(C6 [+ w*(C8 + w*C10)])))  with w = z^2
constexpr float COS_C2 = -1.0f / 2.0f;
constexpr float COS_C4 = 1.0f / 24.0f;
constexpr float COS_C6 = -1.0f / 720.0f;
constexpr float COS_C8 = 1.0f / 40320.0f;
constexpr float COS_C10 = -1.0f / 3628800.0f;

// 3rd-order log(m) minimax on mantissa m in [1,2): ((A*m + B)*m + C)*m + D
constexpr float LOG_A = 0.14081f;
constexpr float LOG_B = -0.86883f;
constexpr float LOG_C = 2.28790f;
constexpr float LOG_D = -1.58710f;
constexpr float LN2 = 0.69314718f;
constexpr float EXP_BIAS = 127.0f;  // fp exponent bias, subtracted to debias
constexpr int FP32_EXP_BIAS = 127;  // imm form for setexp (normalize mantissa into [1,2))
}  // namespace trig_const

// Truncate an fp32 constant to fp16b and materialize it in a vFloat via a single
// fp16b SFPLOADI (the same single-immediate lowering the raw-TTI baseline used).
sfpi_inline vFloat _c16_(float f) { return vFloat(sFloat16b(f)); }

// Shared init for every trig op: program ADDR_MOD_6 with a dest increment of one
// SFPU pass (2 rows on Quasar) so each op's SFPSTORE auto-advances the Dest
// counter to the next row-pair — no per-iteration _incr_counters_ needed. Every
// trig body has the same load/compute/store shape, so one addrmod serves all.
// (No LUT/const preloads: every coefficient fits a single fp16b immediate.)
inline void init_trigonometry() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = ckernel::math::SFP_ROWS},
    }
        .set(ckernel::ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

// round-to-nearest-even of a float via the additive-bias trick (exact for
// |y| < 2^22, which covers the sin/cos argument-reduction domain).
sfpi_inline vFloat _trig_round_(vFloat y) {
    vFloat bias = _c16_(trig_const::RNE_BIAS);
    vFloat r = y + bias;
    r = r - bias;
    return r;
}

// Inline 3rd-order log: log(x) = (biased_exp - 127)*ln2 + poly(mantissa).
// The int32->fp32 cast uses the SFPCAST builtin directly (what int32_to_float
// lowers to) to avoid the deprecated wrapper; 0..255 exponents cast exactly.
sfpi_inline vFloat _trig_log_(vFloat x) {
    using namespace trig_const;
    vInt e_biased = exexp(x, ExponentMode::Biased);
    vFloat exp_f = vFloat(__builtin_rvtt_sfpcast(e_biased.get(), SFPCAST_MOD1_INT32_TO_FP32_RNE)) - _c16_(EXP_BIAS);
    vFloat m = setexp(x, FP32_EXP_BIAS);
    vFloat series = m * _c16_(LOG_A) + _c16_(LOG_B);
    series = series * m + _c16_(LOG_C);
    series = series * m + _c16_(LOG_D);
    return exp_f * _c16_(LN2) + series;
}

// -----------------------------------------------------------------------------
// One SFPU pass (2 rows on Quasar) of any trig / inverse-hyperbolic op. The op
// is selected at compile time by OPERATION, so only the relevant branch is
// emitted per instantiation:
//   sine/cosine : y = x/pi; whole = round(y); z = pi*(y - whole);
//                 out = {sin,cos}(z) via Maclaurin; negate if whole is odd.
//   acosh(x)    = log(x + sqrt(x^2 - 1))            (NaN for x < 1 via sqrt)
//   asinh(x)    = sign(x) * log(|x| + sqrt(x^2 + 1))
//   atanh(x)    = 0.5 * log((1 + x) / (1 - x))
// -----------------------------------------------------------------------------
template <SfpuType OPERATION, bool APPROXIMATION_MODE>
inline void _calculate_trigonometry_sfp_rows_() {
    using namespace trig_const;
    vFloat x = dst_reg[0];

    if constexpr (OPERATION == SfpuType::sine || OPERATION == SfpuType::cosine) {
        // Shared argument reduction into z = pi * frac(x/pi).
        vFloat y = x * _c16_(INV_PI);
        vFloat whole = _trig_round_(y);
        vFloat z = (y - whole) * _c16_(PI);
        vFloat w = z * z;

        vFloat out;
        if constexpr (OPERATION == SfpuType::sine) {
            // Horner polynomial for sin(z)/z.
            vFloat poly;
            if constexpr (APPROXIMATION_MODE) {
                poly = w * _c16_(SIN_C7) + _c16_(SIN_C5);
            } else {
                poly = w * _c16_(SIN_C11) + _c16_(SIN_C9);
                poly = poly * w + _c16_(SIN_C7);
                poly = poly * w + _c16_(SIN_C5);
            }
            poly = poly * w + _c16_(SIN_C3);
            poly = poly * w + vConst1;
            out = z * poly;
        } else  // cosine
        {
            vFloat poly;
            if constexpr (APPROXIMATION_MODE) {
                poly = w * _c16_(COS_C6) + _c16_(COS_C4);
            } else {
                poly = w * _c16_(COS_C10) + _c16_(COS_C8);
                poly = poly * w + _c16_(COS_C6);
                poly = poly * w + _c16_(COS_C4);
            }
            poly = poly * w + _c16_(COS_C2);
            out = poly * w + vConst1;
        }

        // parity(whole) = whole - 2*round(whole/2); negate out when odd.
        vFloat parity = whole - _trig_round_(whole * _c16_(HALF)) * _c16_(TWO);
        v_if(parity != 0.0f) { out = -out; }
        v_endif;

        dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = out;
    } else if constexpr (OPERATION == SfpuType::acosh) {
        vFloat inner = x + approx_sqrt(x * x + vConstNeg1);  // x*x - 1 fused into one MAD
        dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = _trig_log_(inner);
    } else if constexpr (OPERATION == SfpuType::asinh) {
        vFloat inner = abs(x) + approx_sqrt(x * x + vConst1);  // x*x + 1 fused into one MAD
        vFloat out = _trig_log_(inner);
        v_if(x < 0.0f) { out = -out; }
        v_endif;
        dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = out;
    } else if constexpr (OPERATION == SfpuType::atanh) {
        vFloat num = x + vConst1;                // 1 + x
        vFloat den = vConst1 - x;                // 1 - x
        vFloat ratio = num * approx_recip(den);  // (1 + x) / (1 - x)
        dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = _trig_log_(ratio) * _c16_(HALF);
    }
}

/**
 * @brief Apply a trigonometry / inverse-hyperbolic SFPU op over a full Dest tile in place.
 *
 * OPERATION selects the math at compile time; each iteration runs one SFPU pass (2 rows on
 * Quasar) and stores via ADDR_MOD_6, which auto-advances the Dest counter — so there is no
 * per-iteration counter bump.
 *
 * @tparam OPERATION: Trig op to apply, values = <sine/cosine/acosh/asinh/atanh>
 * @tparam APPROXIMATION_MODE: If true, use the shorter lower-order sine/cosine polynomial; ignored by
 * acosh/asinh/atanh.
 * @tparam ITERATIONS: Number of SFPU passes spanning the tile (default = SFPU_ITERATIONS).
 * @note Call @ref init_trigonometry once before this to program the ADDR_MOD_6 the store relies on.
 * @ref _calculate_trigonometry_sfp_rows_ is the single-pass body this loops over (per-op formulas there).
 */
template <SfpuType OPERATION, bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_trigonometry() {
    static_assert(
        OPERATION == SfpuType::sine || OPERATION == SfpuType::cosine || OPERATION == SfpuType::acosh ||
            OPERATION == SfpuType::asinh || OPERATION == SfpuType::atanh,
        "calculate_trigonometry: OPERATION must be a trigonometry SfpuType (sine/cosine/acosh/asinh/atanh)");
    static_assert(
        !APPROXIMATION_MODE || OPERATION == SfpuType::sine || OPERATION == SfpuType::cosine,
        "calculate_trigonometry: APPROXIMATION_MODE only affects sine/cosine; pass false for acosh/asinh/atanh");

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_trigonometry_sfp_rows_<OPERATION, APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
