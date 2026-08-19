// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the selu op (storm contract,
// fresh_cpp/README.md).  Production: metal ckernel_sfpu_selu.h — the
// negative branch runs expm1_cw_clamped (tt_llk ckernel_sfpu_expm1_cw.h
// Cody-Waite: k extracted through raw bit reads of the 1.5*2^23 rounding
// bias, the fused kC231Bias 0x4B3FFF81 exponent trick, PolynomialEvaluator
// text-template Horner) under an unroll-2 pin.  Semantic statement of the
// golden (torch.nn.functional.selu):
//
//   selu(x) = scale * x                    for x >= 0
//           = scale * alpha * expm1(x)     for x <  0
//
// with the production's OWN Cody-Waite constants (the golden tolerance is
// fitted to them): x clamped at -87, k = round(x/ln2) via the shared
// rounding-bias helper (fresh_cpp/helpers.h fresh_round_nearest), the
// split-ln2 residual, the degree-4 bf16-arm polynomial (INP_FLOAT32 is
// never defined in this build), and 2^k by setexp; expm1 = (2^k - 1) +
// 2^k * h.  The bf16 RNE store is the production
// !is_fp32_dest_acc_en arm (the measured row's variant).
#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fixed dispatch scalars shared with the golden and identical to the values
// the production dispatch sends (sfpu_operations.h 0x3f867d5fu/0x3fd62d7du —
// the fp32 roundings of torch's selu scale/alpha defaults).  Both legs must
// always receive identical values.
constexpr std::uint32_t FRESH_SELU_SCALE = 0x3f867d5fu; // 1.0507010221481323f
constexpr std::uint32_t FRESH_SELU_ALPHA = 0x3fd62d7du; // 1.6732631921768188f

template <int ITERATIONS>
__attribute__((noinline)) void calculate_selu_fresh_cpp(const std::uint32_t scale, const std::uint32_t alpha)
{
    // Cody-Waite expm1 constants (production expm1_cw_clamped, bf16 arm).
    constexpr float CW_INV_LN2    = 1.4426950408889634f;
    constexpr float CW_NEG_LN2_HI = -0.6931152343750000f;
    constexpr float CW_NEG_LN2_LO = -3.19461832987e-05f;
    constexpr float P4            = 8.3751315251e-03f;
    constexpr float P3            = 4.1875664145e-02f;
    constexpr float P2            = 1.6666433215e-01f;
    constexpr float P1            = 4.9999371171e-01f;

    const float scale_f     = Converter::as_float(scale);
    const float scale_alpha = scale_f * Converter::as_float(alpha);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];

        // expm1(x), Cody-Waite: x = k*ln2 + r, expm1 = (2^k - 1) + 2^k * (e^r - 1).
        const sfpi::vFloat xc = sfpi::max(x, -87.0f);
        sfpi::vInt k_int;
        const sfpi::vFloat k     = fresh_round_nearest(xc * CW_INV_LN2, k_int);
        sfpi::vFloat r           = k * CW_NEG_LN2_HI + xc;
        r                        = k * CW_NEG_LN2_LO + r;
        sfpi::vFloat h           = ((P4 * r + P3) * r + P2) * r + P1;
        h                        = (h * r + 1.0f) * r; // e^r - 1
        const sfpi::vFloat two_k = sfpi::setexp(sfpi::vFloat(1.0f), k_int + 127);
        const sfpi::vFloat em1   = (two_k - 1.0f) + two_k * h;

        sfpi::vFloat result = scale_alpha * em1;
        v_if (x >= 0.0f)
        {
            result = scale_f * x;
        }
        v_endif;
        result           = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
