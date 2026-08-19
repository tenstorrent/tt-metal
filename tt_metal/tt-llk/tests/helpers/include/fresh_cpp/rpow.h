// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the rpow op (storm contract,
// fresh_cpp/README.md).  Production: metal ckernel_sfpu_rpow.h over
// _sfpu_binary_power_21f_ — the generic base**x machine: a cubic log2
// minimax on the base's mantissa, the exp_21f magic-integer recombination
// spelled through _float_to_int32_positive_ bit offsets, programmed
// constant registers (1/ln2, -127), and the negative-base parity fixups.
// The dispatched base is the CONSTANT 2.0 (0x40000000u, mirrored by the
// golden's _RPOW_BASE), so the semantic statement is direct:
//
//   rpow(x) = 2**x = 2**xi * 2**xf,
//
// the exp_21f exponent/mantissa recombination already established in
// calculate_exp_fresh_cpp with the x/ln2 step removed (log2(2**x) = x).
// Same golden-fitted quadratic refinement constants; same saturation
// above (biased exponent capped at 255) and underflow-to-zero below
// (zeroed exponent source flushes through the bf16 store); same bf16 RNE
// store as the production !is_fp32_dest_acc_en arm (the measured row's
// variant).
#include <cstdint>

namespace ckernel::sfpu
{

// Fixed dispatch base shared with the golden and the production dispatch
// (sfpu_operations.h 0x40000000u == golden_generators _RPOW_BASE = 2.0).
constexpr float FRESH_RPOW_BASE = 2.0f;

template <int ITERATIONS>
__attribute__((noinline)) void calculate_rpow_fresh_cpp()
{
    constexpr float C0 = 1.0017248f;
    constexpr float C1 = 7.839635491371155e-08f;
    constexpr float C2 = 4.791750143340323e-15f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];

        // Biased result exponent; fixed-point encoding of it by mantissa
        // (implicit one) shifted left by the unbiased exponent.
        sfpi::vFloat xlog2   = x + 127.0f;
        xlog2                = sfpi::min(xlog2, 255.0f);
        sfpi::vInt zi        = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
        const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);

        // Quadratic refinement of 2**xf on [0, 1).
        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
        frac              = (C2 * frac + C1) * frac + C0;

        // Underflow: zero the exponent source where the biased exponent is
        // not positive; the zero exponent flushes through the bf16 store.
        sfpi::vFloat zc = z;
        v_if (xlog2 <= 0.0f)
        {
            zc = 0.0f;
        }
        v_endif;

        sfpi::vFloat y   = sfpi::setexp(frac, sfpi::exexp(zc, sfpi::ExponentMode::Biased));
        y                = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
