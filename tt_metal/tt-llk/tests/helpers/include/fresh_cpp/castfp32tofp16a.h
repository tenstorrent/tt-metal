// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `castfp32tofp16a` corpus row (metal
// cast_fp32_to_fp16a).  Mathematical definition (the golden's own model):
// round each lane's fp32 bit pattern to the fp16a MANTISSA — keep the top 10
// fraction bits, dropping 13, with round-half-to-even — while the exponent
// stays fp32-range (no fp16 clamp; a mantissa carry may ripple into the
// exponent, which is correct).  Non-finite inputs (exponent 0xFF) pass
// through unchanged.  Stated directly on the integer view with the classic
// RNE increment bits + 0x0FFF + kept-LSB.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_cast_fp32_to_fp16a_fresh_cpp()
{
    constexpr int EXPONENT_FIELD = 0x7F800000;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        const sfpi::vUInt u  = sfpi::as<sfpi::vUInt>(v);

        // Round-half-to-even on the dropped 13 mantissa bits: adding
        // (halfway - 1) + kept-LSB rounds up exactly when the remainder is
        // above halfway, or equal to halfway with an odd kept bit.
        const sfpi::vUInt kept_lsb = (u >> 13) & 1;
        sfpi::vUInt rounded        = u + 0x0FFF + kept_lsb;
        rounded                    = rounded & 0xFFFFE000u;
        sfpi::vFloat r             = sfpi::as<sfpi::vFloat>(rounded);

        // Non-finite input (inf/nan): pass the bit pattern through unchanged.
        const sfpi::vInt exponent_bits = sfpi::as<sfpi::vInt>(v) & EXPONENT_FIELD;
        v_if (exponent_bits == EXPONENT_FIELD)
        {
            r = v;
        }
        v_endif;

        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
