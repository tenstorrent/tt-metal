// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the tanh op (storm contract: fresh_cpp/README.md).
// Migrated verbatim from fresh_cpp_operations.h (Lane BR batch 1); byte-stable
// algorithm, only the file moved.

#include <cstdint>

namespace ckernel::sfpu
{

// Tanh, bf16 non-approx contract (production: _sfpu_tanh_polynomial_x2_ — an
// explicit two-datum hand software pipeline with three coefficients parked in
// programmed constant registers and a scalar epilogue).  Same Sollya
// polynomial (the golden math), one datum per row, every coefficient a plain
// local: pipelining, unrolling, and constant residency are the compiler's.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_fresh_cpp()
{
    constexpr float C1 = 0.999004364013671875f;
    constexpr float C2 = 3.0897438526153564453125e-2f;
    constexpr float C3 = -0.4890659749507904052734375f;
    constexpr float C4 = 0.281917631626129150390625f;
    constexpr float C5 = -6.6649019718170166015625e-2f;
    constexpr float C6 = 5.876733921468257904052734375e-3f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat r       = C6;
        r                    = r * a + C5;
        r                    = r * a + C4;
        r                    = r * a + C3;
        r                    = r * a + C2;
        r                    = r * a + C1;
        r                    = r * a;
        r                    = sfpi::min(r, 1.0f);
        r                    = sfpi::copysgn(r, x);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
