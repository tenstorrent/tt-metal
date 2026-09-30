// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// erfc — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// erfc(x) = 1 - erf(x), stated through the shared fresh_erf_core fit
// (fresh_cpp/erf.h — the header of record for the polynomial, its argument
// range reduction and the derivation evidence).  The core reduces |x| onto
// the fit domain and saturates the |x| >= 3 tail to the exact limit +/-1, so
// the far tails stay at exactly 0 and 2 as before — the difference is that
// they are now reached for the RIGHT sign of x.
// Before that reduction the out-of-domain polynomial made erfc(11) come back
// as 2.0 instead of 0.0 (1 - (-1): erf's sign inversion, propagated), 16384
// bf16 ULP from the golden for every input in the stratum.
// Golden: torch.erfc (golden_generators._erfc), Float32 corr contract.
#include <cstdint>

#include "erf.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_erfc_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::dst_reg[0] = 1.0f - fresh_erf_core(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
