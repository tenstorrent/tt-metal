// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// erfc — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// erfc(x) = 1 - erf(x), stated through the shared fresh_erf_core fit
// (fresh_cpp/erf.h — the header of record for the polynomial and its
// derivation evidence).  The core's clamp keeps the tails at exactly 0 / 2.
// Golden: torch.erfc (golden_generators._erfc), Float32 corr contract; the
// suite's atol 0.05 dominates the far-tail residue (erfc(3) ~= 2.2e-5).
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
