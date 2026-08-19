// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the softsign op (storm contract:
// fresh_cpp/README.md).  Independent derivation from the PyTorch reference
// (torch.nn.functional.softsign — the production golden):
//
//   softsign(x) = x / (1 + |x|)
//
// The division is stated as multiplication by a Newton-refined reciprocal —
// the same numeric contract the production kernel meets (approx_recip + two
// refinement steps), but through the shared typed helper with its 2.0
// literal instead of the production's vConstFloatPrgm0 parking.

#include <cstdint>

// Shared helper (fresh_recip) still lives in the legacy header pending full
// migration (fresh_cpp/README.md legacy note).
#include "fresh_cpp_operations.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_softsign_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0]     = x * fresh_recip<2>(sfpi::abs(x) + 1.0f);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
