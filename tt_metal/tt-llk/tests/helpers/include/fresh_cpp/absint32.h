// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `absint32` corpus row (metal
// calculate_abs_int32).  Mathematical definition (torch.abs over int32):
// |x| exactly; the golden's stimuli exclude INT32_MIN, whose magnitude is
// unrepresentable.  The typed vInt Dst view carries the representation
// contract; the body states only the value-level negation-select.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_abs_int32_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vInt v = sfpi::dst_reg[0];
        sfpi::vInt r       = v;
        v_if (v < 0)
        {
            r = sfpi::vInt(0) - v;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
