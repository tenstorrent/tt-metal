// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the sign op (storm contract,
// fresh_cpp/README.md).  Production: metal ckernel_sfpu_sign.h
// calculate_sign — already predicate-shaped but pinned with #pragma GCC
// unroll 0 and the legacy _sfpu_is_fp16_zero_ helper (a plain == 0.0f
// compare despite the name).  Semantic statement of the golden
// (torch.sign): -1 where v < 0, 0 where v == 0 (covers -0.0), +1
// otherwise.  Results are exactly representable in every Dest format, so
// no store rounding is needed (the production kernel has none either).
#include <cstdint>

namespace ckernel::sfpu
{

// NOTE on NaN: the "+1 otherwise" default is a statement about the swept
// domain, not about torch.sign, which returns NaN for NaN.  NaN is outside
// that domain (sfpu_domains.py SPECIALS_READY_OPS is empty); see
// fresh_cpp/comp.h finding GH-F1 for how the sfpi float compares lower
// against quiet NaN before relying on either branch there.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_sign_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r       = 1.0f;
        v_if (v < 0.0f)
        {
            r = -1.0f;
        }
        v_elseif (v == 0.0f)
        {
            r = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
