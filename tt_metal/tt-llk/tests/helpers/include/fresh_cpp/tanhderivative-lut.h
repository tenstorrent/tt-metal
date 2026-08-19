// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the tanhderivative-lut op (storm contract:
// fresh_cpp/README.md).  Migrated verbatim from fresh_cpp_operations.h
// (Lane BR batch 1); byte-stable algorithm, only the file moved.

#include <cstdint>

namespace ckernel::sfpu
{

// Tanh-derivative, legacy LUT contract (production: _calculate_tanh_derivative_
// pins l_reg[LReg0..2] across the tile and consumes the raw SFPLUT programmed
// by tanh_derivative_init's TT_SFPLOADI words).  The row's golden IS the
// 3-region piecewise-linear tanh (breakpoints 1.0/2.0, slopes 0.90625 and
// 0.09375x+0.8125, saturation 1.0), so the faithful semantic statement is the
// same piecewise dataflow as typed v_if regions (the sigmoidappx-tree
// precedent), then 1 - t^2.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_derivative_lut_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat t       = 1.0f;
        v_if (a < 1.0f)
        {
            t = a * 0.90625f;
        }
        v_elseif (a < 2.0f)
        {
            t = a * 0.09375f + 0.8125f;
        }
        v_endif;
        sfpi::dst_reg[0] = t * (-t) + 1.0f;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
