// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS=4, int RECIPROCAL_ITERATIONS=2>
inline void calculate_sqrt()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (APPROXIMATION_MODE)
        {
            vUInt magic = l_reg[LRegs::LReg2];

            //sqrt initial approximation
            // adjust bias
            vUInt val_s = magic + reinterpret<vUInt>(val);

            // approximation of square root
            val_s >>= 1;
            dst_reg[0] = reinterpret<vFloat>(val_s);

            l_reg[LRegs::LReg2] = magic;
        }
        else
        {
            // Recip root method
            //// Init approx
            //u.i = SQRT_MAGIC_F - (u.i >> 1);
            vUInt magic = reinterpret<vUInt>(vFloat(s2vFloat16b(0x5f37)));
            vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));

            // Re-load to save a MOV
            val = dst_reg[0];

            //Reciproot iterations
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                //x*r*(1.5f - xhalf*r*r);
                approx = (approx * approx * val * vConstNeg0p5 + vConst1 + 0.5F) * approx;
            }

            dst_reg[0] = approx * val;
        }

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void sqrt_init() {
    if (APPROXIMATION_MODE) {
        TTI_SFPLOADI(2, 0, 127 << 7);
    }
}
} // namespace sfpu
} // namespace ckernel
