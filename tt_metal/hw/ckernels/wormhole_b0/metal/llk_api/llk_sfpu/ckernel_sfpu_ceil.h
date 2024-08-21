// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "noc_nonblocking_api.h"
#include "limits.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_ceil()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat result = dst_reg[0];
        vFloat v = result;
        vInt tmp = float_to_int16(result, 0); //TODO: Replace float_to_int16 to float_to_int32 once it is available
        result = int32_to_float(tmp, 0);
        v_if (result < v){
            result = result + 1;
        }
        v_endif;
        v_if (v <= SHRT_MIN || v >= SHRT_MAX){
            result = v;
        }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
