// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include <limits>

#include "sfpi.h"
#include "llk_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_identity() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_identity_uint() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
