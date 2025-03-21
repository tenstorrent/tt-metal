// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_sfpu_types.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE, int ITERATIONS = 8>
void load_immediate_value(float imm) {
    vFloat const_v = s2vFloat16b(imm);
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = const_v;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
