// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpu/ckernel_sfpu_converter.h"

#include "sfpi.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
void rsub_init() {
    ;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rsub(uint value) {
    vFloat arg2 = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat value = dst_reg[0];
        dst_reg[0] = arg2 - value;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
