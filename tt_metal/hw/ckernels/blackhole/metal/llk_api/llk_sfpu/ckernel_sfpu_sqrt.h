// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en>
inline void calculate_sqrt() {
    _calculate_sqrt_<APPROX_MODE, ITERATIONS, fp32_dest_acc_en>(ITERATIONS);
}

template <ckernel::ApproximationMode APPROX_MODE>
void sqrt_init() {
    _init_sqrt_<APPROX_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
