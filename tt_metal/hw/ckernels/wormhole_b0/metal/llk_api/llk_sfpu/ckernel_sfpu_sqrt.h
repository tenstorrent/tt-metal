// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "llk_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool FAST_APPROX>
inline void calculate_sqrt() {
    _calculate_sqrt_<APPROX_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX>(ITERATIONS);
}

template <ApproximationMode APPROX_MODE>
void sqrt_init() {
    _init_sqrt_<APPROX_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
