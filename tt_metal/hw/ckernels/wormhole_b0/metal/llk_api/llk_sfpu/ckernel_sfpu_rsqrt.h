// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool legacy_compat>
inline void calculate_rsqrt() {
    _calculate_rsqrt_<APPROX_MODE, ITERATIONS, fp32_dest_acc_en, legacy_compat>(ITERATIONS);
}

template <ckernel::ApproximationMode APPROX_MODE, bool legacy_compat>
void rsqrt_init() {
    _init_rsqrt_<APPROX_MODE, legacy_compat>();
}

}  // namespace sfpu
}  // namespace ckernel
