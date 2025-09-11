// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool legacy_compat>
inline void calculate_sqrt() {
    _calculate_sqrt_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, legacy_compat>(ITERATIONS);
}

template <bool APPROXIMATION_MODE, bool legacy_compat>
void sqrt_init() {
    _init_sqrt_<APPROXIMATION_MODE, legacy_compat>();
}

}  // namespace sfpu
}  // namespace ckernel
