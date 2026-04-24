// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_sqrt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool FAST_APPROX>
inline void calculate_sqrt(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_sqrt_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX>(
        dst_index_in, dst_index_out, ITERATIONS);
}

template <bool APPROXIMATION_MODE>
void sqrt_init() {
    _init_sqrt_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
