// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu/ckernel_sfpu_softplus.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_softplus(std::uint32_t beta, std::uint32_t beta_reciprocal, std::uint32_t threshold) {
    _calculate_softplus_<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(beta, beta_reciprocal, threshold);
}

}  // namespace sfpu
}  // namespace ckernel
