// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void cast_fp32_to_fp16a() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloart x = sfpi::dst_reg[0];
        sfpi::dst_reg[0].mode<sfpi::SFPSTORE_MOD0_FMT_FP16A>() =
            sfpi::convert<sfpi::vFloat16a>(x, sfpi::RoundMode::NearestEven);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
