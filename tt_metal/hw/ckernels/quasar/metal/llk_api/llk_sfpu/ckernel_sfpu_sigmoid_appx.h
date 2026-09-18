// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        sfpi::dst_reg[0] = sfpi::lut<sfpi::LutMode::Fp8x3>(val) + 0.5f;

        sfpi::dst_reg++;
    }

}

inline void sigmoid_appx_init() {
    // Quasar holds the same FP8 coefficients in LUT configuration registers.
    // Load via LREG0: immediate SFPCONFIG restores constants instead of these LUT bits.
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 0, 0, sfpi::sLut8si(0.22656f, 0.0f).get());
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 1, 0, sfpi::sLut8si(0.26562f, -0.04687f).get());
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 2, 0, sfpi::sLut8si(0.0f, 0.5f).get());
}

}  // namespace sfpu
}  // namespace ckernel
