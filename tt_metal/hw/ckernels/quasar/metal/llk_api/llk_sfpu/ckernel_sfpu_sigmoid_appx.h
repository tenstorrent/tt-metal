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

// Quasar has no LReg-operand form of sfpi::lut: SFPLUT reads its three FP8 slope/intercept pairs from
// the LUT config registers (CREG 9-11), so the table is loaded by sigmoid_appx_init and the loop uses
// the table-less lut<Fp8x3>. The coefficients are the Blackhole ones. The table overwrites the SFPU
// constants 0.0/1.0/-1.0 in LReg9-11, so no sfpi code that needs them may run between the init and
// the loop; the next op's init restores them.
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
    // Load the 3 fp8 LUT coefficient pairs into the LUT config registers. Written through
    // _sfpu_load_config32_ because the assembler rejects an sfpi SFPCONFIG to CREG 9 and 10
    // (tt-metal #51346).
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 0, 0, sfpi::sLut8si(0.22656f, 0.0f).get());
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 1, 0, sfpi::sLut8si(0.26562f, -0.04687f).get());
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 2, 0, sfpi::sLut8si(0.0f, 0.5f).get());
}

}  // namespace sfpu
}  // namespace ckernel
