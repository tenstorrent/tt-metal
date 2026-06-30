// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void calculate_logical_not() {
    static_assert(
        INSTRUCTION_MODE == InstrModLoadStore::DEFAULT || INSTRUCTION_MODE == InstrModLoadStore::LO16 ||
            INSTRUCTION_MODE == InstrModLoadStore::INT32,
        "INSTRUCTION_MODE must be one of: DEFAULT, LO16, INT32.");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (INSTRUCTION_MODE == InstrModLoadStore::DEFAULT) {
            sfpi::vFloat v = sfpi::dst_reg[0];
            sfpi::vFloat r = 0.0f;
            v_if(v == 0.0f) { r = 1.0f; }
            v_endif;
            sfpi::dst_reg[0] = r;
        } else if constexpr (INSTRUCTION_MODE == InstrModLoadStore::LO16) {
            sfpi::vUInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt r = 0;
            v_if(v == 0) { r = 1; }
            v_endif;
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>() = r;
        } else {  // INT32
            sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vInt r = 0;
            v_if(v == 0) { r = 1; }
            v_endif;
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = r;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
