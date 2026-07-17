// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

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

    constexpr sfpi::DataLayout layout = (INSTRUCTION_MODE == InstrModLoadStore::LO16)    ? sfpi::DataLayout::U16
                                        : (INSTRUCTION_MODE == InstrModLoadStore::INT32) ? sfpi::DataLayout::I32
                                                                                         : sfpi::DataLayout::Default;
    using vType = std::conditional_t<
        INSTRUCTION_MODE == InstrModLoadStore::LO16,
        sfpi::vUInt,
        std::conditional_t<INSTRUCTION_MODE == InstrModLoadStore::INT32, sfpi::vInt, sfpi::vFloat>>;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vType v = sfpi::dst_reg[0].mode<layout>();
        vType r = 0;
        v_if(v == 0) { r = 1; }
        v_endif;
        sfpi::dst_reg[0].mode<layout>() = r;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
