// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void logical_not_unary_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void calculate_logical_not() {
    static_assert(
        INSTRUCTION_MODE == InstrModLoadStore::DEFAULT || INSTRUCTION_MODE == InstrModLoadStore::UINT16 ||
            INSTRUCTION_MODE == InstrModLoadStore::INT32,
        "INSTRUCTION_MODE must be one of: DEFAULT, UINT16, INT32.");

    // DEFAULT uses the native float layout, UINT16 unsigned 16-bit (U16), INT32 two's-complement 32-bit (I32).
    // mode<DataLayout::Default> is equivalent to a plain dst_reg access.
    constexpr sfpi::DataLayout layout = (INSTRUCTION_MODE == InstrModLoadStore::UINT16)  ? sfpi::DataLayout::U16
                                        : (INSTRUCTION_MODE == InstrModLoadStore::INT32) ? sfpi::DataLayout::I32
                                                                                         : sfpi::DataLayout::Default;
    using vType = std::conditional_t<
        INSTRUCTION_MODE == InstrModLoadStore::UINT16,
        sfpi::vUInt,
        std::conditional_t<INSTRUCTION_MODE == InstrModLoadStore::INT32, sfpi::vInt, sfpi::vFloat>>;

    // Integer layouts: lz(v) >> 5 is exactly 1 for 0 and 0 otherwise (SFPLZ returns 32 only
    // for 0), with no lane predication. Float: the default is stored first and the taken arm
    // stored under the predicate, two SFPSTOREs instead of two SFPMOVs and a store.
    auto row = [&](const int r) __attribute__((always_inline)) {
        vType v = sfpi::dst_reg[r].mode<layout>();
        if constexpr (std::is_same_v<vType, sfpi::vFloat>) {
            // v == 0 tests all 32 bits, so -0.0 (0x80000000) needs its sign cleared to count as zero.
            v = sfpi::setsgn(v, 0 /* sgn */);
            sfpi::dst_reg[r] = 0.0f;
            v_if(v == 0) { sfpi::dst_reg[r] = 1.0f; }
            v_endif;
        } else {
            sfpi::vUInt z = sfpi::lz(v);
            sfpi::dst_reg[r].mode<layout>() = sfpi::as<vType>(z >> 5);
        }
    };

    // Two rows share one TTINCRWC.
    static_assert(ITERATIONS % 2 == 0, "calculate_logical_not processes two rows per iteration");
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        row(0);
        row(1);
        sfpi::dst_reg += 2;
    }
}
}  // namespace ckernel::sfpu
