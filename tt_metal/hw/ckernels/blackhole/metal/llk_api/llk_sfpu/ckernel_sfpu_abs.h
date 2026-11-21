// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void abs_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // INT32_MIN / INT32_MAX as SFPI constants, hoisted out of the SFPU loop. The literals
    // are cast through int32_t so they stay signed and do not narrow when handed to the
    // vInt constructor.
    const sfpi::vInt int32_min = static_cast<int32_t>(0x80000000u);
    const sfpi::vInt int32_max = static_cast<int32_t>(0x7FFFFFFF);

    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // sfpi::abs(vInt) lowers to the dedicated SFPABS integer instruction (mod 0),
        // matching the raw TTI sequence (SFPLOAD + SFPABS + SFPSTORE = 3 SFPU ops).
        // On Blackhole INT32_2S_COMP load/store is a no-op vs INT32, so I32 access is
        // byte-for-byte equivalent to the previous mode-12 access. abs() yields a vMag,
        // which stores through the M32 layout.
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        // SFPABS on Blackhole overflows for INT32_MIN (0x80000000); clamp to INT32_MAX
        // before calling abs so the result does not wrap to a negative value.
        v_if(sfpi::as<sfpi::vInt>(v) == sfpi::as<sfpi::vInt>(int32_min)) { v = int32_max; }
        v_endif;
        sfpi::dst_reg[0].mode<sfpi::DataLayout::M32>() = sfpi::abs(v);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
