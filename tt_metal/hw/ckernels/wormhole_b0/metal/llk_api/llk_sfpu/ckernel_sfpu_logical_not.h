// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void calculate_logical_not() {
    static_assert(
        INSTRUCTION_MODE == InstrModLoadStore::DEFAULT || INSTRUCTION_MODE == InstrModLoadStore::LO16 ||
            INSTRUCTION_MODE == InstrModLoadStore::INT32,
        "INSTRUCTION_MODE must be one of: DEFAULT, LO16, INT32.");

    // DEFAULT uses the native float layout, LO16 unsigned 16-bit (U16), INT32 two's-complement 32-bit (I32).
    // mode<DataLayout::Default> is equivalent to a plain dst_reg access.
    constexpr sfpi::DataLayout layout = (INSTRUCTION_MODE == InstrModLoadStore::LO16)    ? sfpi::DataLayout::U16
                                        : (INSTRUCTION_MODE == InstrModLoadStore::INT32) ? sfpi::DataLayout::I32
                                                                                         : sfpi::DataLayout::Default;
    using vType = std::conditional_t<
        INSTRUCTION_MODE == InstrModLoadStore::LO16,
        sfpi::vUInt,
        std::conditional_t<INSTRUCTION_MODE == InstrModLoadStore::INT32, sfpi::vInt, sfpi::vFloat>>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType v = sfpi::dst_reg[0].mode<layout>();
        vType r = 0;
        v_if(v == 0) { r = 1; }
        v_endif;
        sfpi::dst_reg[0].mode<layout>() = r;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// LogicalNot<APPROX, INSTRUCTION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode)
//   backs logical_not_tile<DATA_FORMAT> (INSTRUCTION_MODE is derived from DATA_FORMAT by the API header)
//   and logical_not_tile_init (shared SFPU init only).
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    InstrModLoadStore INSTRUCTION_MODE,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct LogicalNot : SfpuUnaryOp<
                        LogicalNot<APPROXIMATION_MODE, INSTRUCTION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>,
                        DST_SYNC,
                        DST_ACCUM> {
    static void kernel() { calculate_logical_not<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS>(); }
};

}  // namespace ckernel::sfpu
