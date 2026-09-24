// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_fill.h"

namespace ckernel::sfpu {

// Op class that fills every datum of the tile with a float value.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Fill : SfpuUnaryOp<Fill<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(const float value) {
        _calculate_fill_<APPROXIMATION_MODE, ITERATIONS>(value);
    }
};

// Op class that fills every datum of the tile with an integer value, stored with INSTRUCTION_MODE.
template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS = 8>
struct FillInt : SfpuUnaryOp<FillInt<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t value) {
        _calculate_fill_int_<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS>(value);
    }
};

// Op class that fills every datum of the tile with the float whose bit pattern is value_bit_mask.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct FillBitcast : SfpuUnaryOp<FillBitcast<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t value_bit_mask) {
        _calculate_fill_bitcast_<APPROXIMATION_MODE, ITERATIONS>(value_bit_mask);
    }
};

}  // namespace ckernel::sfpu
