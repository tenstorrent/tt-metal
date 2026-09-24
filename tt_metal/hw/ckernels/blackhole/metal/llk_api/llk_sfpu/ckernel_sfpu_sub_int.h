// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_sub_int.h"

namespace ckernel::sfpu {

// Op class for an elementwise integer subtract of two tiles: out = in0 - in1. The kernel lives in tt-llk (_sub_int_).
template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32, int ITERATIONS = 8>
struct SubInt : SfpuBinaryOp<SubInt<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        _sub_int_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, false /* SIGN_MAGNITUDE_FORMAT */>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
};

}  // namespace ckernel::sfpu
