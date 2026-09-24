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
    static constexpr auto& calculate =
        _sub_int_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, false /* SIGN_MAGNITUDE_FORMAT */>;
};

}  // namespace ckernel::sfpu
