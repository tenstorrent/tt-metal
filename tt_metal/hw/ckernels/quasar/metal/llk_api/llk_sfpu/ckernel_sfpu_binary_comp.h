// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "sfpu/ckernel_sfpu_binary_comp.h"

namespace ckernel {
namespace sfpu {

// Op class for elementwise compare of two integer tiles in Dest: out = (in0 OP in1) ? 1 : 0. Same name
// and leading template parameters as on Wormhole/Blackhole; Quasar supports Int32 lt/gt/le/ge only, with
// sign-magnitude operands and result.
template <
    bool APPROXIMATION_MODE,
    CompareOp RELATIONAL_OP,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct BinaryCompInt
    : SfpuBinaryOp<BinaryCompInt<APPROXIMATION_MODE, RELATIONAL_OP, DATA_FORMAT, ITERATIONS, SLOT>, SLOT> {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU integer compare currently supports Int32 only");
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        calculate_binary_comp_int32<
            APPROXIMATION_MODE,
            ITERATIONS,
            RELATIONAL_OP,
            true /*SIGN_MAGNITUDE_FORMAT*/,
            SLOT>(dst_index_in0, dst_index_in1, dst_index_out);
    }
};

}  // namespace sfpu
}  // namespace ckernel
