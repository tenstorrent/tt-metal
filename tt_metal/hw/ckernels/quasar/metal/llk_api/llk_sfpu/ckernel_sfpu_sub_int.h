// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_binary_sfpu.h"

namespace ckernel {
namespace sfpu {

// sub_int (Blackhole / Wormhole tt-llk _sub_int_) is not ported to Quasar. The Compute API keeps
// sub_int_tile on every arch, so its kernel entry point rejects the call here.
template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _sub_int_(
    [[maybe_unused]] const std::uint32_t dst_index_in0,
    [[maybe_unused]] const std::uint32_t dst_index_in1,
    [[maybe_unused]] const std::uint32_t dst_index_out) {
    LLK_ASSERT(false, "sub_int_tile is not supported on Quasar");
}

// Op class for an elementwise integer subtract of two tiles: out = in0 - in1. The kernel lives in tt-llk (_sub_int_).
// Same name and leading template parameters as on Wormhole/Blackhole.
template <
    bool APPROXIMATION_MODE,
    InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32,
    int ITERATIONS = 8,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct SubInt : SfpuBinaryOp<SubInt<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS, SLOT>, SLOT> {
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        _sub_int_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, false /* SIGN_MAGNITUDE_FORMAT */>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
};

}  // namespace sfpu
}  // namespace ckernel
