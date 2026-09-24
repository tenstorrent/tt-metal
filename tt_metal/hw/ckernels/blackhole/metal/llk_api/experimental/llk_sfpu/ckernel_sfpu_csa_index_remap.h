// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_csa_index_remap.h"

namespace ckernel::sfpu {

// Op class that remaps packed CSA [device | bank | row] indices to [bank | row-in-bank], offset by ROW_OFFSET.
// The kernel walks the whole tile itself; ROW_OFFSET is used only by run().
template <std::uint32_t ROW_OFFSET = 0, int ITERATIONS = 32>
struct CsaIndexRemap : SfpuUnaryOp<CsaIndexRemap<ROW_OFFSET, ITERATIONS>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() { _csa_index_remap_<ITERATIONS, ROW_OFFSET>(); }
};

}  // namespace ckernel::sfpu
