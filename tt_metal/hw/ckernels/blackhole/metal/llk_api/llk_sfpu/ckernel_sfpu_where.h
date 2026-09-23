// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_where.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// Where<APPROX, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(in0, in1, in2, out, vector_mode)
//   backs where_tile<data_format> (out = in0 != 0 ? in1 : in2) and where_tile_init
//   (init_kernel programs ADDR_MOD_6 (dest incr 2) and calls _init_where_, which programs the SFPLOADMACRO
//   templates).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DataFormat FORMAT, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Where : SfpuTernaryOp<Where<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(
        std::uint32_t dst_index_in0,
        std::uint32_t dst_index_in1,
        std::uint32_t dst_index_in2,
        std::uint32_t dst_index_out) {
        _calculate_where_<APPROXIMATION_MODE, FORMAT, ITERATIONS>(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out);
    }

    static void init_kernel() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
        _init_where_<APPROXIMATION_MODE>();
    }
};

}  // namespace sfpu
}  // namespace ckernel
