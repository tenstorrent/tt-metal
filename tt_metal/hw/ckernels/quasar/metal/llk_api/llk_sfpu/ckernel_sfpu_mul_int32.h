// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_op.h"
#include "sfpu/ckernel_sfpu_mul_int32.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// MulInt<APPROX, FORMAT, DST_SYNC, DST_ACCUM, SIGN_MAGNITUDE_FORMAT, ITERATIONS>. Same interface as WH/BH;
// Quasar supports Int32 only. SIGN_MAGNITUDE_FORMAT selects the sign-magnitude Int32 dest encoding produced by
// Int8 copy_tile + fp32_dest_acc FPU (native Int32 tiles are 2's complement in dest).
//   calculate(in0, in1, out, vector_mode) -> _mul_int32_
//   init()                                -> bare init (_llk_math_eltwise_sfpu_init_)
// Backs mul_int_tile / mul_int_tile_init and llk_math_eltwise_binary_sfpu_mul_int.
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    DataFormat FORMAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    bool SIGN_MAGNITUDE_FORMAT = false,
    int ITERATIONS = SFPU_ITERATIONS>
struct MulInt : SfpuBinaryOp<
                    MulInt<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, SIGN_MAGNITUDE_FORMAT, ITERATIONS>,
                    DST_SYNC,
                    DST_ACCUM> {
    static_assert(FORMAT == DataFormat::Int32, "Quasar SFPU mul_int currently supports Int32 only");

    static void kernel(std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out) {
        _mul_int32_<APPROXIMATION_MODE, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(dst_index_in0, dst_index_in1, dst_index_out);
    }
};

}  // namespace sfpu
}  // namespace ckernel
