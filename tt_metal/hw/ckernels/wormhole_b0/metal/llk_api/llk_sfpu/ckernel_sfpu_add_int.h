// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_math_eltwise_sfpu_op.h"
#include "sfpu/ckernel_sfpu_add_int.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// AddInt<APPROX, FORMAT, DST_SYNC, DST_ACCUM, SIGN_MAGNITUDE_FORMAT, ITERATIONS>
//   calculate(in0, in1, out, vector_mode) -> _add_int_ (LO16 loads for UInt16, INT32 otherwise)
//   init()                                -> bare init
// Backs add_int_tile / add_int_tile_init.
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    DataFormat FORMAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    bool SIGN_MAGNITUDE_FORMAT = false,
    int ITERATIONS = 8>
struct AddInt : SfpuBinaryOp<
                    AddInt<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, SIGN_MAGNITUDE_FORMAT, ITERATIONS>,
                    DST_SYNC,
                    DST_ACCUM> {
    static_assert(
        FORMAT == DataFormat::Int32 || FORMAT == DataFormat::UInt32 || FORMAT == DataFormat::UInt16,
        "Unsupported data format for add_int. Supported data formats are: Int32, UInt32, UInt16");
    static constexpr InstrModLoadStore instruction_mode =
        (FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

    static void kernel(std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out) {
        _add_int_<APPROXIMATION_MODE, ITERATIONS, instruction_mode, SIGN_MAGNITUDE_FORMAT>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
};

}  // namespace sfpu
}  // namespace ckernel
