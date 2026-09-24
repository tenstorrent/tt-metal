// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_add_int.h"

namespace ckernel::sfpu {

// Op class for an elementwise integer add of two tiles. The kernel lives in tt-llk (_add_int_).
template <bool APPROXIMATION_MODE, DataFormat data_format = DataFormat::Int32, int ITERATIONS = 8>
struct AddInt : SfpuBinaryOp<AddInt<APPROXIMATION_MODE, data_format, ITERATIONS>> {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for add_int. Supported data formats are: Int32, UInt32, UInt16");
    static constexpr InstrModLoadStore INSTRUCTION_MODE =
        (data_format == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

    static constexpr auto& calculate =
        _add_int_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, false /* SIGN_MAGNITUDE_FORMAT */>;
};

}  // namespace ckernel::sfpu
