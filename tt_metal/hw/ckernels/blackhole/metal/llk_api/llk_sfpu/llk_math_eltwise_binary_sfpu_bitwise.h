// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_bitwise.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_bitwise_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, sfpu::BinaryBitwiseOp BITWISE_OP, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_bitwise(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for bitwise operation. Supported data formats are: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        DATA_FORMAT == DataFormat::UInt16 ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_sfpu_binary_bitwise<APPROXIMATE, BITWISE_OP, INSTRUCTION_MODE>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
