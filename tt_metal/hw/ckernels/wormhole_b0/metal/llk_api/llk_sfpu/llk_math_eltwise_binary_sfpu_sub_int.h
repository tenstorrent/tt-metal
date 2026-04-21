// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel {

inline void llk_math_eltwise_binary_sfpu_sub_int_init() { SFPU_BINARY_INIT(unused); }

template <bool APPROXIMATE, int ITERATIONS = 8, DataFormat DATA_FORMAT, bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_sub_int(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for sub_int. Supported data formats are: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _sub_int_,
        (APPROXIMATE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT),
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
