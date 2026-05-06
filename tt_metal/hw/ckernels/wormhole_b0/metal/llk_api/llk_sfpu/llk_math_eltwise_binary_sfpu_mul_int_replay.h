// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "sfpu/ckernel_sfpu_mul_int_replay.h"

namespace ckernel {

template <DataFormat data_format>
ALWI void llk_math_eltwise_binary_sfpu_init_replay() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int replay. Supported: Int32, UInt32, UInt16");
    if constexpr (data_format == DataFormat::UInt16) {
        _llk_math_eltwise_binary_sfpu_init_uint16_replay_();
    } else {
        _llk_math_eltwise_binary_sfpu_init_int32_replay_();
    }
}

template <DataFormat data_format>
ALWI void llk_math_eltwise_binary_sfpu_run_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int replay. Supported: Int32, UInt32, UInt16");
    (void)idst1;
    (void)odst;
    if constexpr (data_format == DataFormat::UInt16) {
        _llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_MUL_UINT16_REPLAY_LEN>(idst0);
    } else {
        _llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_MUL_INT32_REPLAY_LEN>(idst0);
    }
}

}  // namespace ckernel
