// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "sfpu/ckernel_sfpu_mul_replay.h"

namespace ckernel {

template <DataFormat data_format>
ALWI void llk_init_replay_binary_sfpu_mul_integer() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int replay. Supported: Int32, UInt32, UInt16");
    if constexpr (data_format == DataFormat::UInt16) {
        _init_replay_binary_sfpu_mul_uint16_();
    } else {
        _init_replay_binary_sfpu_mul_int32_();
    }
}

template <DataFormat data_format>
ALWI void llk_replay_binary_sfpu_mul_integer(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int replay. Supported: Int32, UInt32, UInt16");
    (void)idst1;
    (void)odst;
    if constexpr (data_format == DataFormat::UInt16) {
        _llk_replay_binary_sfpu_eltwise_<SFPU_BINARY_MUL_UINT16_REPLAY_LEN>(idst0);
    } else {
        _llk_replay_binary_sfpu_eltwise_<SFPU_BINARY_MUL_INT32_REPLAY_LEN>(idst0);
    }
}

ALWI void llk_replay_binary_sfpu_mul_float(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    _llk_replay_binary_sfpu_eltwise_<SFPU_BINARY_MUL_REPLAY_LEN>(idst0);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
ALWI void llk_init_replay_binary_sfpu_mul_float() {
    _init_replay_binary_sfpu_mul_float_<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

}  // namespace ckernel
