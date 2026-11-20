// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_binary_max_min.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, bool IS_MAX_OP>
inline void llk_math_eltwise_binary_sfpu_max_min(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out,
    const VectorMode vector_mode) {
    SFPU_BINARY_CALL(
        DST_SYNC,
        DST_ACCUM,
        calculate_binary_max_min,
        (IS_MAX_OP),
        dst_index_in0,
        dst_index_in1,
        dst_index_out,
        vector_mode);
}

template <DstSync DST_SYNC, bool DST_ACCUM, bool IS_MAX_OP, bool IS_UNSIGNED>
inline void llk_math_eltwise_binary_sfpu_max_min_int32(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out,
    const VectorMode vector_mode = VectorMode::RC) {
    SFPU_BINARY_CALL(
        DST_SYNC,
        DST_ACCUM,
        calculate_binary_max_min_int32,
        (IS_MAX_OP, IS_UNSIGNED),
        dst_index_in0,
        dst_index_in1,
        dst_index_out,
        vector_mode);
}

}  // namespace ckernel
