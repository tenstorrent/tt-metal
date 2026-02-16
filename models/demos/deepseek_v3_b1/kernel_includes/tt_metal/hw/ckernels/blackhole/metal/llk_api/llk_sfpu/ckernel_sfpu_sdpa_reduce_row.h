// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "../../../../../../../tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sdpa_reduce_row.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <DataFormat format>
inline void init_sdpa_reduce_row() {
    _init_sdpa_reduce_row_8x32_<format>();
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false>
inline void calculate_sdpa_reduce_max_row(uint src_index, uint dst_index, bool prev_max = false) {
    _calculate_sdpa_reduce_max_row_8x32_<format, block_width, skip_signalling>(src_index, dst_index, prev_max);
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false>
inline void calculate_sdpa_reduce_sum_row(uint src_index, uint dst_index, bool prev_sum = false) {
    _calculate_sdpa_reduce_sum_row_8x32_<format, block_width, skip_signalling>(src_index, dst_index, prev_sum);
}
}  // namespace sfpu
}  // namespace ckernel
