// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sdpa_broadcast_col.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat format, EltwiseBinaryType binary_type>
inline void llk_math_sfpu_sdpa_broadcast_col_init() {
    sfpu::init_sdpa_broadcast_col<format, binary_type>();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat format, uint32_t block_width>
inline void llk_math_sfpu_sdpa_broadcast_mul_col(uint dst_index) {
    sfpu::calculate_sdpa_broadcast_mul_col<format, block_width>(dst_index);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat format, uint32_t block_width>
inline void llk_math_sfpu_sdpa_broadcast_sub_col(uint dst_index) {
    sfpu::calculate_sdpa_broadcast_sub_col<format, block_width>(dst_index);
}

}  // namespace ckernel
