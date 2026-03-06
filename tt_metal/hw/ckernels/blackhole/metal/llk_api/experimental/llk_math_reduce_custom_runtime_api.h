// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "experimental/llk_math_reduce_runtime_custom.h"

template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_init_runtime(uint32_t block_ct_dim) {
    _llk_math_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(block_ct_dim);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_runtime(const uint32_t dst_index) {
    _llk_math_reduce_block_max_row_runtime_<is_fp32_dest_acc_en>(dst_index);
}
