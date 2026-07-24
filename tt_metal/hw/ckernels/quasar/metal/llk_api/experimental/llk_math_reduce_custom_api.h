// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "experimental/llk_math_reduce_custom.h"

/*************************************************************************
 * LLK REDUCE CUSTOM - Specialized reduce_max_row operations (Quasar)
 *************************************************************************/

/**
 * @brief Initializes the block reduce_max_row math thread (compile-time block_ct_dim).
 *
 * @tparam block_ct_dim  Number of tiles in the width dimension processed as one block.
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode.
 * @note Specialized for SDPA/softmax block row-max; not a substitute for llk_math_reduce_init.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_init(const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_init_<block_ct_dim, is_fp32_dest_acc_en>(tensor_shape);
}

template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_mop_config(const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_mop_config_<block_ct_dim, is_fp32_dest_acc_en>(tensor_shape);
}

/**
 * @brief Executes the block reduce_max_row (compile-time block_ct_dim): accumulate the row-max across
 *        the block, then transpose once into a reduced column at DEST[dst_index].
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_<block_ct_dim, is_fp32_dest_acc_en>(dst_index, tensor_shape);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_uninit() {
    _llk_math_reduce_block_max_row_uninit_<is_fp32_dest_acc_en>();
}
