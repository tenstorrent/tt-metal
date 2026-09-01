// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "experimental/llk_math_reduce_runtime_custom.h"

/*************************************************************************
 * LLK REDUCE CUSTOM (runtime) - Specialized reduce_max_row operations (Quasar)
 *
 * Runtime-block_ct_dim variants: block_ct_dim is a runtime argument (matches the WH/BH _runtime_api
 * naming so the arch-agnostic Compute API resolves the same call across arches).
 *************************************************************************/

/**
 * @brief Initializes the block reduce_max_row math thread (runtime block_ct_dim).
 *
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode (not yet supported on Quasar).
 * @param block_ct_dim  Number of tiles in the width dimension processed as one block.
 * @param tensor_shape  Operand tile shape baked into the pool MOP.
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_init_runtime(
    const std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * @brief Programs the block reduce_max_row pool MOP (runtime block_ct_dim). Forwards to the lib.
 *
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode (not yet supported on Quasar).
 * @param block_ct_dim  Number of tiles in the width dimension processed as one block.
 * @param tensor_shape  Operand tile shape driving the pool MOP.
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_mop_config_runtime(
    const std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_mop_config_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * @brief Executes the block reduce_max_row (runtime): the block tile count is baked into the MOP by
 *        init, so only dst_index / tensor_shape are needed here.
 *
 * @param dst_index    DEST tile index that receives the reduced column.
 * @param tensor_shape Operand tile shape (drives the transpose row count).
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_runtime(
    const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_runtime_<is_fp32_dest_acc_en>(dst_index, tensor_shape);
}

/**
 * @brief Uninit for the block reduce_max_row math thread (runtime). No-op; kept for init/execute/uninit symmetry.
 *
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode (not yet supported on Quasar).
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_uninit_runtime() {
    _llk_math_reduce_block_max_row_uninit_runtime_<is_fp32_dest_acc_en>();
}
