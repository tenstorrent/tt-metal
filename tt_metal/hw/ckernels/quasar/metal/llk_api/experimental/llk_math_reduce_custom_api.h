// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "experimental/llk_math_reduce_runtime_custom.h"

/*************************************************************************
 * LLK REDUCE CUSTOM - Specialized reduce_max_row operations (Quasar)
 *
 * These are the compile-time-block_ct_dim entry points the arch-agnostic Compute API (reduce_custom.h)
 * and the SDPA kernel call via the `<block_ct_dim>` template family.
 *
 * WHY THEY FORWARD TO THE RUNTIME LIB: Quasar has NO separate compile-time LLK lib. Blackhole builds
 * one on top of `lltt` (a constexpr MOP builder) that bakes block_ct_dim into the MOP at compile time;
 * Quasar has no `lltt`, and its only MOP mechanism (`ckernel_template`) takes runtime loop bounds. So
 * block_ct_dim reaches the MOP as a runtime value no matter what. These wrappers therefore just pass the
 * template block_ct_dim as a runtime argument to the single runtime lib -- the template exists purely
 * for Compute-API call-shape parity and buys no constant-folding on Quasar.
 *************************************************************************/

/**
 * @brief Initializes the block reduce_max_row math thread (compile-time block_ct_dim).
 *
 * @tparam block_ct_dim  Number of tiles in the width dimension processed as one block.
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode (not yet supported on Quasar).
 * @param tensor_shape  Operand tile shape (face count / dims) baked into the pool MOP.
 * @note Specialized for SDPA/softmax block row-max; not a substitute for llk_math_reduce_init.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_init(const ckernel::TensorShape& tensor_shape) {
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    _llk_math_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * @brief Programs the block reduce_max_row pool MOP (compile-time block_ct_dim, forwarded as a runtime arg).
 *
 * @tparam block_ct_dim  Number of tiles in the width dimension processed as one block.
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode (not yet supported on Quasar).
 * @param tensor_shape  Operand tile shape driving the pool MOP.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_mop_config(const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_mop_config_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * @brief Executes the block reduce_max_row: accumulate the row-max across the block, then transpose
 *        each pooled row partial into a reduced column at DEST[dst_index].
 *
 * @param dst_index    DEST tile index that receives the reduced column.
 * @param tensor_shape Operand tile shape (drives the transpose row count).
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_block_max_row_runtime_<is_fp32_dest_acc_en>(dst_index, tensor_shape);
}

/**
 * @brief Uninit for the block reduce_max_row math thread. No-op; kept for init/execute/uninit symmetry.
 *
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode (not yet supported on Quasar).
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_uninit() {
    _llk_math_reduce_block_max_row_uninit_runtime_<is_fp32_dest_acc_en>();
}
