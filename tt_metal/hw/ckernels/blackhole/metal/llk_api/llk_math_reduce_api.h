// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_reduce.h"

/*************************************************************************
 * LLK REDUCE
 *************************************************************************/

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool is_int_fpu_en = false>
inline void llk_math_reduce(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");
    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, math_fidelity, is_int_fpu_en>(dst_index, tensor_shape);
}

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool is_int_fpu_en = false>
inline void llk_math_reduce(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t dst_index) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, math_fidelity, is_int_fpu_en>(dst_index, tensor_shape);
}

// Unified init core (explicit shape), shared by the CB-id API and the LLKOperand API (experimental/2_0/).
// Reduce math is FORMAT-FREE: it consumes only operand A's tile geometry. The reduce math EXECUTE already has
// an explicit-shape overload (llk_math_reduce(dst_index, tensor_shape)) reused directly by the id-free path.
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void llk_math_reduce_init_impl(const ckernel::TensorShape& tensor_shape) {
    _llk_math_reduce_init_<type, dim, is_fp32_dest_acc_en, math_fidelity>(tensor_shape);
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void llk_math_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    llk_math_reduce_init_impl<type, dim, is_fp32_dest_acc_en, math_fidelity>(tensor_shape);
}

inline void llk_math_reduce_uninit() { _llk_math_reduce_uninit_(); }
