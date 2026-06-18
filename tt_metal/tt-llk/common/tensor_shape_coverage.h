// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Common TensorShape coverage definitions shared by TRISC-specific coverage tables.
//
// Regenerate by running the functional pytests with --logging-level=DEBUG
// and feeding the per-worker test_run_gw*.log files through /tmp/ts-coverage/parse.py.
//

#pragma once

// Match tensor_shape.h's gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

#include <array>
#include <cstddef>

#include "tensor_shape.h"

namespace ckernel::coverage
{

enum class TensorShapeFunctionCoverage
{
    _llk_math_eltwise_binary_standard_,
    _llk_math_eltwise_binary_standard_init_,
    _llk_math_eltwise_binary_with_dest_reuse_,
    _llk_math_eltwise_binary_with_dest_reuse_init_,
    _llk_math_reduce_,
    _llk_math_reduce_init_,
    _llk_unpack_AB_init_,
    _llk_unpack_AB_mop_config_,
    _llk_unpack_AB_reduce_init_,
    _llk_unpack_reduce_init_,
    _llk_unpack_AB_reduce_mop_config_,
    _llk_unpack_A_init_,
    _llk_unpack_A_mop_config_,
    eltwise_binary_configure_mop_standard,
    eltwise_binary_configure_mop_with_dest_reuse,
};

constexpr const char* tensor_shape_function_name(const TensorShapeFunctionCoverage fn)
{
    using Function = TensorShapeFunctionCoverage;
    switch (fn)
    {
        case Function::_llk_math_eltwise_binary_standard_:
            return "_llk_math_eltwise_binary_standard_";
        case Function::_llk_math_eltwise_binary_standard_init_:
            return "_llk_math_eltwise_binary_standard_init_";
        case Function::_llk_math_eltwise_binary_with_dest_reuse_:
            return "_llk_math_eltwise_binary_with_dest_reuse_";
        case Function::_llk_math_eltwise_binary_with_dest_reuse_init_:
            return "_llk_math_eltwise_binary_with_dest_reuse_init_";
        case Function::_llk_math_reduce_:
            return "_llk_math_reduce_";
        case Function::_llk_math_reduce_init_:
            return "_llk_math_reduce_init_";
        case Function::_llk_unpack_AB_init_:
            return "_llk_unpack_AB_init_";
        case Function::_llk_unpack_AB_mop_config_:
            return "_llk_unpack_AB_mop_config_";
        case Function::_llk_unpack_AB_reduce_init_:
            return "_llk_unpack_AB_reduce_init_";
        case Function::_llk_unpack_reduce_init_:
            return "_llk_unpack_reduce_init_";
        case Function::_llk_unpack_AB_reduce_mop_config_:
            return "_llk_unpack_AB_reduce_mop_config_";
        case Function::_llk_unpack_A_init_:
            return "_llk_unpack_A_init_";
        case Function::_llk_unpack_A_mop_config_:
            return "_llk_unpack_A_mop_config_";
        case Function::eltwise_binary_configure_mop_standard:
            return "eltwise_binary_configure_mop_standard";
        case Function::eltwise_binary_configure_mop_with_dest_reuse:
            return "eltwise_binary_configure_mop_with_dest_reuse";
    }
    return "unknown";
}

constexpr bool tensor_shape_eq(const TensorShape& lhs, const TensorShape& rhs)
{
    return lhs.face_r_dim == rhs.face_r_dim && lhs.face_c_dim == rhs.face_c_dim && lhs.num_faces_r_dim == rhs.num_faces_r_dim &&
           lhs.num_faces_c_dim == rhs.num_faces_c_dim;
}

template <std::size_t N>
constexpr bool contains_tensor_shape(const std::array<TensorShape, N>& covered_shapes, const TensorShape& tensor_shape)
{
    for (const TensorShape& covered_shape : covered_shapes)
    {
        if (tensor_shape_eq(covered_shape, tensor_shape))
        {
            return true;
        }
    }
    return false;
}

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
