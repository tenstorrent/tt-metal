// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_shape_coverage.h"

// TensorShape coverage tables for math LLKs. Each table lists the TensorShape
// values observed for the corresponding math init, execute, or MOP
// configuration function, and the validation helpers use these lists to flag
// unobserved shapes during assert/debug-print coverage runs.
//
// Match tensor_shape_coverage.h's gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

namespace ckernel::coverage
{

constexpr bool is_math_eltwise_binary_standard_shape(const TensorShape& tensor_shape)
{
    return tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR1_NF1x2) || tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR2_NF1x2) ||
           tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR4_NF1x2) || tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR8_NF1x2) ||
           tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR16_NF1x1) || tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR16_NF1x2) ||
           tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR16_NF2x1) || tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR16_NF2x2);
}

constexpr bool is_math_eltwise_binary_with_dest_reuse_shape(const TensorShape& tensor_shape)
{
    return tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR16_NF1x2) || tensor_shape_eq(tensor_shape, TENSOR_SHAPE_FR16_NF2x2);
}

inline bool is_math_tensor_shape_covered(const TensorShapeFunctionCoverage fn, const TensorShape& tensor_shape)
{
    using Function = TensorShapeFunctionCoverage;
    switch (fn)
    {
        case Function::_llk_math_eltwise_binary_standard_:
        case Function::_llk_math_eltwise_binary_standard_init_:
        case Function::eltwise_binary_configure_mop_standard:
            return is_math_eltwise_binary_standard_shape(tensor_shape);
        case Function::_llk_math_eltwise_binary_with_dest_reuse_:
        case Function::_llk_math_eltwise_binary_with_dest_reuse_init_:
        case Function::eltwise_binary_configure_mop_with_dest_reuse:
            return is_math_eltwise_binary_with_dest_reuse_shape(tensor_shape);
        case Function::_llk_math_reduce_:
        case Function::_llk_math_reduce_init_:
            return is_math_eltwise_binary_standard_shape(tensor_shape);
        default:
            return false;
    }
}

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
