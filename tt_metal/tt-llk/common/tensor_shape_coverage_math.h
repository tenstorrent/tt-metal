// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Match tensor_shape.h's gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

#include <array>

#include "tensor_shape_coverage.h"
#include "tensor_shape_coverage_matmul.h"

namespace ckernel::coverage
{

inline constexpr std::array<TensorShape, 8> covered_shapes_llk_math_eltwise_binary_standard = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 8> covered_shapes_llk_math_eltwise_binary_standard_init = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 2> covered_shapes_llk_math_eltwise_binary_with_dest_reuse = {{
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 2> covered_shapes_llk_math_eltwise_binary_with_dest_reuse_init = {{
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 7> covered_shapes_llk_math_reduce = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 8> covered_shapes_eltwise_binary_configure_mop_standard = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 2> covered_shapes_eltwise_binary_configure_mop_with_dest_reuse = {{
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

constexpr bool is_math_tensor_shape_covered(const TensorShapeFunctionCoverage fn, const TensorShape& tensor_shape)
{
    using Function = TensorShapeFunctionCoverage;
    switch (fn)
    {
        case Function::_llk_math_eltwise_binary_standard_:
            return contains_tensor_shape(covered_shapes_llk_math_eltwise_binary_standard, tensor_shape);
        case Function::_llk_math_eltwise_binary_standard_init_:
            return contains_tensor_shape(covered_shapes_llk_math_eltwise_binary_standard_init, tensor_shape);
        case Function::_llk_math_eltwise_binary_with_dest_reuse_:
            return contains_tensor_shape(covered_shapes_llk_math_eltwise_binary_with_dest_reuse, tensor_shape);
        case Function::_llk_math_eltwise_binary_with_dest_reuse_init_:
            return contains_tensor_shape(covered_shapes_llk_math_eltwise_binary_with_dest_reuse_init, tensor_shape);
        case Function::_llk_math_reduce_:
        case Function::_llk_math_reduce_init_:
            return contains_tensor_shape(covered_shapes_llk_math_reduce, tensor_shape);
        case Function::eltwise_binary_configure_mop_standard:
            return contains_tensor_shape(covered_shapes_eltwise_binary_configure_mop_standard, tensor_shape);
        case Function::eltwise_binary_configure_mop_with_dest_reuse:
            return contains_tensor_shape(covered_shapes_eltwise_binary_configure_mop_with_dest_reuse, tensor_shape);
        default:
            return false;
    }
}

constexpr bool is_math_tensor_shape_pair_covered(const TensorShapeFunctionCoverage fn, const TensorShape& in0, const TensorShape& in1)
{
    using Function = TensorShapeFunctionCoverage;
    switch (fn)
    {
        case Function::matmul_configure_addrmod:
            return contains_tensor_shape_pair(covered_shape_pairs_matmul_configure_addrmod, in0, in1);
        case Function::matmul_configure_mop:
            return contains_tensor_shape_pair(covered_shape_pairs_matmul_configure_mop, in0, in1);
        case Function::matmul_configure_mop_throttled:
            return contains_tensor_shape_pair(covered_shape_pairs_matmul_configure_mop_throttled, in0, in1);
        case Function::_llk_math_matmul_init_:
            return contains_tensor_shape_pair(covered_shape_pairs_llk_math_matmul_init, in0, in1);
        default:
            return false;
    }
}

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
