// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_shape_coverage.h"

// TensorShape coverage tables for unpack LLKs. Each table lists the TensorShape
// values observed for the corresponding unpack init or MOP configuration
// function, and the validation helpers use these lists to flag unobserved
// shapes during assert/debug-print coverage runs.
//
// Match tensor_shape_coverage.h's gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

#include <array>

namespace ckernel::coverage
{

inline std::array<TensorShape, 9> covered_shapes_llk_unpack_AB = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x1, // TODO(#47307): ndivnic to add a test in tt-llk to cover this case
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline std::array<TensorShape, 8> covered_shapes_llk_unpack_AB_reduce = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline std::array<TensorShape, 9> covered_shapes_llk_unpack_A = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x1,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

constexpr bool is_unpack_tensor_shape_covered(const TensorShapeFunctionCoverage fn, const TensorShape& tensor_shape)
{
    using Function = TensorShapeFunctionCoverage;
    switch (fn)
    {
        case Function::_llk_unpack_AB_init_:
        case Function::_llk_unpack_AB_mop_config_:
            return contains_tensor_shape(covered_shapes_llk_unpack_AB, tensor_shape);
        case Function::_llk_unpack_AB_reduce_init_:
        case Function::_llk_unpack_reduce_init_:
        case Function::_llk_unpack_AB_reduce_mop_config_:
            return contains_tensor_shape(covered_shapes_llk_unpack_AB_reduce, tensor_shape);
        case Function::_llk_unpack_A_init_:
        case Function::_llk_unpack_A_mop_config_:
            return contains_tensor_shape(covered_shapes_llk_unpack_A, tensor_shape);
        default:
            return false;
    }
}

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
