// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// AUTO-GENERATED: pack TensorShape coverage observed across LLK pytests.
// Sourced from DPRINT lines emitted by LLK_VALIDATE_TENSOR_SHAPE_PACK before
// assert_tensor_shape_unobserved_() on uncovered shapes.
//
// Regenerate: run WH pack pytests with TT_LLK_DISABLE_ASSERTS=1 and
// --logging-level=debug, then parse.py harvest <name> and parse.py emit-pack.
//
// Last update : 2026-06-18T14:47:51+00:00
// Tests run   : 4

#pragma once

// Match tensor_shape.h's gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

#include <array>

#include "tensor_shape_coverage.h"

namespace ckernel::coverage
{

// _llk_pack_hw_configure_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_pack_hw_configure = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_pack_init_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_pack_init = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_pack_mop_config_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_pack_mop_config = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_pack_reconfig_data_format_: 1 unique TensorShape(s)
inline constexpr std::array<TensorShape, 1> covered_shapes_llk_pack_reconfig_data_format = {{
    TENSOR_SHAPE_FR16_NF2x2,
}};

constexpr bool is_pack_tensor_shape_covered(const TensorShapeFunctionCoverage fn, const TensorShape& tensor_shape)
{
    using Function = TensorShapeFunctionCoverage;
    switch (fn)
    {
        case Function::_llk_pack_hw_configure_:
            return contains_tensor_shape(covered_shapes_llk_pack_hw_configure, tensor_shape);
        case Function::_llk_pack_init_:
            return contains_tensor_shape(covered_shapes_llk_pack_init, tensor_shape);
        case Function::_llk_pack_mop_config_:
            return contains_tensor_shape(covered_shapes_llk_pack_mop_config, tensor_shape);
        case Function::_llk_pack_reconfig_data_format_:
            return contains_tensor_shape(covered_shapes_llk_pack_reconfig_data_format, tensor_shape);
        default:
            return false;
    }
}

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
