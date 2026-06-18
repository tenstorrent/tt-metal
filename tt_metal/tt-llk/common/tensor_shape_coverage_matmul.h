// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// AUTO-GENERATED: per-function TensorShape pair coverage for matmul helpers.
//
// Regenerate by running matmul pytests with --logging-level=DEBUG,
// TT_LLK_DISABLE_ASSERTS=1, then: parse.py harvest && parse.py emit-math
//
// Last update : 2026-06-18T14:28:53+00:00
// Tests run   : 2

#pragma once

#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

#include <array>

#include "tensor_shape_coverage.h"

namespace ckernel::coverage
{

// _llk_math_matmul_init_: 6 unique TensorShape pair(s)
inline constexpr std::array<TensorShapePair, 6> covered_shape_pairs_llk_math_matmul_init = {{
    {TENSOR_SHAPE_FR1_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR2_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR4_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR8_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR16_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR16_NF2x2, TENSOR_SHAPE_FR16_NF2x2},
}};

// matmul_configure_addrmod: 6 unique TensorShape pair(s)
inline constexpr std::array<TensorShapePair, 6> covered_shape_pairs_matmul_configure_addrmod = {{
    {TENSOR_SHAPE_FR1_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR2_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR4_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR8_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR16_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR16_NF2x2, TENSOR_SHAPE_FR16_NF2x2},
}};

// matmul_configure_mop: 6 unique TensorShape pair(s)
inline constexpr std::array<TensorShapePair, 6> covered_shape_pairs_matmul_configure_mop = {{
    {TENSOR_SHAPE_FR1_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR2_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR4_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR8_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR16_NF1x2, TENSOR_SHAPE_FR16_NF2x2},
    {TENSOR_SHAPE_FR16_NF2x2, TENSOR_SHAPE_FR16_NF2x2},
}};

// matmul_configure_mop_throttled: 1 unique TensorShape pair(s)
inline constexpr std::array<TensorShapePair, 1> covered_shape_pairs_matmul_configure_mop_throttled = {{
    {TENSOR_SHAPE_FR16_NF2x2, TENSOR_SHAPE_FR16_NF2x2},
}};

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
