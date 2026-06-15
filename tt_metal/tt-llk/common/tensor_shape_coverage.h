// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// AUTO-GENERATED: per-function TensorShape coverage observed across LLK pytests.
// Sourced from DPRINT lines emitted by LLK_DPRINT_TENSOR_SHAPE in front of
// LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(...)) call sites.
//
// Regenerate by running the BH functional pytests with --logging-level=DEBUG
// and feeding the per-worker test_run_gw*.log files through /tmp/ts-coverage/parse.py.
//
// Last update : 2026-06-15T11:25:24+00:00
// Tests run   : 26
// Architecture: blackhole

#pragma once

#include <array>

#include "tensor_shape.h"

// The TENSOR_SHAPE_FR*_NF*x* constants this manifest references are themselves
// gated to ENABLE_LLK_ASSERT or DEBUG_PRINT_ENABLED builds (see tensor_shape.h).
// Mirror the same gate here so this header stays a no-op when included from a
// production kernel build that defines neither flag.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

namespace ckernel::coverage
{

// _llk_math_eltwise_binary_standard_: 8 unique TensorShape(s)
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

// _llk_math_eltwise_binary_standard_init_: 8 unique TensorShape(s)
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

// _llk_math_eltwise_binary_with_dest_reuse_: 2 unique TensorShape(s)
inline constexpr std::array<TensorShape, 2> covered_shapes_llk_math_eltwise_binary_with_dest_reuse = {{
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_math_eltwise_binary_with_dest_reuse_init_: 2 unique TensorShape(s)
inline constexpr std::array<TensorShape, 2> covered_shapes_llk_math_eltwise_binary_with_dest_reuse_init = {{
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_math_reduce_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_math_reduce = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_unpack_AB_init_: 8 unique TensorShape(s)
inline constexpr std::array<TensorShape, 8> covered_shapes_llk_unpack_AB_init = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_unpack_AB_mop_config_: 8 unique TensorShape(s)
inline constexpr std::array<TensorShape, 8> covered_shapes_llk_unpack_AB_mop_config = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_unpack_AB_reduce_init_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_AB_reduce_init = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_unpack_AB_reduce_mop_config_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_AB_reduce_mop_config = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_unpack_A_init_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_A_init = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// _llk_unpack_A_mop_config_: 7 unique TensorShape(s)
inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_A_mop_config = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

// eltwise_binary_configure_mop_standard: 8 unique TensorShape(s)
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

// eltwise_binary_configure_mop_with_dest_reuse: 2 unique TensorShape(s)
inline constexpr std::array<TensorShape, 2> covered_shapes_eltwise_binary_configure_mop_with_dest_reuse = {{
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
