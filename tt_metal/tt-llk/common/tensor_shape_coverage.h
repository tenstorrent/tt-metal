// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// AUTO-GENERATED: TensorShape coverage observed across LLK pytests.
// Sourced from LLK_DPRINT_TENSOR_SHAPE lines emitted before validation asserts.
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

// Match tensor_shape.h's gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

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

inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_AB_reduce_init = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_AB_reduce_mop_config = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x1,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_A_init = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
    TENSOR_SHAPE_FR16_NF2x2,
}};

inline constexpr std::array<TensorShape, 7> covered_shapes_llk_unpack_A_mop_config = {{
    TENSOR_SHAPE_FR1_NF1x2,
    TENSOR_SHAPE_FR2_NF1x2,
    TENSOR_SHAPE_FR4_NF1x2,
    TENSOR_SHAPE_FR8_NF1x2,
    TENSOR_SHAPE_FR16_NF1x1,
    TENSOR_SHAPE_FR16_NF1x2,
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

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
