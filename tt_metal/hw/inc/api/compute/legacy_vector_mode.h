// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "sfpu/shim.h"
#endif

namespace ckernel::detail {

#ifdef TRISC_MATH
/// Region of a legacy VectorMode other than RC, R and C.
inline constexpr TensorShape LEGACY_SINGLE_CALL_SHAPE = make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, 1, 1);

/// A TensorShape as a type, for passing to a generic lambda.
template <TensorShape Shape>
struct ShapeTag {
    static constexpr TensorShape value = Shape;
};

/// @brief Call @p call with the ShapeTag of the region @p vector_mode selects.
template <typename Call>
ALWI void with_legacy_vector_mode_shape(const VectorMode vector_mode, Call&& call) {
    if (vector_mode == VectorMode::RC) {
        call(ShapeTag<llk::sfpu::detail::vector_mode_shape<VectorMode::RC>>{});
    } else if (vector_mode == VectorMode::R) {
        call(ShapeTag<llk::sfpu::detail::vector_mode_shape<VectorMode::R>>{});
    } else if (vector_mode == VectorMode::C) {
        call(ShapeTag<llk::sfpu::detail::vector_mode_shape<VectorMode::C>>{});
    } else {
        call(ShapeTag<LEGACY_SINGLE_CALL_SHAPE>{});
    }
}
#endif

}  // namespace ckernel::detail
