// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "llk_assert.h"
#include "llk_operands.h"

/**
 * @brief Faces per custom_mm in1 tile, read from the operand's CB: 4 for [32, 32], 2 for [32, 16].
 *
 * Reads the CB's 2D face grid, since a flat face count of 2 cannot tell [32, 16] (2x1 faces) from [16, 32]
 * (1x2 faces).
 *
 * @tparam transpose: Whether in1 is read transposed; [32, 16] is only supported without transpose.
 * @param operand_id: Operand id of in1.
 */
template <bool transpose = false>
inline std::uint32_t llk_custom_mm_in1_num_faces(const std::uint32_t operand_id) {
    const ckernel::TensorShape in1_shape = get_operand_tensor_shape(operand_id);
    LLK_ASSERT(
        in1_shape.face_r_dim == FACE_R_DIM && in1_shape.num_faces_r_dim == 2 &&
            (in1_shape.num_faces_c_dim == 2 || (in1_shape.num_faces_c_dim == 1 && !transpose)),
        "custom_mm: in1 tile shape must be [32, 32], or [32, 16] without transpose");
    return in1_shape.num_faces_c_dim == 1 ? 2 : TILE_NUM_FACES;
}
