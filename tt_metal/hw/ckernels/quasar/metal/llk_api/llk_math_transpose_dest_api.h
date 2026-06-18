// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"

/**
 * @brief Performs an in-place 32x32 transpose on a tile in the destination
 * register at dst_index.
 *
 * @tparam transpose_of_faces Transpose faces as well
 * @tparam EN_32BIT_DEST True if dest is in 32-bit mode.
 */
template <bool transpose_of_faces = true, bool EN_32BIT_DEST = false>
inline void llk_math_transpose_dest(uint dst_index) {
    _llk_math_transpose_dest_(dst_index);
}

/**
 * @brief Initializes transpose-dest. Reprograms the ALU to FP32-dest mode when
 * EN_32BIT_DEST=true (required for Int32 transpose-dest, for Float32 the
 * ALU is already in fp32_dest mode so this is a no-op write), then loads
 * the bank0 replay buffer with the transpose-dest MOP.
 *
 * @tparam transpose_of_faces Transpose faces as well.
 * @tparam EN_32BIT_DEST True if dest is in 32-bit mode.
 */
template <bool transpose_of_faces = true, bool EN_32BIT_DEST = false>
inline void llk_math_transpose_dest_init() {
    if constexpr (EN_32BIT_DEST) {
        _llk_math_upk_to_dest_hw_configure_<
            false /*EN_IMPLIED_MATH_FORMAT*/,
            true /*EN_FP32_MATH_FORMAT*/,
            false /*EN_INT32_MATH_FORMAT*/>();
    }
    _llk_math_transpose_dest_init_<transpose_of_faces, EN_32BIT_DEST>();
}
