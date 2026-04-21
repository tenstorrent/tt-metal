// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"

/*************************************************************************
 * LLK TRANSPOSE DEST
 *************************************************************************/

/**
 * @brief Initialize FPU to perform a transpose operation on the destination register.
 * Transpose within faces is done by default, while transpose of faces is optional and passed as an argument.
 *
 * @tparam TRANSPOSE_OF_FACES: Set to true to transpose the faces of a tile (swap F1 and F2)
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit mode
 */
template <bool TRANSPOSE_OF_FACES, bool EN_32BIT_DEST>
inline void llk_math_transpose_dest_init() {
    _llk_math_transpose_dest_init_<TRANSPOSE_OF_FACES, EN_32BIT_DEST>();
}

/**
 * @brief Perform a transpose operation on the destination register.
 *
 * @param dst_index: The index of the tile in the destination register to transpose
 */
inline void llk_math_transpose_dest(std::uint32_t dst_index) { _llk_math_transpose_dest_(dst_index); }
