// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TensorShape helpers for LLK tests only.

#include <cstdint>

#include "tensor_shape.h"

/**
 * @brief Build a TensorShape from total tile dimensions.
 *
 * Uses the default face split for a tile: up to 16 rows is one face row with face_r_dim = tile_r_dim,
 * 32 rows is two 16-row face rows; 16 columns is one face column, 32 columns is two.
 * Examples: 8x32 -> {8, 16, 1, 2}, 32x16 -> {16, 16, 2, 1}, 32x32 -> {16, 16, 2, 2}.
 *
 * @param tile_r_dim: Tile rows (1, 2, 4, 8, 16, or 32).
 * @param tile_c_dim: Tile columns (16 or 32).
 */
constexpr ckernel::TensorShape tensor_shape_from_tile_dims(const std::uint32_t tile_r_dim, const std::uint32_t tile_c_dim)
{
    const std::uint8_t face_r_dim      = static_cast<std::uint8_t>(tile_r_dim > ckernel::MAX_FACE_R_DIM ? ckernel::MAX_FACE_R_DIM : tile_r_dim);
    const std::uint8_t num_faces_r_dim = tile_r_dim > ckernel::MAX_FACE_R_DIM ? ckernel::MAX_NUM_FACES_R_DIM : 1;
    const std::uint8_t num_faces_c_dim = tile_c_dim > ckernel::MAX_FACE_C_DIM ? ckernel::MAX_NUM_FACES_C_DIM : 1;
    return ckernel::TensorShape {face_r_dim, ckernel::MAX_FACE_C_DIM, num_faces_r_dim, num_faces_c_dim};
}
