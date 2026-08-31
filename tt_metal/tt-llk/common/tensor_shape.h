// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"

namespace ckernel
{

/*
   The current max constraints are set for large default size of 32x32, but that is until tensorShape is piped to all ops
   Once it is piped to all ops, we can relax max number of faces, to be closer to description of a tensorshape
*/
constexpr std::uint8_t MAX_FACE_R_DIM      = 16;
constexpr std::uint8_t MAX_FACE_C_DIM      = 16;
constexpr std::uint8_t MAX_TILE_R_DIM      = 32;
constexpr std::uint8_t MAX_TILE_C_DIM      = 32;
constexpr std::uint8_t MAX_NUM_FACES_R_DIM = 2;
constexpr std::uint8_t MAX_NUM_FACES_C_DIM = 2;
constexpr std::uint8_t MAX_NUM_FACES       = MAX_NUM_FACES_R_DIM * MAX_NUM_FACES_C_DIM;

constexpr std::uint8_t MAX_FPU_ROWS           = 8;
constexpr std::uint8_t MAX_FPU_ROWS_LOG2      = 3;
constexpr std::uint8_t MAX_TILES_IN_HALF_DEST = 8;

/**
 * @brief Standardized tensor shape representation for LLK operations.
 *
 * Replaces inconsistent tile size parameters (num_faces, face_r_dim,
 * narrow_tile, partial_face, VectorMode) with a unified struct.
 *
 * A tile is composed of faces arranged in a grid:
 * - Total tile rows = face_r_dim * num_faces_r_dim
 * - Total tile cols = face_c_dim * num_faces_c_dim
 *
 * Example: 32x32 tile = 4 faces of 16x16 (num_faces_r_dim=2, num_faces_c_dim=2)
 * Example: 32x16 tile = 2 faces of 16x16 (num_faces_r_dim=2, num_faces_c_dim=1)
 */
struct __attribute__((packed)) TensorShape
{
    std::uint8_t face_r_dim;      ///< Row dimension of each face (valid: 1/2/4/8/16; full face is 16)
    std::uint8_t face_c_dim;      ///< Column dimension of each face (always 16 for HW)
    std::uint8_t num_faces_r_dim; ///< Number of faces in row dimension
    std::uint8_t num_faces_c_dim; ///< Number of faces in column dimension

    /// @brief Get total tile row dimension
    constexpr std::uint16_t total_row_dim() const
    {
        return static_cast<std::uint16_t>(face_r_dim) * num_faces_r_dim;
    }

    /// @brief Get total tile column dimension
    constexpr std::uint16_t total_col_dim() const
    {
        return static_cast<std::uint16_t>(face_c_dim) * num_faces_c_dim;
    }

    /// @brief Get total number of datums in the tile
    constexpr std::uint16_t total_tensor_size() const
    {
        return total_row_dim() * total_col_dim();
    }

    /// @brief Get total number of faces
    constexpr std::uint8_t total_num_faces() const
    {
        return num_faces_r_dim * num_faces_c_dim;
    }
};

static_assert(sizeof(TensorShape) == 4, "TensorShape must be 4 bytes");

constexpr TensorShape DEFAULT_TENSOR_SHAPE = {MAX_FACE_R_DIM, MAX_FACE_C_DIM, MAX_NUM_FACES_R_DIM, MAX_NUM_FACES_C_DIM};

/// Build a TensorShape from explicit face dimensions and face-grid counts.
constexpr TensorShape make_tensor_shape(
    const std::uint8_t face_r_dim, const std::uint8_t face_c_dim, const std::uint8_t num_faces_r_dim, const std::uint8_t num_faces_c_dim)
{
    return TensorShape {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
}

/**
 * @brief Build a TensorShape from a face row dimension and flat face count.
 *
 * Bridges legacy APIs that still carry a flat face_r_dim / num_faces (e.g. CB metadata) into a
 * TensorShape. Argument order matches @ref make_tensor_shape_from_legacy: (face_r_dim, num_faces).
 * Uses a canonical face-grid decomposition for tile-dependent ops, constrained to
 * num_faces in {1, 2, 4}:
 *   - 1 -> 1x1 (e.g. 16x16 with face_r_dim=16)
 *   - 2 -> 1x2 (e.g. 16x32 with face_r_dim=16)
 *   - 4 -> 2x2 (e.g. 32x32 with face_r_dim=16)
 * The face column dimension is always 16 in hardware.
 *
 * Ambiguity: a flat num_faces == 2 cannot distinguish wide 16x32 (1x2) from narrow 32x16 (2x1).
 * This helper always chooses 1x2. It cannot express 32x16 / FR16_NF2x1. Callers that need a
 * narrow tile must use @ref make_tensor_shape with an explicit face grid instead. Metal-side
 * paths that already know real row/col dims (e.g. get_operand_tensor_shape) should not go
 * through this helper.
 *
 * @param face_r_dim: Row dimension of each face (defaults to the full 16-row face).
 * @param num_faces: Total number of faces in the tile (1, 2, or 4).
 */
constexpr TensorShape tensor_shape_from_num_faces(const std::uint32_t face_r_dim = MAX_FACE_R_DIM, const std::uint32_t num_faces = MAX_NUM_FACES)
{
    // num_faces == 2 always maps to 1x2 (wide), never 2x1 (narrow). See note above.
    const std::uint8_t num_faces_r_dim = (num_faces == 4) ? 2 : 1;
    const std::uint8_t num_faces_c_dim = (num_faces == 4) ? 2 : static_cast<std::uint8_t>(num_faces);
    return TensorShape {static_cast<std::uint8_t>(face_r_dim), MAX_FACE_C_DIM, num_faces_r_dim, num_faces_c_dim};
}

/**
 * @brief Whether Input 0 (SrcB) and Input 1 (SrcA) form a supported matmul TensorShape pair.
 */
constexpr bool validate_matmul_tensor_shapes_(const TensorShape& src_b_shape, const TensorShape& src_a_shape)
{
    const bool supported_width =
        src_a_shape.face_c_dim == MAX_FACE_C_DIM && (src_a_shape.num_faces_c_dim == 1 || src_a_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM);
    const bool wide_src_b = src_b_shape.face_c_dim == MAX_FACE_C_DIM && src_b_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM &&
                            ((src_b_shape.num_faces_r_dim == 1 && (src_b_shape.face_r_dim == 1 || src_b_shape.face_r_dim == 2 || src_b_shape.face_r_dim == 4 ||
                                                                   src_b_shape.face_r_dim == 8 || src_b_shape.face_r_dim == MAX_FACE_R_DIM)) ||
                             (src_b_shape.face_r_dim == MAX_FACE_R_DIM && src_b_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM));
    const bool full_k_src_a = src_a_shape.face_r_dim == MAX_FACE_R_DIM && src_a_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM;
    const bool narrow_src_b = src_b_shape.face_r_dim == MAX_FACE_R_DIM && src_b_shape.face_c_dim == MAX_FACE_C_DIM &&
                              src_b_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM && src_b_shape.num_faces_c_dim == 1;
    const bool half_k_src_a      = src_a_shape.face_r_dim == MAX_FACE_R_DIM && src_a_shape.num_faces_r_dim == 1;
    const bool single_face_src_b = src_b_shape.face_r_dim == MAX_FACE_R_DIM && src_b_shape.face_c_dim == MAX_FACE_C_DIM && src_b_shape.num_faces_r_dim == 1 &&
                                   src_b_shape.num_faces_c_dim == 1;
    const bool single_face_src_a = half_k_src_a && src_a_shape.num_faces_c_dim == 1;
    return supported_width && ((wide_src_b && full_k_src_a) || (narrow_src_b && half_k_src_a) || (single_face_src_b && single_face_src_a));
}

/**
 * @brief Construct a TensorShape from the legacy (face_r_dim, num_faces) pair.
 *
 * Same argument order, mapping, and limitations as @ref tensor_shape_from_num_faces: num_faces == 2
 * becomes 1x2 (16x32-class), not 2x1 (32x16). Prefer @ref make_tensor_shape for narrow tiles.
 */
inline TensorShape make_tensor_shape_from_legacy(const std::uint8_t face_r_dim, const std::uint8_t num_faces)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be one of the valid values: 1, 2, or 4");
    return tensor_shape_from_num_faces(face_r_dim, num_faces);
}

/**
 * @brief Validates shapes for ops that depend on face positioning within a tile.
 *
 * Keep this conservative until larger TensorShape variants have coverage.
 **/
__attribute__((noinline)) inline bool validate_tensor_shape_tile_dependent_ops_(const TensorShape& tensor_shape)
{
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
}

/**
 * @brief Whether a tile shape is supported by the SDPA blocked bcast-col SUB path.
 *
 * Stricter than @ref validate_tensor_shape_tile_dependent_ops_ on purpose: the COL face pattern needs
 * two full 16-row faces per face-row (the math walk covers a face-row with four 8-row ops and reads
 * the face-row's first SrcB face twice), so only a 2-face-column grid of full faces works - 32x32
 * (2x2 faces) or 16x32 (1x2 faces). A shape with one face column (32x16) or short faces
 * (face_r_dim < 16) would make the walk write more dest rows than the tile owns.
 *
 * One predicate for all three call sites of the op: Unpack init, Math init, and the math
 * execute that derives the dest slot stride from the same shape, instead of hand-synced copies.
 **/
constexpr bool validate_tensor_shape_sub_bcast_col_custom_(const TensorShape& tensor_shape)
{
    return tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.face_c_dim == MAX_FACE_C_DIM && tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM &&
           (tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM || tensor_shape.num_faces_r_dim == 1);
}

} // namespace ckernel
