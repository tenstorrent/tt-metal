// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
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
    std::uint8_t face_r_dim;      ///< Row dimension of each face (typically 16)
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

/**
 * @brief Enumeration of all currently-supported TensorShape values.
 *
 * Matches validate_tensor_shape_tile_dependent_ops_ plus the per-axis HW limits,
 * yielding 20 unique shapes. Naming convention:
 *   TENSOR_SHAPE_FR{face_r_dim}_NF{num_faces_r_dim}x{num_faces_c_dim}
 *
 * Kept out of production kernel builds unless asserts or tensor-shape DPRINT
 * coverage need the table.
 */
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

// face_r_dim = 1
constexpr TensorShape TENSOR_SHAPE_FR1_NF1x1 = {1, MAX_FACE_C_DIM, 1, 1}; ///<  1 ×16
constexpr TensorShape TENSOR_SHAPE_FR1_NF1x2 = {1, MAX_FACE_C_DIM, 1, 2}; ///<  1 ×32
constexpr TensorShape TENSOR_SHAPE_FR1_NF2x1 = {1, MAX_FACE_C_DIM, 2, 1}; ///<  2 ×16
constexpr TensorShape TENSOR_SHAPE_FR1_NF2x2 = {1, MAX_FACE_C_DIM, 2, 2}; ///<  2 ×32

// face_r_dim = 2
constexpr TensorShape TENSOR_SHAPE_FR2_NF1x1 = {2, MAX_FACE_C_DIM, 1, 1}; ///<  2 ×16
constexpr TensorShape TENSOR_SHAPE_FR2_NF1x2 = {2, MAX_FACE_C_DIM, 1, 2}; ///<  2 ×32
constexpr TensorShape TENSOR_SHAPE_FR2_NF2x1 = {2, MAX_FACE_C_DIM, 2, 1}; ///<  4 ×16
constexpr TensorShape TENSOR_SHAPE_FR2_NF2x2 = {2, MAX_FACE_C_DIM, 2, 2}; ///<  4 ×32

// face_r_dim = 4
constexpr TensorShape TENSOR_SHAPE_FR4_NF1x1 = {4, MAX_FACE_C_DIM, 1, 1}; ///<  4 ×16
constexpr TensorShape TENSOR_SHAPE_FR4_NF1x2 = {4, MAX_FACE_C_DIM, 1, 2}; ///<  4 ×32
constexpr TensorShape TENSOR_SHAPE_FR4_NF2x1 = {4, MAX_FACE_C_DIM, 2, 1}; ///<  8 ×16
constexpr TensorShape TENSOR_SHAPE_FR4_NF2x2 = {4, MAX_FACE_C_DIM, 2, 2}; ///<  8 ×32

// face_r_dim = 8
constexpr TensorShape TENSOR_SHAPE_FR8_NF1x1 = {8, MAX_FACE_C_DIM, 1, 1}; ///<  8 ×16
constexpr TensorShape TENSOR_SHAPE_FR8_NF1x2 = {8, MAX_FACE_C_DIM, 1, 2}; ///<  8 ×32
constexpr TensorShape TENSOR_SHAPE_FR8_NF2x1 = {8, MAX_FACE_C_DIM, 2, 1}; ///< 16 ×16
constexpr TensorShape TENSOR_SHAPE_FR8_NF2x2 = {8, MAX_FACE_C_DIM, 2, 2}; ///< 16 ×32

// face_r_dim = 16
constexpr TensorShape TENSOR_SHAPE_FR16_NF1x1 = {16, MAX_FACE_C_DIM, 1, 1}; ///< 16 ×16
constexpr TensorShape TENSOR_SHAPE_FR16_NF1x2 = {16, MAX_FACE_C_DIM, 1, 2}; ///< 16 ×32
constexpr TensorShape TENSOR_SHAPE_FR16_NF2x1 = {16, MAX_FACE_C_DIM, 2, 1}; ///< 32 ×16
constexpr TensorShape TENSOR_SHAPE_FR16_NF2x2 = {16, MAX_FACE_C_DIM, 2, 2}; ///< 32 ×32 (== DEFAULT_TENSOR_SHAPE)

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

/// Build a TensorShape from explicit face dimensions and face-grid counts.
constexpr TensorShape make_tensor_shape(
    const std::uint8_t face_r_dim, const std::uint8_t face_c_dim, const std::uint8_t num_faces_r_dim, const std::uint8_t num_faces_c_dim)
{
    return TensorShape {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
}

/**
 * @brief Construct a TensorShape from the legacy (face_r_dim, num_faces) pair.
 *
 * Maps the historical scalar parameters used across LLK call sites:
 * - num_faces == 1: 1x1 face grid (face_r_dim × 16)
 * - num_faces == 2: 1x2 face grid (face_r_dim × 32)
 * - num_faces == 4: 2x2 face grid (face_r_dim*2 × 32; 32x32 when face_r_dim == 16)
 */
constexpr TensorShape make_tensor_shape_from_legacy(const std::uint8_t face_r_dim, const std::uint8_t num_faces)
{
    return TensorShape {face_r_dim, MAX_FACE_C_DIM, static_cast<std::uint8_t>(num_faces == 4 ? 2 : 1), static_cast<std::uint8_t>(num_faces == 1 ? 1 : 2)};
}

/**
 * @brief Validates shapes for ops that depend on face positioning within a tile.
 *
 * Keep this conservative until larger TensorShape variants have coverage.
 **/
__attribute__((noinline)) bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape)
{
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) && (face_c_dim == 16);
}

/**
 * @brief Fires the LLK_ASSERT for an unobserved TensorShape coverage hole.
 *
 * Keeping the assert in a noinline helper gives triage scripts a real frame
 * containing an `ASSERT(` token instead of pointing at the macro call site.
 */
__attribute__((noinline, cold)) inline void assert_tensor_shape_unobserved_()
{
    LLK_ASSERT(false, "TensorShape not observed before, please add it to the coverage table.");
}

} // namespace ckernel

/**
 * @brief Emit a TensorShape via DEVICE_PRINT (see tests/DPRINT.md).
 *
 * Place in front of validation asserts so the failing assert is preceded by the
 * offending shape. Outside debug/assert builds this compiles to a no-op.
 */
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

// Pull the correct DEVICE_PRINT wrapper only for DPRINT builds; assert-only builds
// still use the dedupe path but skip the print emission.
#ifdef DEBUG_PRINT_ENABLED
#ifdef ENV_LLK_INFRA
#include "dprint.h"
#else
#include "api/debug/dprint.h"
#endif
// Concatenate fn_name into the literal instead of using CTSTR(fn_name); CTSTR's
// COMDAT string object conflicts with DEVICE_PRINT's own string-section metadata
// at inline template call sites.
#define LLK_DPRINT_TENSOR_SHAPE_EMIT_(fn_name, ts)                                                         \
    DEVICE_PRINT(                                                                                          \
        "[" fn_name "] tensor_shape: face_r_dim={} face_c_dim={} num_faces_r_dim={} num_faces_c_dim={}\n", \
        static_cast<std::uint32_t>((ts).face_r_dim),                                                       \
        static_cast<std::uint32_t>((ts).face_c_dim),                                                       \
        static_cast<std::uint32_t>((ts).num_faces_r_dim),                                                  \
        static_cast<std::uint32_t>((ts).num_faces_c_dim))
#else
#define LLK_DPRINT_TENSOR_SHAPE_EMIT_(fn_name, ts) ((void)0)
#endif

// Per-call-site dedupe trims DPRINT volume to unique shapes per variant while
// preserving the host-side parser contract. The table covers the 20 valid shapes
// plus slack for out-of-bounds regressions.
#define LLK_DPRINT_DEDUP_MAX 32
#define LLK_DPRINT_TENSOR_SHAPE(fn_name, ts)                                                                                                                \
    do                                                                                                                                                      \
    {                                                                                                                                                       \
        static std::uint32_t _llk_dprint_seen[LLK_DPRINT_DEDUP_MAX] = {0};                                                                                  \
        static std::uint8_t _llk_dprint_count                       = 0;                                                                                    \
        const std::uint32_t _llk_dprint_key = (static_cast<std::uint32_t>((ts).face_r_dim) << 24) | (static_cast<std::uint32_t>((ts).face_c_dim) << 16) |   \
                                              (static_cast<std::uint32_t>((ts).num_faces_r_dim) << 8) | (static_cast<std::uint32_t>((ts).num_faces_c_dim)); \
        bool _llk_dprint_already_seen = false;                                                                                                              \
        for (std::uint8_t _llk_dprint_i = 0; _llk_dprint_i < _llk_dprint_count; ++_llk_dprint_i)                                                            \
        {                                                                                                                                                   \
            if (_llk_dprint_seen[_llk_dprint_i] == _llk_dprint_key)                                                                                         \
            {                                                                                                                                               \
                _llk_dprint_already_seen = true;                                                                                                            \
                break;                                                                                                                                      \
            }                                                                                                                                               \
        }                                                                                                                                                   \
        if (!_llk_dprint_already_seen && _llk_dprint_count < LLK_DPRINT_DEDUP_MAX)                                                                          \
        {                                                                                                                                                   \
            _llk_dprint_seen[_llk_dprint_count++] = _llk_dprint_key;                                                                                        \
            LLK_DPRINT_TENSOR_SHAPE_EMIT_(fn_name, ts);                                                                                                     \
            ::ckernel::assert_tensor_shape_unobserved_();                                                                                                   \
        }                                                                                                                                                   \
    } while (0)

#else

#define LLK_DPRINT_TENSOR_SHAPE(fn_name, ts) ((void)0)

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
