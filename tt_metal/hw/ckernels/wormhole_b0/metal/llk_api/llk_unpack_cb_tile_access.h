// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/circular_buffer_interface.h"
#include "ckernel_globals.h"
#include "llk_assert.h"

/**
 * @file
 * @brief Unpack CB / tile access layer (GH #26202).
 *
 * Single place that owns how the unpacker indexes into circular buffers by
 * tile:
 *   - Production accessors (always run): compute the per-tile stride and L1
 *     read address used by the unpacker.
 *   - Validation entry points (run only inside LLK_ASSERT_BLOCK): perform the
 *     unpack CB/tile access checks.
 *
 * Background: the LLK uses a CB's @c fifo_page_size as the per-tile byte stride
 * when indexing tiles. That is only correct when the page size equals the
 * operand's actual tile size. The real tile size is available independently as
 * @c unpack_tile_size[] (constexpr, 16B words, derived from the operand data
 * format); @c fifo_page_size for compute is also in 16B words, so the two are
 * directly comparable.
 */

namespace ckernel {

/**
 * @brief Per-tile stride, in 16B words, for unpack/read indexing of an operand.
 *
 * @param cb_id Operand (circular buffer) index.
 * @return The CB's @c fifo_page_size, used as the per-tile stride.
 */
FORCE_INLINE std::uint32_t llk_unpack_tile_stride(std::uint32_t cb_id) {
    return get_local_cb_interface(cb_id).fifo_page_size;
}

/**
 * @brief L1 read address, in 16B words, of a tile within an operand.
 *
 * Centralizes the indexed-read address pattern, including the @c fifo_rd_ptr-1
 * base adjustment used by the unpacker.
 *
 * @param cb_id      Operand (circular buffer) index.
 * @param tile_index Zero-based tile index within the CB.
 * @return The L1 address of the requested tile.
 */
FORCE_INLINE std::uint32_t llk_unpack_tile_address(std::uint32_t cb_id, std::uint32_t tile_index) {
    return (get_local_cb_interface(cb_id).fifo_rd_ptr - 1) + llk_unpack_tile_stride(cb_id) * tile_index;
}

/**
 * @brief Validate that an operand's CB page size matches its actual tile size.
 *
 * Guards the GH #26202 invariant on which the unpacker's tile-offset arithmetic
 * relies. Intended to be invoked through @c LLK_ASSERT_BLOCK.
 *
 * @param cb_id Operand (circular buffer) index.
 */
inline void validate_unpack_tile_layout(std::uint32_t cb_id) {
    LLK_ASSERT(
        get_local_cb_interface(cb_id).fifo_page_size == static_cast<std::uint32_t>(unpack_tile_size[cb_id]),
        "CB page_size != unpack tile_size: tile-offset arithmetic assumes page_size == tile_size (GH #26202)");
}

/**
 * @brief Validate an indexed unpack/read access.
 *
 * Runs all unpack CB/tile access checks and is intended to be invoked through
 * @c LLK_ASSERT_BLOCK so the entire call compiles out when asserts are
 * disabled. Marked @c noinline (mirroring the other LLK validators such as
 * @c is_unpacker_A_configured_correctly) so the check code is emitted once
 * rather than at every call site.
 *
 * Checks:
 *   - @c fifo_page_size matches the operand's actual tile size (GH #26202).
 *   - The requested tile range stays within the CB bounds.
 *
 * @param cb_id            Operand (circular buffer) index.
 * @param start_tile_index Zero-based index of the first tile accessed.
 * @param num_tiles        Number of tiles accessed (defaults to 1).
 */
inline __attribute__((noinline)) void validate_unpack_tile_access(
    std::uint32_t cb_id, std::uint32_t start_tile_index, std::uint32_t num_tiles = 1) {
    validate_unpack_tile_layout(cb_id);
    LLK_ASSERT(cb_access_within_bounds(cb_id, start_tile_index, num_tiles), "Indexed tile read exceeds CB boundary");
}

}  // namespace ckernel
