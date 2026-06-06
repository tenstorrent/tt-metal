// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/circular_buffer_interface.h"
#include "ckernel_globals.h"
#include "llk_assert.h"

/**
 * @file
 * @brief Pack CB / tile access layer (GH #26202).
 *
 * Single place that owns how the packer indexes into circular buffers by tile:
 *   - Production accessor (always run): the per-tile stride used by the packer
 *     when advancing the write pointer.
 *   - Validation entry points (run only inside LLK_ASSERT_BLOCK): perform the
 *     pack CB/tile access checks.
 *
 * Background: the LLK uses a CB's @c fifo_page_size as the per-tile byte stride
 * when indexing tiles. That is only correct when the page size equals the
 * operand's actual tile size. The real tile size is available independently as
 * @c pack_tile_size[] (constexpr, 16B words, derived from the operand data
 * format); @c fifo_page_size for compute is also in 16B words, so the two are
 * directly comparable.
 */

namespace ckernel {

/**
 * @brief Per-tile stride, in 16B words, for pack/write indexing of an output.
 *
 * @param cb_id Output (circular buffer) index.
 * @return The CB's @c fifo_page_size, used as the per-tile stride.
 */
FORCE_INLINE std::uint32_t llk_pack_tile_stride(std::uint32_t cb_id) {
    return get_local_cb_interface(cb_id).fifo_page_size;
}

/**
 * @brief Validate that an output's CB page size matches its actual tile size.
 *
 * Guards the GH #26202 invariant on which the packer's tile-offset arithmetic
 * relies. Intended to be invoked through @c LLK_ASSERT_BLOCK.
 *
 * @param cb_id Output (circular buffer) index.
 */
inline void validate_pack_tile_layout(std::uint32_t cb_id) {
    LLK_ASSERT(
        get_local_cb_interface(cb_id).fifo_page_size == static_cast<std::uint32_t>(pack_tile_size[cb_id]),
        "CB page_size != pack tile_size: tile-offset arithmetic assumes page_size == tile_size (GH #26202)");
}

/**
 * @brief Validate a pack/push of tiles into an output CB.
 *
 * Runs all pack CB/tile access checks and is intended to be invoked through
 * @c LLK_ASSERT_BLOCK so the entire call compiles out when asserts are
 * disabled. Marked @c noinline (mirroring the other LLK validators such as
 * @c are_packers_configured_correctly) so the check code is emitted once rather
 * than at every call site.
 *
 * Checks:
 *   - @c fifo_page_size matches the output's actual tile size (GH #26202).
 *   - The write pointer is not already at or past @c fifo_limit.
 *   - Pushing @p num_tiles tiles would not advance the write pointer past
 *     @c fifo_limit.
 *
 * @param cb_id     Output (circular buffer) index.
 * @param num_tiles Number of tiles being pushed.
 */
inline __attribute__((noinline)) void validate_pack_tile_push(std::uint32_t cb_id, std::uint32_t num_tiles) {
    validate_pack_tile_layout(cb_id);

    const auto& cb = get_local_cb_interface(cb_id);
    LLK_ASSERT(cb.fifo_wr_ptr < cb.fifo_limit, "CB push_back: fifo_wr_ptr already at or past fifo_limit");
    LLK_ASSERT(
        (cb.fifo_limit - cb.fifo_wr_ptr) >= num_tiles * llk_pack_tile_stride(cb_id),
        "CB push_back: fifo_wr_ptr would exceed fifo_limit");
}

}  // namespace ckernel
