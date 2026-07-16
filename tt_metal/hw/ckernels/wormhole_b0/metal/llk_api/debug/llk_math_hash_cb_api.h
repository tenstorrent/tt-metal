// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef DEBUG_CB_HASH
#include "llk_math_common_api.h"
#include "debug/llk_math_hash_cb.h"
#endif

// ===========================================================================
// LLK MATH HASH CB (SFPU) — MATH-side surface for hash_cb_sfpu.
//
// Wraps the SFPU-backed FNV23 lanewise hash in llk_lib/debug/llk_math_hash_cb.h.
// Orchestration is in api/compute/debug/cb_hash.h; see that header for the
// user-facing contract. Call order: init -> tile (per input tile) -> store_to_dest.
// ===========================================================================

/**
 * @brief Seed the per-lane FNV23 accumulators and configure DEST addressing.
 *
 * @pre Call once before any @ref llk_math_hash_cb_tile.
 */
inline void llk_math_hash_cb_init() {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_init_();
#endif
}

/**
 * @brief Fold one INT32 DEST tile into the 32 per-lane FNV23 accumulators.
 *
 * @param dst_tile_idx: DEST tile slot holding the input (the orchestration uses slot 0).
 * @pre @ref llk_math_hash_cb_init, and the tile already datacopied into DEST.
 */
inline void llk_math_hash_cb_tile(uint32_t dst_tile_idx) {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_tile_(dst_tile_idx);
#else
    (void)dst_tile_idx;
#endif
}

/**
 * @brief Write the 32 per-lane accumulators back into DEST for the packer.
 *
 * Leaves the accumulators in DEST row 0 (rest of the tile zeroed) so the packer
 * can move the tile to L1, where a scalar consumer XOR-folds it.
 *
 * @pre @ref llk_math_hash_cb_tile has folded all input tiles.
 * @post Caller packs DEST; see api/compute/debug/cb_hash.h.
 */
inline void llk_math_hash_cb_store_to_dest() {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_store_to_dest_();
#endif
}
