// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef DEBUG_CB_HASH
#include "risc_common.h"
#include "internal/circular_buffer_interface.h"
#include "api/debug/dprint.h"
#endif

// ===========================================================================
// LLK-API entrypoints for the DEBUG_CB_HASH compute-API surface.
//
//   - llk_hash_cb_trisc : scalar FNV-1a-32 over a CB's L1 bytes, printed via
//                         DPRINT. Pure RISC-V, no Tensix Engine state.
//
// The SFPU variant's MATH side lives in debug/llk_math_hash_cb_api.h; its result
// is left in DEST for the packer (see api/compute/debug/cb_hash.h::hash_cb_sfpu).
//
// All entrypoints expand to empty inlines when DEBUG_CB_HASH is undefined.
// ===========================================================================

#ifdef DEBUG_CB_HASH
// FNV-1a-32 constants (word-granularity variant; see cb_hash.h).
static constexpr uint32_t FNV1A32_INIT = 0x811c9dc5u;
static constexpr uint32_t FNV1A32_PRIME = 0x01000193u;
#endif

/**
 * @brief Scalar FNV-1a-32 over a circular buffer's L1 bytes, printed via DPRINT.
 *
 * Pure RISC-V; touches no Tensix Engine state. fifo_rd_ptr / fifo_page_size on
 * TRISC are stored in 16B units, so they are shifted by cb_addr_shift
 * (== CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT == 4) to get bytes.
 *
 * @param cb_id: Circular buffer to hash.
 * @param num_tiles: Number of tiles from the front of the CB to include.
 * @param label: Caller tag echoed in the DPRINT line.
 * @note Runs on whichever TRISC dispatches it; defaults to UNPACK (the thread
 *       with a DPRINT buffer and a populated fifo_rd_ptr).
 */
inline void llk_hash_cb_trisc(uint32_t cb_id, uint32_t num_tiles, uint32_t label) {
#ifdef DEBUG_CB_HASH
    const uint32_t base_bytes = get_local_cb_interface(cb_id).fifo_rd_ptr << cb_addr_shift;
    const uint32_t total_bytes = (num_tiles * get_local_cb_interface(cb_id).fifo_page_size) << cb_addr_shift;
    const uint32_t n_words = total_bytes >> 2;

    // volatile to keep the FNV loop intact (no vectorization/reordering).
    volatile tt_l1_ptr uint32_t* const p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_bytes);
    uint32_t h = FNV1A32_INIT;
    for (uint32_t i = 0; i < n_words; ++i) {
        h = (h ^ p[i]) * FNV1A32_PRIME;
    }

    // Print the hash in the diff-friendly format documented in cb_hash.h.
    // DPRINT is safe here because hash_cb_trisc defaults to the UNPACK thread
    // (the only TRISC with a DPRINT buffer) and the hash loop above has
    // completed — no concurrent L1 access conflicts.
    DPRINT("hash[0x{:x}] cb={} tiles={} = 0x{:x}\n", label, cb_id, num_tiles, h);
#else
    (void)cb_id;
    (void)num_tiles;
    (void)label;
#endif
}
