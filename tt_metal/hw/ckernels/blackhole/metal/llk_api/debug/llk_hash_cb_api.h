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
//   - llk_hash_cb_trisc            : scalar FNV-1a-32 over a CB's L1 bytes.
//                                    Pure RISC-V, no Tensix Engine state.
//   - llk_hash_cb_sfpu_reset_ready : UNPACK side: clear the L1 ready flag
//                                    before the SFPU variant starts.
//   - llk_hash_cb_sfpu_print_from_l1: UNPACK side: poll the L1 ready flag,
//                                    then read the hash u32 and DPRINT.
//
// The SFPU variant's MATH side lives in debug/llk_math_hash_cb_api.h;
// orchestration is in api/compute/debug/cb_hash.h::hash_cb_sfpu.
//
// All entrypoints expand to empty inlines when DEBUG_CB_HASH is undefined.
// ===========================================================================

// Scalar FNV-1a-32 over a circular buffer's L1 bytes, printed via DPRINT.
// fifo_rd_ptr / fifo_page_size on TRISC are stored in 16B units; shift by
// cb_addr_shift (== CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT == 4) to get bytes.
inline void llk_hash_cb_trisc(uint32_t cb_id, uint32_t num_tiles, uint32_t label) {
#ifdef DEBUG_CB_HASH
    const uint32_t base_bytes = get_local_cb_interface(cb_id).fifo_rd_ptr << cb_addr_shift;
    const uint32_t total_bytes = (num_tiles * get_local_cb_interface(cb_id).fifo_page_size) << cb_addr_shift;
    const uint32_t n_words = total_bytes >> 2;

    // volatile to keep the FNV loop intact (no vectorization/reordering).
    volatile tt_l1_ptr uint32_t* const p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_bytes);
    uint32_t h = 0x811c9dc5u;
    for (uint32_t i = 0; i < n_words; ++i) {
        h = (h ^ p[i]) * 0x01000193u;
    }

    // Print the hash in the diff-friendly format documented in cb_hash.h.
    // DPRINT is safe here because hash_cb_trisc defaults to the UNPACK thread
    // (the only TRISC with a DPRINT buffer) and the hash loop above has
    // completed — no concurrent L1 access conflicts.
    DPRINT << "hash[0x" << HEX() << label << "] cb=" << DEC() << cb_id << " tiles=" << num_tiles << " = 0x" << HEX()
           << h << DEC() << ENDL();
#else
    (void)cb_id;
    (void)num_tiles;
    (void)label;
#endif
}

// UNPACK-side helper for hash_cb_sfpu. Clears the L1 ready flag before MATH
// publishes a new hash, so the post-compute poll cannot return early on a
// stale flag from a previous probe.
inline void llk_hash_cb_sfpu_reset_ready(uint32_t l1_ready_addr) {
#ifdef DEBUG_CB_HASH
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_ready_addr) = 0u;
#else
    (void)l1_ready_addr;
#endif
}

// UNPACK-side helper for hash_cb_sfpu. Polls the L1 ready flag set by MATH,
// then reads the hash u32 out of L1 and prints in the same line format as
// llk_hash_cb_trisc so the two variants diff cleanly side by side.
inline void llk_hash_cb_sfpu_print_from_l1(
    uint32_t l1_hash_addr, uint32_t l1_ready_addr, uint32_t cb_id, uint32_t num_tiles, uint32_t label) {
#ifdef DEBUG_CB_HASH
    volatile tt_l1_ptr uint32_t* const ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_ready_addr);
    while (*ready_ptr == 0u) {
        invalidate_l1_cache();
    }
    const uint32_t h = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_hash_addr);
    DPRINT << "hash[0x" << HEX() << label << "] cb=" << DEC() << cb_id << " tiles=" << num_tiles << " = 0x" << HEX()
           << h << DEC() << ENDL();
#else
    (void)l1_hash_addr;
    (void)l1_ready_addr;
    (void)cb_id;
    (void)num_tiles;
    (void)label;
#endif
}
