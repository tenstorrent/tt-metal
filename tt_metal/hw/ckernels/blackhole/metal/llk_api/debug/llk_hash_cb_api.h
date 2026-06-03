// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef DEBUG_CB_HASH
#include "internal/circular_buffer_interface.h"
#include "api/debug/device_print.h"
#endif

// Scalar FNV-1a over a circular buffer's L1 bytes, printed via DPRINT.
// Runs as plain RISC-V on whichever TRISC dispatches it; issues no Tensix
// instructions and touches no DEST/SFPU state. Intended for bisecting
// non-deterministic kernels by diffing per-stage hashes across runs.
//
// fifo_rd_ptr / fifo_page_size on TRISC are stored in 16B units; shift by
// cb_addr_shift (== CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT == 4) to get bytes.
//
// Gated on DEBUG_CB_HASH; compiles to an empty inline when undefined.
inline void llk_hash_cb(uint32_t cb_id, uint32_t num_tiles, uint32_t label) {
#ifdef DEBUG_CB_HASH
    const uint32_t base_bytes = get_local_cb_interface(cb_id).fifo_rd_ptr << cb_addr_shift;
    const uint32_t total_bytes = (num_tiles * get_local_cb_interface(cb_id).fifo_page_size) << cb_addr_shift;
    const uint32_t n_words = total_bytes >> 2;

    volatile tt_l1_ptr uint32_t* const p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_bytes);
    uint32_t h = 0x811c9dc5u;
    for (uint32_t i = 0; i < n_words; ++i) {
        h = (h ^ p[i]) * 0x01000193u;
    }

    // main made old-style DPRINT a hard error; use fmt-style DEVICE_PRINT (needs TT_METAL_DEVICE_PRINT=1).
    DEVICE_PRINT("hash[0x{:x}] cb={} tiles={} = 0x{:08x}", label, cb_id, num_tiles, h);
#else
    (void)cb_id;
    (void)num_tiles;
    (void)label;
#endif
}
