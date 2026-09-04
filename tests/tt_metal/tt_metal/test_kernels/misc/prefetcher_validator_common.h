// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The half of the prefetcher validator receivers that does not depend on the transport: deriving a
// receiver's expected tiles for one block, and comparing them against what was delivered.
//
// The derivation is the prefetcher-vs-matmul contract itself
// (tt_metal/impl/buffers/prefetcher_matmul_design.md, §3 "Per-block source tiles"), and the point
// of a validator is that the host and the kernel arrive at it independently. It lives here so the
// GlobalCircularBuffer and PrefetcherPipe receivers cannot drift into validating two different
// contracts -- each only supplies its own wait/read/pop.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace prefetcher_validator {

// Gather this receiver's expected tiles for one block into `scratch_addr`, laid out as the sender
// pushes them: page row h holds tiles (phys_blk * k_block_w_tiles + h, n_col_start + n) for n in
// [0, n_per_recv_tiles). One accessor call per tile keeps bank routing out of the kernel. Returns
// once the reads have landed.
template <typename Accessor>
FORCE_INLINE void read_expected_block_tiles(
    const Accessor& accessor,
    uint32_t scratch_addr,
    uint32_t tile_bytes,
    uint32_t phys_blk,
    uint32_t k_block_w_tiles,
    uint32_t total_n_tiles,
    uint32_t n_col_start,
    uint32_t n_per_recv_tiles) {
    uint32_t scratch_cursor = scratch_addr;
    for (uint32_t h = 0; h < k_block_w_tiles; ++h) {
        const uint32_t k_row = phys_blk * k_block_w_tiles + h;
        const uint32_t row_page_base = k_row * total_n_tiles + n_col_start;
        for (uint32_t n = 0; n < n_per_recv_tiles; ++n) {
            const uint64_t src_noc = accessor.get_noc_addr(row_page_base + n);
            noc_async_read(src_noc, scratch_cursor, tile_bytes);
            scratch_cursor += tile_bytes;
        }
    }
    noc_async_read_barrier();
}

// Index of the first word where the delivered block differs from the expected one, or the word
// count when they match. Word-strided so a mismatch reports where it starts rather than just that
// it happened.
FORCE_INLINE uint32_t first_mismatching_word(uint32_t received_addr, uint32_t expected_addr, uint32_t page_bytes) {
    volatile tt_l1_ptr uint32_t* received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(received_addr);
    volatile tt_l1_ptr uint32_t* expected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(expected_addr);
    const uint32_t words = page_bytes / sizeof(uint32_t);
    for (uint32_t w = 0; w < words; ++w) {
        if (received[w] != expected[w]) {
            return w;
        }
    }
    return words;
}

}  // namespace prefetcher_validator
