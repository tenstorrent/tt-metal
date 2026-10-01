// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _DRAM_TEST_UTILS_H
#define _DRAM_TEST_UTILS_H

#include <stdint.h>
#include "noc_parameters.h"
#include "dev_mem_map.h"
#include "common_dram.hpp"

static_assert(NOC_WORD_BYTES == DRAM_TEST_NOC_WORD_BYTES, "NOC word size mismatch");
static_assert(MEM_DRAM_SIZE == DRAM_TEST_MAX_BANK_BYTES, "DRAM size mismatch");

static inline uint32_t dram_xorshift32_step(uint32_t& state) {
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

static inline uint32_t dram_choose_transfer_len(
    const DramTestParameters& p, uint32_t remaining_bytes, uint32_t& rng_state) {
    uint32_t len = p.chunk_bytes;

    if (p.max_burst_len != 0u && p.max_burst_len < len) {
        len = p.max_burst_len;
    }

    if (remaining_bytes < len) {
        len = remaining_bytes;
    }

    if (p.transfer_len_mode == 1u) {
        uint32_t r = dram_xorshift32_step(rng_state);
        r = ~(r | (r << 5) | (r << 8) | (r << 11));

        if (len != 0u) {
            len = (r % len) + 1u;
        }
    } else if (p.transfer_len_mode == 2u) {
        uint32_t r = dram_xorshift32_step(rng_state);

        uint32_t tile_size_bits = r & 0x1Fu;
        uint32_t header_size_bits = (r >> 5) & 0x3u;

        uint32_t tile_size = tile_size_bits * 512u;

        uint32_t header_size = 0u;
        if (header_size_bits == 1u) {
            header_size = NOC_WORD_BYTES;
        } else if (header_size_bits == 2u) {
            header_size = NOC_WORD_BYTES;
        } else if (header_size_bits == 3u) {
            header_size = 2u * NOC_WORD_BYTES;
        }

        uint32_t candidate = tile_size + header_size;
        if (candidate < NOC_WORD_BYTES) {
            candidate = NOC_WORD_BYTES;
        }

        if (candidate < len) {
            len = candidate;
        }
    }

    len = (len + NOC_WORD_BYTES - 1u) & ~(NOC_WORD_BYTES - 1u);

    if (len > remaining_bytes) {
        len = remaining_bytes & ~(NOC_WORD_BYTES - 1u);
        if (len == 0u) {
            len = remaining_bytes;
        }
    }

    return len;
}

#endif /* _DRAM_TEST_UTILS_H */
