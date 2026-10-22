// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/circular_buffer_interface.h"
#include "llk_assert.h"
#include "noc_parameters.h"

// Page size in 16 B words. A descriptor tile size that is a multiple of DRAM_ALIGNMENT is the page size;
// factories that align pages up (tiny block-float tiles) leave it unequal, so those read L1.
inline uint32_t get_cb_page_size(const uint32_t cb_id, const uint32_t tile_bytes) {
    if (tile_bytes % DRAM_ALIGNMENT != 0) {
        return get_local_cb_interface(cb_id).fifo_page_size;
    }
    LLK_ASSERT(
        (tile_bytes >> 4) == get_local_cb_interface(cb_id).fifo_page_size,
        "CB page size differs from the descriptor tile size");
    return tile_bytes >> 4;
}
