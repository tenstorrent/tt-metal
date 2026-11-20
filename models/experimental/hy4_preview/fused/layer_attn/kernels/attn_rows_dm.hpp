// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Data-movement helpers of the 16-row kernels of program P2 (attn_rows.hpp says what the row mask and the bytes of
// a count word are). HY4A_MASK: the L1 address, the same on every core of every chip, of the pass's row mask.

#pragma once

#include <cstdint>

#include "attn_common.hpp"
#include "attn_rows.hpp"

namespace hy4a {

FORCE_INLINE uint32_t row_mask() {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(HY4A_MASK) & ((1u << hy4r::TOKEN_ROWS) - 1u);
}

// Poll until byte p of this core's word i reaches `need`. False when the bound ran out (wait_word says what then).
inline bool wait_field(uint32_t i, uint32_t p, uint32_t need, uint32_t code) {
    volatile tt_l1_ptr uint32_t* w = word(i);
    uint32_t seen = 0;
    for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT; ++polls) {
        invalidate_l1_cache();
        seen = w[0];
        if (hy4r::field(seen, p) >= need) {
            return true;
        }
    }
    word(A_STATUS)[0] = code | (seen & 0xFFFFFF);
    return false;
}

// Take `n` off byte p of this core's word i (an atomic on itself).
FORCE_INLINE void take_field(uint32_t i, uint32_t p, uint32_t n) {
    noc_semaphore_inc(get_noc_addr(word_addr(i)), 0u - n * hy4r::one(p));
    noc_async_atomic_barrier();
}

}  // namespace hy4a
