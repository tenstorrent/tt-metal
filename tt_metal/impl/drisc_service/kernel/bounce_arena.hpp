// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "api/alignment.h"

// FIFO byte ring over the bounce region. Allocations are freed in allocation order, so free space is always
// size - used and no head pointer is needed. Each allocation is 64 B aligned (NOC address congruence). One that would
// cross the end of the region starts at the base instead, and the skipped tail bytes are charged to it, so free()
// needs only the alloc_bytes that try_alloc returned.
class BounceArena {
public:
    BounceArena(uint32_t base, uint32_t size) : base_(base), size_(size) {}

    struct Allocation {
        uint32_t addr;
        uint32_t alloc_bytes;  // ring bytes charged: the 64 B round-up plus any skipped tail
    };

    // Returns false, changing nothing, when the ring cannot fit `bytes` now.
    bool try_alloc(uint32_t bytes, Allocation& out) {
        const uint32_t aligned = align_power_of_2(bytes, 64);
        const bool wrap = tail_ + aligned > size_;
        const uint32_t pad = wrap ? size_ - tail_ : 0;
        if (used_ + pad + aligned > size_) {
            return false;
        }
        if (wrap) {
            tail_ = 0;
        }
        out = {base_ + tail_, pad + aligned};
        tail_ += aligned;
        used_ += out.alloc_bytes;
        return true;
    }

    // `alloc_bytes` must come from the oldest live allocation.
    void free(uint32_t alloc_bytes) { used_ -= alloc_bytes; }

private:
    uint32_t base_;
    uint32_t size_;
    uint32_t tail_ = 0;
    uint32_t used_ = 0;
};
