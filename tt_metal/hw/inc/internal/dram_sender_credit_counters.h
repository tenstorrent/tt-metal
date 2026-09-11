// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The credit bookkeeping the two DRAM-sender transports share.
//
// A programmable DRAM core (Blackhole DRISC) can drive either a DRAM-sender GlobalCircularBuffer or
// a DRAM-sender PrefetcherPipe. Both keep one (sent, acked) counter pair per receiver in the
// sender's own L1, count credit in L1_ALIGNMENT-sized units, and read acks the same way; they
// differ only in the byte stride between pairs, which is why that stride is a parameter here rather
// than a constant. A GlobalCircularBuffer packs the pairs at uint32 stride
// (REMOTE_CB_LOCAL_PAGES_STRIDE), a PrefetcherPipe keeps the 2 * L1_ALIGNMENT stride its
// worker-sender counterpart uses. Either way acked sits half a stride above sent.
//
// These are the two loops whose reasoning is easy to get subtly wrong -- the clamp below, and the
// ack spin -- so both transports run one copy.

// Include order, as for its sibling internal/prefetcher_pipe_dram_sender.h: this header names
// invalidate_l1_cache() and so must follow the dataflow API a DRISC kernel already includes.

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"

namespace experimental {

// Free credit units at the most-backed-up receiver, without blocking. `ring_units` is the ring's
// capacity in the same units, and doubles as the answer when nothing is outstanding.
FORCE_INLINE uint32_t dram_sender_min_free_units(
    uint32_t local_counters_ptr, uint32_t num_receivers, uint32_t local_pages_stride, uint32_t ring_units) {
    volatile tt_l1_ptr uint32_t* sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_counters_ptr);
    volatile tt_l1_ptr uint32_t* acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_counters_ptr + local_pages_stride / 2);
    uint32_t min_free = ring_units;
    invalidate_l1_cache();
    for (uint32_t r = 0; r < num_receivers; ++r) {
        const uint32_t outstanding = *sent_ptr - *acked_ptr;
        // Clamp rather than subtract blindly: a resize padding credit (sent bumped without a
        // free-space reserve) can transiently push sent further ahead of acked than the ring holds,
        // and an underflow here would wrap to a huge value and defeat receiver backpressure.
        const uint32_t free_units = outstanding >= ring_units ? 0u : ring_units - outstanding;
        if (free_units < min_free) {
            min_free = free_units;
        }
        sent_ptr += local_pages_stride / sizeof(uint32_t);
        acked_ptr += local_pages_stride / sizeof(uint32_t);
    }
    return min_free;
}

// Spin until every receiver has acked everything this sender published.
FORCE_INLINE void dram_sender_barrier(
    uint32_t local_counters_ptr, uint32_t num_receivers, uint32_t local_pages_stride) {
    volatile tt_l1_ptr uint32_t* sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_counters_ptr);
    for (uint32_t r = 0; r < num_receivers; ++r) {
        volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (local_pages_stride / 2) / sizeof(uint32_t);
        while (true) {
            invalidate_l1_cache();
            if (*acked_ptr == *sent_ptr) {
                break;
            }
        }
        sent_ptr += local_pages_stride / sizeof(uint32_t);
    }
}

}  // namespace experimental
