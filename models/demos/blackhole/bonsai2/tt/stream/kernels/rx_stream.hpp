// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Receiver side of the decode weight stream (stream.py): this core's ring of RX_RING_SLOTS slots of RX_SLOT bytes, its
// "filled" counter (bumped by the sender), its "consumed" counter (kept across ops) and its sender's freed[r] entry.
// RX_INFO (define) is the address of this core's info record: receiver index, sender NOC x, y, index r in the
// sender's group.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

struct RxRing {
    uint32_t ring;
    volatile tt_l1_ptr uint32_t* filled;
    volatile tt_l1_ptr uint32_t* consumed;
    uint64_t freed;
    uint32_t c;
};

// args: ring, rcount (filled at +0, consumed at +32), the sender's freed array
FORCE_INLINE RxRing rx_open(uint32_t ring, uint32_t rcount, uint32_t freed_base) {
    volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RX_INFO);
    RxRing x;
    x.ring = ring;
    x.filled = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rcount);
    x.consumed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rcount + 32);
    x.freed = get_noc_addr(info[1], info[2], freed_base + 16 * info[3]);
    x.c = *x.consumed;
    return x;
}

FORCE_INLINE bool rx_ready(const RxRing& x) {
    invalidate_l1_cache();
    return (int32_t)(*x.filled - x.c) > 0;
}

// the next slot's address once it has landed
FORCE_INLINE uint32_t rx_wait(RxRing& x) {
    while (!rx_ready(x)) {
    }
    return x.ring + (x.c % RX_RING_SLOTS) * RX_SLOT;
}

FORCE_INLINE void rx_release(RxRing& x) {
    noc_semaphore_inc(x.freed, 1);
    ++x.c;
}

FORCE_INLINE void rx_close(RxRing& x) {
    *x.consumed = x.c;
    noc_async_atomic_barrier();
}
