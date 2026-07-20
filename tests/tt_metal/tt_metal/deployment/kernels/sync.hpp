// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _SYNC_H
#define _SYNC_H

#include "sync_types.hpp"

static inline void spinlock_init(spinlock* lock) { lock->clear(std::memory_order_release); }

void spinlock_lock(spinlock* lock) {
    do {
        invalidate_l1_cache();
    } while (lock->test_and_set(std::memory_order_acquire));
}

static inline void spinlock_unlock(spinlock* lock) { lock->clear(std::memory_order_release); }

static inline void barrier_init(struct barrier* b, int total_threads) {
    spinlock_init(&b->lock);
    b->total_threads = total_threads;
    b->waiting_threads = 0;
    b->flag = 0;
}

void barrier_wait(struct barrier* b) {
    invalidate_l1_cache();
    uint32_t local_sense = !b->flag;

    spinlock_lock(&b->lock);

    b->waiting_threads++;

    if (b->total_threads == b->waiting_threads) {
        b->waiting_threads = 0;
        b->flag = local_sense;
        spinlock_unlock(&b->lock);
    } else {
        spinlock_unlock(&b->lock);
        do {
            invalidate_l1_cache();
        } while (b->flag != local_sense);
    }
}

#endif /* _SYNC_H */
