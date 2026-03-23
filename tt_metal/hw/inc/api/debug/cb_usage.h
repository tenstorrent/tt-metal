// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/cb_usage.h
//
// Tracks circular buffer reserve/push/wait/pop usage per CB per processor.
// Used by the watcher to detect potential race conditions where
// multiple processors push or pop from the same CB, and to verify
// protocol correctness (reserve/push and wait/pop pairing, page balance).
// Also tracks kernel launch count.
// All functionality gated behind WATCHER_ENABLED && !WATCHER_DISABLE_CB_USAGE.
//
#pragma once

#include "hostdev/dev_msgs.h"
#include "internal/hw_thread.h"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_CB_USAGE) && !defined(FORCE_WATCHER_OFF)

inline void watcher_record_cb_reserve(uint32_t cb_id) {
    if (cb_id >= DEBUG_CB_USAGE_NUM_CBS) {
        return;
    }
    auto usage = GET_MAILBOX_ADDRESS_DEV(watcher.cb_usage);
    usage->reserve_count[cb_id]++;
    usage->reserve_risc_mask[cb_id] |= (1u << internal_::get_hw_thread_idx());
}

inline void watcher_record_cb_push(uint32_t cb_id, uint32_t num_pages) {
    if (cb_id >= DEBUG_CB_USAGE_NUM_CBS) {
        return;
    }
    auto usage = GET_MAILBOX_ADDRESS_DEV(watcher.cb_usage);
    usage->push_count[cb_id]++;
    usage->pages_pushed[cb_id] += num_pages;
    usage->push_risc_mask[cb_id] |= (1u << internal_::get_hw_thread_idx());
}

inline void watcher_record_cb_wait(uint32_t cb_id) {
    if (cb_id >= DEBUG_CB_USAGE_NUM_CBS) {
        return;
    }
    auto usage = GET_MAILBOX_ADDRESS_DEV(watcher.cb_usage);
    usage->wait_count[cb_id]++;
    usage->wait_risc_mask[cb_id] |= (1u << internal_::get_hw_thread_idx());
}

inline void watcher_record_cb_pop(uint32_t cb_id, uint32_t num_pages) {
    if (cb_id >= DEBUG_CB_USAGE_NUM_CBS) {
        return;
    }
    auto usage = GET_MAILBOX_ADDRESS_DEV(watcher.cb_usage);
    usage->pop_count[cb_id]++;
    usage->pages_popped[cb_id] += num_pages;
    usage->pop_risc_mask[cb_id] |= (1u << internal_::get_hw_thread_idx());
}

inline void watcher_increment_kernel_count() {
    auto usage = GET_MAILBOX_ADDRESS_DEV(watcher.cb_usage);
    usage->kernel_count++;
}

#define WATCHER_CB_RESERVE(cb_id) watcher_record_cb_reserve(cb_id)
#define WATCHER_CB_PUSH(cb_id, num_pages) watcher_record_cb_push(cb_id, num_pages)
#define WATCHER_CB_WAIT(cb_id) watcher_record_cb_wait(cb_id)
#define WATCHER_CB_POP(cb_id, num_pages) watcher_record_cb_pop(cb_id, num_pages)
#define WATCHER_KERNEL_COUNT_INC() watcher_increment_kernel_count()

#else  // !WATCHER_ENABLED || WATCHER_DISABLE_CB_USAGE || FORCE_WATCHER_OFF

#define WATCHER_CB_RESERVE(cb_id)
#define WATCHER_CB_PUSH(cb_id, num_pages)
#define WATCHER_CB_WAIT(cb_id)
#define WATCHER_CB_POP(cb_id, num_pages)
#define WATCHER_KERNEL_COUNT_INC()

#endif  // WATCHER_ENABLED && !WATCHER_DISABLE_CB_USAGE && !FORCE_WATCHER_OFF

#endif  // KERNEL_BUILD || FW_BUILD
