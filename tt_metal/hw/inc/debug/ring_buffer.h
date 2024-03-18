// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(WATCHER_ENABLED)

#include "hostdevcommon/debug_ring_buffer_common.h"

// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline uint32_t* get_debug_ring_buffer() {
#if defined(COMPILE_FOR_ERISC)
    return reinterpret_cast<uint32_t*>(eth_l1_mem::address_map::ERISC_RING_BUFFER_ADDR);
#else
    return reinterpret_cast<uint32_t*>(RING_BUFFER_ADDR);
#endif
}

void push_to_ring_buffer(uint32_t val) {
    auto buf = get_debug_ring_buffer();
    volatile tt_l1_ptr int32_t* curr_ptr = &reinterpret_cast<DebugRingBufMemLayout *>(buf)->current_ptr;
    uint32_t* data = reinterpret_cast<DebugRingBufMemLayout *>(buf)->data;

    // Bounds check, set to -1 to wrap since we increment before using.
    if (*curr_ptr >= RING_BUFFER_ELEMENTS - 1)
        *curr_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
    data[++(*curr_ptr)] = val;
}

#define WATCHER_RING_BUFFER_PUSH(x) push_to_ring_buffer(x)
#else  // !defined(WATCHER_ENABLED)
#define WATCHER_RING_BUFFER_PUSH(x)
#endif  // defined(WATCHER_ENABLED)
