// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// We use a magic value to initialize the ring buffer to, so that we can avoid printing it to the
// watcher log if no ring buffer data has been written. Choose -1 so that we can increment it to
// 0 and immediately use it as an index for the first write.
constexpr static int16_t DEBUG_RING_BUFFER_STARTING_INDEX = -1;

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#include <dev_msgs.h>

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER) && !defined(FORCE_WATCHER_OFF)

void push_to_ring_buffer(uint32_t val) {
    auto buf = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    volatile tt_l1_ptr int16_t* curr_ptr = &buf->current_ptr;
    volatile tt_l1_ptr uint16_t* wrapped = &buf->wrapped;
    uint32_t* data = buf->data;

    // Bounds check, set to -1 to wrap since we increment before using.
    if (*curr_ptr >= DEBUG_RING_BUFFER_ELEMENTS - 1) {
        *curr_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
        *wrapped = 1;
    }
    data[++(*curr_ptr)] = val;
}

#define WATCHER_RING_BUFFER_PUSH(x) push_to_ring_buffer(x)
#else  // !defined(WATCHER_ENABLED)
#define WATCHER_RING_BUFFER_PUSH(x)
#endif  // defined(WATCHER_ENABLED)

#endif  // KERNEL_BUILD || FW_BUILD
