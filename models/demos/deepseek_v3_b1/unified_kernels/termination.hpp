// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

namespace deepseek_b1_ops {

FORCE_INLINE bool socket_wait_for_pages_with_termination(
    const SocketReceiverInterface& socket, uint32_t num_pages, volatile tt_l1_ptr uint32_t* termination_semaphore) {
    while (!socket_wait_for_pages(socket, num_pages, 1000)) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            return false;
        }
    }
    return true;
}

// waits for free space in the host FIFO before sending; stops waiting if host process terminates
FORCE_INLINE bool socket_reserve_pages_with_termination(
    const SocketSenderInterface& socket, uint32_t num_pages, volatile tt_l1_ptr uint32_t* termination_semaphore) {
    uint32_t num_bytes = num_pages * socket.page_size;
    volatile tt_l1_ptr uint32_t* bytes_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_base_addr);
    uint32_t bytes_acked_end = socket.bytes_acked_base_addr + socket.num_downstreams * bytes_acked_size_bytes;
    while (reinterpret_cast<uint32_t>(bytes_acked_ptr) < bytes_acked_end) {
        uint32_t bytes_free;
        do {
            invalidate_l1_cache();
            if (termination_semaphore[0] == 1) {
                return false;
            }
            bytes_free = socket.downstream_fifo_total_size - (socket.bytes_sent - *bytes_acked_ptr);
        } while (bytes_free < num_bytes);
        bytes_acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            reinterpret_cast<uint32_t>(bytes_acked_ptr) + bytes_acked_size_bytes);
    }
    return true;
}

FORCE_INLINE bool cb_wait_for_pages_with_termination(
    uint32_t cb_index, uint32_t num_pages, volatile tt_l1_ptr uint32_t* termination_semaphore) {
    while (!cb_pages_available_at_front(cb_index, num_pages)) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            return false;
        }
    }
    return true;
}

}  // namespace deepseek_b1_ops
