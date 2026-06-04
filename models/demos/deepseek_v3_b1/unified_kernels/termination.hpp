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
