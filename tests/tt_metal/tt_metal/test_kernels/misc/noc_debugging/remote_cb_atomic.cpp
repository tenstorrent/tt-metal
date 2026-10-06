// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/remote_circular_buffer.h"

void kernel_main() {
    RemoteReceiverCBInterface receiver{};
    receiver.sender_noc_x = get_arg_val<uint32_t>(0);
    receiver.sender_noc_y = get_arg_val<uint32_t>(1);
    receiver.aligned_pages_acked_ptr = get_arg_val<uint32_t>(2);
    receiver.remote_pages_acked_ptr = get_arg_val<uint32_t>(3);

    experimental::detail::update_pages_acked(
        receiver, 1, noc_index, /*posted=*/false, experimental::detail::default_cmd_buf);

#if defined(USE_ATOMIC_BARRIER)
    noc_async_atomic_barrier();
#endif
}
