// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke-test worker receiver for DramSenderGlobalCircularBuffer. Waits for
// num_pages pushed by the DRISC sender, then pops them. The data itself
// lives at the GCB cb_buffer address in this worker's L1; the host reads it
// directly after the program finishes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);

    experimental::remote_cb_wait_front(remote_cb_id, num_pages);
    experimental::remote_cb_pop_front(remote_cb_id, num_pages);
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
}
