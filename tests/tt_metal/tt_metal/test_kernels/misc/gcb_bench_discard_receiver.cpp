// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bandwidth-bench worker receiver for the DRAM-core vs worker-core prefetcher
// BW comparison. Per-page wait_front + pop_front in a loop, discards data.
//
// Unlike gcb_smoke_receiver.cpp (one big wait_front(num_pages) + pop_front(num_pages)),
// this kernel drains pages one-at-a-time so the GCB fifo (which only holds a few
// pages of in-flight data) keeps refilling as the sender pushes through num_iters
// total pages.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_iters = get_compile_time_arg_val(1);

    for (uint32_t i = 0; i < num_iters; ++i) {
        experimental::remote_cb_wait_front(remote_cb_id, 1);
        experimental::remote_cb_pop_front(remote_cb_id, 1);
    }
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
}
