// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Generic-op receiver for the Tensor prefetcher: drains `num_pages` GCB pages one at a time
// (like the bench discard receiver) and, on the LAST page of the run, copies the page into this
// core's shard of an L1-sharded output tensor so the host can check the byte layout the
// prefetcher delivered (block block_count-1 of this receiver's columns in batched mode).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t copy_last = get_compile_time_arg_val(3);
    const uint32_t out_addr = get_arg_val<uint32_t>(0);  // this core's L1 output shard (same on every core)

    for (uint32_t i = 0; i < num_pages; ++i) {
        experimental::remote_cb_wait_front(remote_cb_id, 1);
        if (copy_last && i + 1 == num_pages) {
            // Local L1 -> L1 copy through the NoC (own core), the fastest bulk copy a RISC can issue.
            const uint32_t src = get_remote_receiver_cb_interface(remote_cb_id).fifo_rd_ptr;
            noc_async_write(src, get_noc_addr(out_addr), page_bytes);
            noc_async_write_barrier();
        }
        experimental::remote_cb_pop_front(remote_cb_id, 1);
    }
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
}
