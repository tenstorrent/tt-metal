// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tune-helper mcast_pipe bake-off: RECEIVER micro-kernel (object API).
// Style axes via -D defines (must match the sender):
//   STAGING_COUNTER(0/1) -> 0: flag wait(VALID)+reset | 1: counter wait_min(iter+1)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

#ifndef STAGING_COUNTER
#define STAGING_COUNTER 0
#endif

void kernel_main() {
    constexpr uint32_t cb_dst = get_compile_time_arg_val(0);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t payload_pages = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_iters = get_compile_time_arg_val(5);
    constexpr uint32_t pre_handshake = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_start_id = get_arg_val<uint32_t>(1);
    const uint32_t sender_x = get_arg_val<uint32_t>(2);
    const uint32_t sender_y = get_arg_val<uint32_t>(3);

    Noc noc;
    Semaphore<> data_ready_sem(data_ready_sem_id);
    Semaphore<> consumed_sem(consumed_sem_id);
    CircularBuffer cb_dst_obj(cb_dst);

    // reserve the landing region: write_ptr == base == the address the sender mcasts to
    cb_dst_obj.reserve_back(payload_pages);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        if constexpr (pre_handshake) {
            // tell the sender "my dest is free / I am ready" (remote atomic inc on sender's counter)
            consumed_sem.up(noc, sender_x, sender_y, 1);
        }
#if STAGING_COUNTER
        data_ready_sem.wait_min(iter + 1);
#else
        data_ready_sem.wait(VALID);
        data_ready_sem.set(INVALID);  // reset for next round (clear-before-next-signal)
#endif
    }

    cb_dst_obj.push_back(payload_pages);

    // write the received payload out to DRAM for host verification
    const auto out = TensorAccessor(out_args, output_addr);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        cb_dst_obj.wait_front(1);
        noc.async_write(cb_dst_obj, out, page_bytes, {}, {.page_id = output_start_id + i});
        noc.async_writes_flushed();
        cb_dst_obj.pop_front(1);
    }
    noc.async_write_barrier();
}
