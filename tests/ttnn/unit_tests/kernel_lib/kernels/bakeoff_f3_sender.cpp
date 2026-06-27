// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tune-helper mcast_pipe F3 bake-off: SENDER-IN-RECT micro-kernel (object API).
// The sender is one core of the mcast rectangle and must end up with the payload in its OWN
// cb_dst (like every receiver), then writes that to its own output shard for verification.
// Two arms, switched by -D LOOPBACK_INCLUDE:
//   1  -> INCLUDE_SRC: one mcast, hardware loopback-writes the payload to self too.
//   0  -> EXCLUDE_SRC + a local NoC self-copy (cb_src -> cb_dst) — the "unicast on the sender".
// Only the sender runs this kernel; the OTHER rect cores run bakeoff_mcast_receiver.cpp.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

#ifndef LOOPBACK_INCLUDE
#define LOOPBACK_INCLUDE 1
#endif

#if LOOPBACK_INCLUDE
constexpr auto MCAST_MODE = Noc::McastMode::INCLUDE_SRC;
#else
constexpr auto MCAST_MODE = Noc::McastMode::EXCLUDE_SRC;
#endif

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_dests = get_compile_time_arg_val(3);  // INCLUDE: rect cores; EXCLUDE: rect-1
    constexpr uint32_t payload_pages = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_iters = get_compile_time_arg_val(6);
    constexpr auto in_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    const uint32_t x0 = get_arg_val<uint32_t>(2);
    const uint32_t y0 = get_arg_val<uint32_t>(3);
    const uint32_t x1 = get_arg_val<uint32_t>(4);
    const uint32_t y1 = get_arg_val<uint32_t>(5);
    const uint32_t output_addr = get_arg_val<uint32_t>(6);
    const uint32_t self_start_id = get_arg_val<uint32_t>(7);

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    Semaphore<> data_ready_sem(data_ready_sem_id);
    CircularBuffer cb_src_obj(cb_src);
    CircularBuffer cb_dst_obj(cb_dst);
    MulticastEndpoint mcast_ep;

    // stage payload into cb_src from DRAM
    const auto in = TensorAccessor(in_args, input_addr);
    cb_src_obj.reserve_back(payload_pages);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        noc.async_read(in, cb_src_obj, page_bytes, {.page_id = input_start_id + i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    cb_src_obj.push_back(payload_pages);
    cb_src_obj.wait_front(payload_pages);

    cb_dst_obj.reserve_back(payload_pages);  // write_ptr == base == mcast dst addr
    const uint32_t dst_addr = cb_dst_obj.get_write_ptr();
    const uint32_t src_addr = cb_src_obj.get_read_ptr();

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        noc.async_write_multicast<MCAST_MODE>(
            cb_src_obj,
            mcast_ep,
            payload_bytes,
            num_dests,
            {},
            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = dst_addr},
            /*linked=*/false);
#if !LOOPBACK_INCLUDE
        // EXCLUDE arm: hardware skipped self -> fill our own dst with a local NoC self-copy.
        const uint64_t self_noc_src = get_noc_addr(my_x[noc_index], my_y[noc_index], src_addr);
        noc_async_read(self_noc_src, dst_addr, payload_bytes);
#endif
        data_ready_sem.set(VALID);
        data_ready_sem.set_multicast<MCAST_MODE>(noc, x0, y0, x1, y1, num_dests, /*linked=*/false);
        noc.async_writes_flushed();
#if !LOOPBACK_INCLUDE
        noc.async_read_barrier();  // ensure the self-copy landed before we read cb_dst below
#endif
    }

    cb_dst_obj.push_back(payload_pages);

    // write our own received payload out to DRAM (sender's shard) for verification
    const auto out = TensorAccessor(out_args, output_addr);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        cb_dst_obj.wait_front(1);
        noc.async_write(cb_dst_obj, out, page_bytes, {}, {.page_id = self_start_id + i});
        noc.async_writes_flushed();
        cb_dst_obj.pop_front(1);
    }
    noc.async_write_barrier();
}
