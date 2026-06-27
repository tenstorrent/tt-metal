// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tune-helper mcast_pipe bake-off: SENDER micro-kernel (object API).
// One kernel, style axes switched by -D defines from the harness:
//   FENCE_BARRIER  (0/1)  -> 0: async_writes_flushed (SENT)  | 1: async_write_barrier (ACKed)
//   STAGING_COUNTER(0/1)  -> 0: flag set_multicast(VALID)    | 1: counter inc_multicast + wait_min
//   LOOPBACK_INCLUDE(0/1) -> 0: EXCLUDE_SRC                  | 1: INCLUDE_SRC
//   LINKED         (0/1)  -> 0: unlinked                     | 1: linked data+flag pair, flush only
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

#ifndef FENCE_BARRIER
#define FENCE_BARRIER 0
#endif
#ifndef STAGING_COUNTER
#define STAGING_COUNTER 0
#endif
#ifndef LOOPBACK_INCLUDE
#define LOOPBACK_INCLUDE 0
#endif
#ifndef LINKED
#define LINKED 0
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
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t num_dests = get_compile_time_arg_val(4);
    constexpr uint32_t payload_pages = get_compile_time_arg_val(5);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t num_iters = get_compile_time_arg_val(7);
    constexpr uint32_t pre_handshake = get_compile_time_arg_val(8);
    constexpr auto in_args = TensorAccessorArgs<9>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    const uint32_t x0 = get_arg_val<uint32_t>(2);
    const uint32_t y0 = get_arg_val<uint32_t>(3);
    const uint32_t x1 = get_arg_val<uint32_t>(4);
    const uint32_t y1 = get_arg_val<uint32_t>(5);

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    Semaphore<> data_ready_sem(data_ready_sem_id);
    Semaphore<> consumed_sem(consumed_sem_id);
    CircularBuffer cb_src_obj(cb_src);
    CircularBuffer cb_dst_obj(cb_dst);
    MulticastEndpoint mcast_ep;

    // --- stage the payload into cb_src (read from DRAM input) ---
    const auto in = TensorAccessor(in_args, input_addr);
    cb_src_obj.reserve_back(payload_pages);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        noc.async_read(in, cb_src_obj, page_bytes, {.page_id = input_start_id + i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    cb_src_obj.push_back(payload_pages);
    cb_src_obj.wait_front(payload_pages);  // read_ptr now points at the staged payload

    // the common mcast destination L1 address (cb_dst shares index/addr with receivers)
    const uint32_t dst_addr = cb_dst_obj.get_write_ptr();

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        if constexpr (pre_handshake) {
            consumed_sem.wait(num_dests);
            consumed_sem.set(0);
        }

        // ---- data multicast ----
        noc.async_write_multicast<MCAST_MODE>(
            cb_src_obj,
            mcast_ep,
            payload_bytes,
            num_dests,
            {},
            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = dst_addr},
            /*linked=*/LINKED != 0);

        // ---- handshake (data-ready signal) ----
#if STAGING_COUNTER
        data_ready_sem.inc_multicast(noc, x0, y0, x1, y1, /*value=*/1, num_dests);
#else
        data_ready_sem.set(VALID);
        data_ready_sem.set_multicast<MCAST_MODE>(noc, x0, y0, x1, y1, num_dests, /*linked=*/false);
#endif

        // ---- fence ----
#if STAGING_COUNTER
        // inc_multicast is a NON-POSTED multicast atomic: it expects num_dests ACKs that must be
        // drained, so the data write flush is NOT sufficient — an atomic barrier is required.
        noc.async_writes_flushed();
        noc.async_atomic_barrier();
#elif LINKED
        noc.async_writes_flushed();  // linked pair: flush only, no barrier between
#elif FENCE_BARRIER
        noc.async_write_barrier();
#else
        noc.async_writes_flushed();
#endif
    }
}
