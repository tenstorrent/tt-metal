// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Placeholder sender-side worker kernel for D2DStreamService tests. Stands in
// for a real producer op: it fills the sender backing tensor with a
// per-iteration value, then runs the inverted handshake against the persistent
// sender service kernel.
//
// Per iteration:
//   1. write value (fill_base + iter) into every page of the sender backing
//      tensor,
//   2. atomic-inc data_ready_counter on the sender service core (the service
//      kernel waits for num_workers of these),
//   3. spin on the local consumed_sem until the service multicast-incs it
//      (transfer drained over fabric), then reset it to 0.
//
// Runs a fixed num_iters then exits, so a test can Finish() the worker workload.
// `fill_base` lets the host pick a distinct per-launch seed (used by the reuse
// test to give each round a unique value).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t consumed_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t num_pages = get_compile_time_arg_val(2);
constexpr uint32_t tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_iters = get_compile_time_arg_val(4);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t fill_base = get_compile_time_arg_val(6);
// Optional metadata (indices 7..8). When enabled, the designated worker writes a
// {-1, 0, fill_base+iter} blob into the sender service core's metadata L1 before
// acking. Unused when metadata_enabled == 0.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(7);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(8);
constexpr auto backing_tensor_accessor_args = TensorAccessorArgs<9>();

void kernel_main() {
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(3);       // 1 only on the designated core
    const uint32_t sender_metadata_l1_addr = get_arg_val<uint32_t>(4);  // service-core L1 metadata buffer

    auto backing = TensorAccessor(backing_tensor_accessor_args, backing_tensor_addr);

    volatile tt_l1_ptr uint32_t* consumed_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem_addr);
    const uint64_t data_ready_counter_noc = get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);

    // 2.0 NoC interface for the L1->DRAM backing fill and the unicast metadata write.
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    const uint32_t cb_l1_addr = scratch_cb.get_write_ptr();
    // Raw volatile view for the CPU-side L1 fill loop (kept as-is: that is an L1
    // store loop, not a NoC data-movement op). CoreLocalMem view is the NoC source.
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1_addr);
    CoreLocalMem<uint32_t> scratch_src(cb_l1_addr);
    const uint32_t page_elems = tensor_page_size / sizeof(uint32_t);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // 1. Produce this iteration's slice as an iota: the row-major element at
        //    global index i holds (base + i), base = fill_base + iter. Distinct per
        //    element (so a transposed / mis-addressed page is caught) and shifted by
        //    1 every iteration (so a stuck/reused/partially-written transfer reads
        //    back the wrong base). Each page holds different values, so refill the
        //    scratch CB per page and barrier before reusing it as the NoC source.
        const uint32_t base = fill_base + iter;
        for (uint32_t p = 0; p < num_pages; ++p) {
            const uint32_t page_base = base + p * page_elems;
            for (uint32_t e = 0; e < page_elems; ++e) {
                scratch[e] = page_base + e;
            }
            noc.async_write(scratch_src, backing, tensor_page_size, {}, {.page_id = p});
            noc.async_write_barrier();
        }

        // 1b. (Designated core only) write this iter's metadata blob to the sender
        //     service core's L1 metadata buffer, BEFORE acking. The data writes are
        //     already flushed (barrier above), so the scratch CB is free to reuse as
        //     the NoC source (a CB-backed address — never a stack-local). The
        //     metadata scalar is the iteration's base offset.
        if constexpr (metadata_enabled) {
            if (is_metadata_writer != 0) {
                scratch[0] = static_cast<uint32_t>(-1);
                scratch[1] = 0u;
                scratch[2] = base;
                noc.async_write(
                    scratch_src,
                    UnicastEndpoint{},
                    metadata_size_bytes,
                    {},
                    {.noc_x = service_noc_x, .noc_y = service_noc_y, .addr = sender_metadata_l1_addr});
                noc.async_write_barrier();
            }
        }

        // 2. Ack into data_ready_counter — the service kernel waits for num_workers.
        noc_semaphore_inc(data_ready_counter_noc, 1);
        noc.async_atomic_barrier();

        // 3. Wait for the service to confirm the transfer drained, then reset.
        while (*consumed_sem == 0) {
            invalidate_l1_cache();
        }
        *consumed_sem = 0;
    }
}
