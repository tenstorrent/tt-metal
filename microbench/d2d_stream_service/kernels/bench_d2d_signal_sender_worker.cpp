// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Signal-only sender-side worker for the D2D stream-service THROUGHPUT benchmark.
//
// Unlike placeholder_d2d_sender_worker.cpp (which fills the whole backing tensor
// from L1 on every worker core), this kernel does NOT touch the backing tensor: the
// benchmark leaves the sender backing resident in DRAM and the persistent sender
// service re-reads it over fabric each transfer. Filling here would add
// num_cores * tensor_bytes of redundant DRAM writes per iteration, swamping the
// fabric-transfer time we want to measure. So the worker only runs the inverted
// handshake against the sender service:
//
//   Per iteration:
//     1. (designated worker only, if metadata enabled) write a {-1, 0, fill_base+iter}
//        blob into the sender service core's metadata L1 before acking,
//     2. atomic-inc data_ready_counter on the sender service core (the service waits
//        for num_workers of these, then ships the backing tensor over fabric),
//     3. spin on the local consumed_sem until the service multicast-incs it (transfer
//        drained), then reset it to 0.
//
// Runs a fixed num_iters then exits, so the host can Finish() the workload.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"

constexpr uint32_t consumed_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t num_iters = get_compile_time_arg_val(1);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t fill_base = get_compile_time_arg_val(3);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(4);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(5);

void kernel_main() {
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(3);       // 1 only on the designated core
    const uint32_t sender_metadata_l1_addr = get_arg_val<uint32_t>(4);  // service-core L1 metadata buffer

    volatile tt_l1_ptr uint32_t* consumed_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem_addr);
    const uint64_t data_ready_counter_noc = get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);

    Noc noc;
    // CB-backed scratch slot is only used as the NoC source for the metadata write
    // (a CB-backed address, never a stack-local — required for NoC multicast/unicast).
    CircularBuffer scratch_cb(scratch_cb_index);
    const uint32_t cb_l1_addr = scratch_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1_addr);
    CoreLocalMem<uint32_t> scratch_src(cb_l1_addr);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // 1. (designated core only) stage this iter's metadata blob on the sender
        //    service core's L1 before acking, so the service ships it with the transfer.
        if constexpr (metadata_enabled) {
            if (is_metadata_writer != 0) {
                scratch[0] = static_cast<uint32_t>(-1);
                scratch[1] = 0u;
                scratch[2] = fill_base + iter;
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
