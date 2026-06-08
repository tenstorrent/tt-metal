// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Single worker kernel for D2HStreamService worker-sync.
//
// Every worker core runs this same kernel; the only role difference is a runtime
// arg (is_master). There is no cross-talk between worker cores:
//   * Every worker:  wait transfer_done -> write its backing slice -> notify the
//     service core (atomic-inc write_ack).
//   * The master (is_master == 1) additionally fans the replicated inline metadata
//     IN to the service-core staging region before it acks. The metadata is
//     identical on every worker core, so a single designated core forwards it.
//
// The service core waits for all num_workers acks before streaming, so the master
// writing metadata before its own ack is sufficient ordering — no inter-worker
// semaphore is needed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(3);
// 0 when no inline metadata; otherwise the replicated metadata byte count.
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(4);
// Local worker L1 holding this core's replicated metadata copy (the master's src).
constexpr uint32_t worker_metadata_l1_addr = get_compile_time_arg_val(5);
constexpr auto acc_args = TensorAccessorArgs<6>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t fill_seed = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t write_ack_counter_addr = get_arg_val<uint32_t>(5);
    // Master role + the service-core staging region this core fans metadata in to.
    // Only meaningful when metadata_size_bytes > 0 && is_master == 1.
    const uint32_t is_master = get_arg_val<uint32_t>(6);
    const uint32_t metadata_input_addr = get_arg_val<uint32_t>(7);

    auto backing = TensorAccessor(acc_args, backing_tensor_addr);
    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* transfer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(transfer_done_sem_addr);

    const uint64_t write_ack_noc = get_noc_addr(service_noc_x, service_noc_y, write_ack_counter_addr);

    while (true) {
        invalidate_l1_cache();
        if (*transfer_done > 0) {
            *transfer_done = 0;
            break;
        }
    }

    for (uint32_t p = start_page; p < end_page; ++p) {
        for (uint32_t i = 0; i < page_size / sizeof(uint32_t); ++i) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1)[i] = fill_seed + p + i;
        }
        noc_async_write<page_size>(cb_l1, backing.get_noc_addr(p), page_size);
    }
    noc_async_write_barrier();

    if constexpr (metadata_size_bytes > 0) {
        if (is_master) {
            // Fan the replicated metadata IN to the service-core staging region the
            // sender ships from. worker_metadata_l1_addr is a NOC-accessible src, so
            // issue the NOC write straight from it — no intermediate CB copy needed.
            const uint64_t metadata_service_noc = get_noc_addr(service_noc_x, service_noc_y, metadata_input_addr);
            noc_async_write(worker_metadata_l1_addr, metadata_service_noc, metadata_size_bytes);
            noc_async_write_barrier();
        }
    }

    noc_semaphore_inc(write_ack_noc, 1);
    noc_async_atomic_barrier();
}
