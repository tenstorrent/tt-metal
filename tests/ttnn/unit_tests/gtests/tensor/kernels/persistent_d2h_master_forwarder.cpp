// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Master forwarder for D2HStreamService worker-sync with metadata.
//
// The metadata is identical (replicated) on every worker core, so a single
// designated core forwards it host-ward instead of all of them. This is the D2H
// reverse of H2D: H2D fans metadata OUT from the service core to the workers;
// here the master fans it IN from worker L1 to the service core, where the
// persistent sender picks it up and ships it to the host.
//
// Compile-time designated core (fixed via Config::master_forwarder_core). One iteration:
//   1. Wait for transfer_done multicast (backing unlocked).
//   2. Write this core's backing slice.
//   3. Wait for peer workers to finish writing (worker_done roll call).
//   4. Read this core's replicated metadata copy from local worker L1 and write
//      it to the service-core staging region the sender ships from.
//   5. Multicast metadata_ready so peers can proceed to ack.
//   6. Atomic-inc the service-core write_ack counter.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t worker_done_counter_addr = get_compile_time_arg_val(4);
constexpr uint32_t num_peer_workers = get_compile_time_arg_val(5);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(6);
constexpr uint32_t worker_metadata_l1_addr = get_compile_time_arg_val(7);
constexpr uint32_t metadata_ready_sem_addr = get_compile_time_arg_val(8);
constexpr uint32_t worker_mcast_noc_x_start = get_compile_time_arg_val(9);
constexpr uint32_t worker_mcast_noc_y_start = get_compile_time_arg_val(10);
constexpr uint32_t worker_mcast_noc_x_end = get_compile_time_arg_val(11);
constexpr uint32_t worker_mcast_noc_y_end = get_compile_time_arg_val(12);
constexpr auto acc_args = TensorAccessorArgs<13>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t fill_seed = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t write_ack_counter_addr = get_arg_val<uint32_t>(5);
    const uint32_t metadata_input_addr = get_arg_val<uint32_t>(6);

    auto backing = TensorAccessor(acc_args, backing_tensor_addr);
    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* transfer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(transfer_done_sem_addr);
    volatile tt_l1_ptr uint32_t* worker_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_done_counter_addr);

    const uint64_t write_ack_noc = get_noc_addr(service_noc_x, service_noc_y, write_ack_counter_addr);
    // Destination of the metadata fan-in: the service-core staging region the
    // sender ships from.
    const uint64_t metadata_service_noc = get_noc_addr(service_noc_x, service_noc_y, metadata_input_addr);
    const uint64_t metadata_ready_mcast = get_noc_multicast_addr(
        worker_mcast_noc_x_start,
        worker_mcast_noc_y_start,
        worker_mcast_noc_x_end,
        worker_mcast_noc_y_end,
        metadata_ready_sem_addr);

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

    while (true) {
        invalidate_l1_cache();
        if (*worker_done >= num_peer_workers) {
            *worker_done = 0;
            break;
        }
    }

    // Fan the replicated metadata IN to the host. The metadata is identical on
    // every worker core, so the master reads its own local copy from worker L1
    // and writes it to the service-core staging region the sender ships from.
    invalidate_l1_cache();
    auto* worker_metadata_src = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(worker_metadata_l1_addr);
    auto* cb = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cb_l1);
    for (uint32_t i = 0; i < metadata_size_bytes; ++i) {
        cb[i] = worker_metadata_src[i];
    }
    noc_async_write(cb_l1, metadata_service_noc, metadata_size_bytes);
    noc_async_write_barrier();

    // Release the peer workers. With fan-in there is no metadata multicast to the
    // peers, so metadata_ready is now a pure "master done, you may ack" barrier.
    // The master sits inside the multicast bbox and a (non-loopback) NoC
    // multicast does not deliver to its own source, so num_dests is
    // num_peer_workers — otherwise the barrier would wait on the master's own
    // ack, which is never delivered.
    noc_semaphore_inc_multicast(metadata_ready_mcast, /*incr=*/1, /*num_dests=*/num_peer_workers);
    noc_async_atomic_barrier();

    noc_semaphore_inc(write_ack_noc, 1);
    noc_async_atomic_barrier();
}
