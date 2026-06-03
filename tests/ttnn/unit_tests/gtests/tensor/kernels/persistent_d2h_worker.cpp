// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Worker kernel for D2HStreamService worker-sync.
//
// metadata_peer_enabled == 0 (no inline metadata):
//   transfer_done → write slice → write_ack on service core.
//
// metadata_peer_enabled == 1 (peer of master forwarder):
//   transfer_done → write slice → worker_done on master → wait metadata_ready → write_ack.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t metadata_peer_enabled = get_compile_time_arg_val(4);
constexpr uint32_t master_noc_x = get_compile_time_arg_val(5);
constexpr uint32_t master_noc_y = get_compile_time_arg_val(6);
constexpr uint32_t worker_done_counter_addr = get_compile_time_arg_val(7);
constexpr uint32_t metadata_ready_sem_addr = get_compile_time_arg_val(8);
constexpr auto acc_args = TensorAccessorArgs<9>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t fill_seed = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t write_ack_counter_addr = get_arg_val<uint32_t>(5);

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

    if constexpr (metadata_peer_enabled) {
        const uint64_t worker_done_noc = get_noc_addr(master_noc_x, master_noc_y, worker_done_counter_addr);
        noc_semaphore_inc(worker_done_noc, 1);
        noc_async_atomic_barrier();

        volatile tt_l1_ptr uint32_t* metadata_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_ready_sem_addr);
        while (true) {
            invalidate_l1_cache();
            if (*metadata_ready > 0) {
                *metadata_ready = 0;
                break;
            }
        }
    }

    noc_semaphore_inc(write_ack_noc, 1);
    noc_async_atomic_barrier();
}
