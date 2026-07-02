// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Single worker kernel for D2HStreamService worker-sync.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(4);
constexpr uint32_t worker_metadata_l1_addr = get_compile_time_arg_val(5);
constexpr auto acc_args = TensorAccessorArgs<6>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t fill_seed = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t write_ack_counter_addr = get_arg_val<uint32_t>(5);
    const uint32_t is_master = get_arg_val<uint32_t>(6);
    const uint32_t metadata_input_addr = get_arg_val<uint32_t>(7);

    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);

    auto backing = TensorAccessor(acc_args, backing_tensor_addr);
    const uint32_t cb_l1 = scratch_cb.get_write_ptr();

    volatile tt_l1_ptr uint32_t* transfer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(transfer_done_sem_addr);

    while (true) {
        invalidate_l1_cache();
        if (*transfer_done > 0) {
            *transfer_done = 0;
            break;
        }
    }

    CoreLocalMem<uint32_t> scratch(cb_l1);
    for (uint32_t p = start_page; p < end_page; ++p) {
        for (uint32_t i = 0; i < page_size / sizeof(uint32_t); ++i) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1)[i] = fill_seed + p + i;
        }
        noc.async_write<NocOptions::DEFAULT, page_size>(scratch, backing, page_size, {}, {.page_id = p});
    }
    noc.async_write_barrier();

    if constexpr (metadata_size_bytes > 0) {
        if (is_master) {
            CoreLocalMem<uint32_t> metadata_src(worker_metadata_l1_addr);
            UnicastEndpoint service;
            noc.async_write(
                metadata_src,
                service,
                metadata_size_bytes,
                {},
                {.noc_x = service_noc_x, .noc_y = service_noc_y, .addr = metadata_input_addr});
            noc.async_write_barrier();
        }
    }

    // Device 2.0 migration: legacy primitive retained — write_ack_counter_addr is a raw service-core L1
    // counter address (allocated by the ServiceCoreManager), not a kernel semaphore id. Semaphore<> binds
    // to per-program ids via get_semaphore<>(id), so Semaphore::up cannot target this address.
    const uint64_t write_ack_noc = get_noc_addr(service_noc_x, service_noc_y, write_ack_counter_addr);
    noc_semaphore_inc(write_ack_noc, 1);
    noc.async_atomic_barrier();
}
