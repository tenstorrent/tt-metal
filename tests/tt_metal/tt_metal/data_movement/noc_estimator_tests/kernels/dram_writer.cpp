// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRAM write kernel for NOC estimator tests.
// Uses TensorAccessor with experimental Noc 2.0 API.
// Supports interleaved (round-robin across banks) and sharded (fixed bank per core) modes.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"
#include "barrier_sync.hpp"
#include "log_helpers.hpp"

void kernel_main() {
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t test_id = get_compile_time_arg_val(2);
    constexpr uint32_t sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t memory_type = get_compile_time_arg_val(4);
    constexpr uint32_t mechanism = get_compile_time_arg_val(5);
    constexpr uint32_t pattern = get_compile_time_arg_val(6);
    constexpr uint32_t num_pages = get_compile_time_arg_val(7);

    // TensorAccessorArgs appended starting at compile arg index 8
    auto accessor_args = TensorAccessorArgs<8>();

    uint32_t dst_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_read_addr = get_arg_val<uint32_t>(1);
    uint32_t start_page = get_arg_val<uint32_t>(2);
    // Barrier args
    uint32_t barrier_sem_id = get_arg_val<uint32_t>(3);
    uint32_t barrier_coord_x = get_arg_val<uint32_t>(4);
    uint32_t barrier_coord_y = get_arg_val<uint32_t>(5);
    uint32_t num_cores = get_arg_val<uint32_t>(6);
    uint32_t local_scratch_addr = get_arg_val<uint32_t>(7);

    auto s = TensorAccessor(accessor_args, dst_buffer_addr, page_size_bytes);

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_ep;

    barrier_sync(barrier_sem_id, barrier_coord_x, barrier_coord_y, num_cores, local_scratch_addr);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            uint32_t page_id;
            if constexpr (memory_type == MEMORY_TYPE_DRAM_SHARDED) {
                page_id = start_page;
            } else {
                page_id = (start_page + i) % num_pages;
            }
            noc.async_write(unicast_ep, s, page_size_bytes, {.addr = l1_read_addr}, {.page_id = page_id});
        }
        noc.async_write_barrier();
    }

    log_estimator_metadata(
        test_id, noc.get_noc_id(), num_of_transactions, page_size_bytes, memory_type, mechanism, pattern, 0, 0, 0, 0);
}
