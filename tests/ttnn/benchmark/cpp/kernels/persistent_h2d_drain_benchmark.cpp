// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(3);
constexpr auto input_acc_args = TensorAccessorArgs<4>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t total_iters = get_arg_val<uint32_t>(5);

    auto input = TensorAccessor(input_acc_args, input_tensor_addr);
    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* data_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    const uint64_t consumed_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);

    for (uint32_t iter = 0; iter < total_iters; ++iter) {
        while (true) {
            invalidate_l1_cache();
            if (*data_ready_sem > 0) {
                *data_ready_sem = 0;
                break;
            }
        }

        for (uint32_t page = start_page; page < end_page; ++page) {
            noc_async_read(input.get_noc_addr(page), cb_l1, page_size);
            noc_async_read_barrier();
        }

        noc_semaphore_inc(consumed_noc, 1);
        noc_async_atomic_barrier();
    }
}
