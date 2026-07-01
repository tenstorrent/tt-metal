// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Worker-side kernel for the H2DStreamService worker-sync handshake test.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(4);
// Metadata copy block (indices 5..8). Unused when metadata_enabled is 0.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(5);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(6);
constexpr uint32_t metadata_input_addr = get_compile_time_arg_val(7);
constexpr uint32_t metadata_output_addr = get_compile_time_arg_val(8);
constexpr auto acc_args = TensorAccessorArgs<9>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);

    auto input = TensorAccessor(acc_args, input_tensor_addr);
    auto output = TensorAccessor(acc_args, output_tensor_addr);
    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    while (true) {
        invalidate_l1_cache();
        if (*sem > 0) {
            *sem = 0;
            break;
        }
    }

    for (uint32_t p = start_page; p < end_page; ++p) {
        noc_async_read(input.get_noc_addr(p), cb_l1, page_size);
        noc_async_read_barrier();
        noc_async_write<page_size>(cb_l1, output.get_noc_addr(p), page_size);
    }
    noc_async_write_barrier();

    if constexpr (metadata_enabled) {
        auto* src = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(metadata_input_addr);
        auto* dst = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(metadata_output_addr);
        for (uint32_t i = 0; i < metadata_size_bytes; ++i) {
            dst[i] = src[i];
        }
    }

    const uint64_t consumed_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);
    noc_semaphore_inc(consumed_noc, 1);
    noc_async_atomic_barrier();
}
