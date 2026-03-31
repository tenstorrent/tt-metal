// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simple binary reader: reads tiles from two source tensors into CB0 and CB1

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    const uint32_t page_bytes_0 = get_local_cb_interface(cb_id_in0).fifo_page_size;
    const uint32_t page_bytes_1 = get_local_cb_interface(cb_id_in1).fifo_page_size;

    constexpr uint32_t onepage = 1;

    const auto s0 = TensorAccessor(src0_args, src0_addr, page_bytes_0);
    const auto s1 = TensorAccessor(src1_args, src1_addr, page_bytes_1);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, onepage);
        uint32_t l1_write_addr_0 = get_write_ptr(cb_id_in0);
        noc_async_read_page(i, s0, l1_write_addr_0);

        cb_reserve_back(cb_id_in1, onepage);
        uint32_t l1_write_addr_1 = get_write_ptr(cb_id_in1);
        noc_async_read_page(i, s1, l1_write_addr_1);

        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onepage);
        cb_push_back(cb_id_in1, onepage);
    }
}
