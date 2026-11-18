// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel that reads every page of the input through a TensorAccessor built with an explicit, possibly unaligned page
// size. Page i goes to slot i of an L1 buffer, which is first filled with 0xFF so that bytes past the page stay
// visible, and the slots are then written to a DRAM output buffer for host-side verification.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t input_page_size = get_arg_val<uint32_t>(2);
    uint32_t num_pages = get_arg_val<uint32_t>(3);
    uint32_t slot_size = get_arg_val<uint32_t>(4);

    constexpr uint32_t input_ta_cta_offset = 0;
    constexpr auto input_ta_args = TensorAccessorArgs<input_ta_cta_offset>();

    constexpr uint32_t output_ta_cta_offset = input_ta_args.next_compile_time_args_offset();
    constexpr auto output_ta_args = TensorAccessorArgs<output_ta_cta_offset>();

    auto input_accessor = TensorAccessor(input_ta_args, input_addr, input_page_size);
    auto output_accessor = TensorAccessor(output_ta_args, output_addr);

    constexpr uint32_t output_cb = 0;
    cb_reserve_back(output_cb, 1);
    uint32_t l1_addr = get_write_ptr(output_cb);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < num_pages * slot_size / sizeof(uint32_t); i++) {
        ptr[i] = 0xFFFFFFFF;
    }

    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {
        noc_async_read_page(page_id, input_accessor, l1_addr + page_id * slot_size);
    }
    noc_async_read_barrier();

    noc_async_write(l1_addr, output_accessor.get_noc_addr(0), num_pages * slot_size);
    noc_async_write_barrier();
}
