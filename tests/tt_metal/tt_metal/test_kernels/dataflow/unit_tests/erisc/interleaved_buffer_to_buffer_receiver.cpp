// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t page_size = get_arg_val<uint32_t>(1);
    uint32_t max_buffer_size = get_arg_val<uint32_t>(2);
    uint32_t num_loops = get_arg_val<uint32_t>(3);
    uint32_t pages_per_loop = get_arg_val<uint32_t>(4);
    uint32_t remaining_bytes = get_arg_val<uint32_t>(5);
    uint32_t remaining_pages = get_arg_val<uint32_t>(6);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(dst_args, dst_addr, page_size);
    uint32_t elements_per_page = page_size / sizeof(std::uint32_t);

    experimental::Noc noc;

    uint32_t page_idx = 0;
    for (uint32_t i = 0; i < num_loops; ++i) {
        experimental::CoreLocalMem<std::uint32_t> src_l1(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);
        eth_wait_for_bytes(max_buffer_size);

        for (uint32_t j = 0; j < pages_per_loop; ++j) {
            noc.async_write(src_l1, s, page_size, {}, {.page_id = page_idx});

            page_idx++;
            src_l1 += elements_per_page;
            noc.async_write_barrier();
        }
        eth_receiver_done();
    }
    if (remaining_bytes > 0) {
        experimental::CoreLocalMem<std::uint32_t> src_l1(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);
        eth_wait_for_bytes(remaining_bytes);
        for (uint32_t j = 0; j < remaining_pages; ++j) {
            noc.async_write(src_l1, s, page_size, {}, {.page_id = page_idx});
            page_idx++;
            src_l1 += elements_per_page;
            noc.async_write_barrier();
        }
        eth_receiver_done();
    }
}
