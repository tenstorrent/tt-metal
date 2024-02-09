// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t buffer0_addr = get_arg_val<uint32_t>(1);
    const uint32_t buffer1_addr = get_arg_val<uint32_t>(2);
    const uint32_t sem_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t sender_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t sender_noc_y = get_compile_time_arg_val(1);

    constexpr bool dst_is_dram = get_compile_time_arg_val(2) == 1;

    constexpr DataFormat df = static_cast<DataFormat>(get_compile_time_arg_val(3));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(4);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t page_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_pages = get_compile_time_arg_val(7);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(9);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t row_offset = get_compile_time_arg_val(11);
    constexpr uint32_t col_offset = get_compile_time_arg_val(12);
    constexpr uint32_t num_rows = get_compile_time_arg_val(13);
    constexpr uint32_t num_cols = get_compile_time_arg_val(14);

    const InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = page_size,
        .data_format = df
    };

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sem_addr);

    uint32_t buffer_addrs[2] = {buffer0_addr, buffer1_addr};

    const auto& get_and_sync_data = [&](const uint32_t num_pages, const uint32_t num_bytes, uint32_t& curr_buffer_idx, uint32_t& next_buffer_idx) __attribute__((always_inline)) {
        for (uint32_t i = 0; i < num_transfers; ++i) {
            uint32_t local_eth_l1_curr_src_addr = buffer_addrs[curr_buffer_idx] + 32;

            eth_wait_for_bytes(num_bytes);
            noc_semaphore_inc(sender_semaphore_noc_addr, 1);
            volatile tt_l1_ptr uint32_t * header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_addrs[curr_buffer_idx]);

            uint32_t output_page_idx = header[0];
            uint32_t col_idx = header[1];
            uint32_t row_idx = header[2];

            for (uint32_t i = 0; i < num_pages; ++i) {
                noc_async_write_tile(output_page_idx, d, local_eth_l1_curr_src_addr);
                local_eth_l1_curr_src_addr += page_size;
                output_page_idx++;
                col_idx++;
                if (col_idx == num_cols) {
                    output_page_idx += col_offset;
                    col_idx = 0;
                    row_idx++;
                    if (row_idx == num_rows) {
                        row_idx = 0;
                        output_page_idx += row_offset;
                    }
                }
            }
            std::swap(curr_buffer_idx, next_buffer_idx);
            eth_noc_async_write_barrier();
            eth_noc_semaphore_wait(receiver_semaphore_addr_ptr, 1);
            noc_semaphore_set(receiver_semaphore_addr_ptr, 0);
            eth_receiver_done();
        }
    };

    uint32_t curr_buffer_idx = 0, next_buffer_idx = 1;
    for (uint32_t i = 0; i < num_full_chunks; ++i) {
        get_and_sync_data(num_pages, num_bytes, curr_buffer_idx, next_buffer_idx);
    }
    if constexpr (rem_num_pages > 0) {
        get_and_sync_data(rem_num_pages, rem_num_bytes, curr_buffer_idx, next_buffer_idx);
    }
}
