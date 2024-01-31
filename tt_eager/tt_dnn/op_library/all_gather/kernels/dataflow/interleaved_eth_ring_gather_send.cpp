// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t buffer0_addr = get_arg_val<uint32_t>(2);
    const uint32_t buffer1_addr = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t receiver_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_noc_y = get_compile_time_arg_val(1);

    constexpr bool src_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr DataFormat df = static_cast<DataFormat>(get_compile_time_arg_val(4));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(5);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t page_size = get_compile_time_arg_val(7);
    constexpr uint32_t num_pages = get_compile_time_arg_val(8);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(10);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t global_start_idx = get_compile_time_arg_val(12);
    constexpr uint32_t row_offset = get_compile_time_arg_val(13);
    constexpr uint32_t col_offset = get_compile_time_arg_val(14);
    constexpr uint32_t num_rows = get_compile_time_arg_val(15);
    constexpr uint32_t num_cols = get_compile_time_arg_val(16);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = page_size,
        .data_format = df
    };

    const InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = page_size,
        .data_format = df
    };

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t receiver_semaphore_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, sem_addr);

    uint32_t buffer_addrs[2] = {buffer0_addr, buffer1_addr};
    uint64_t receiver_buffer_addrs[2] = {get_noc_addr(receiver_noc_x, receiver_noc_y, buffer_addrs[0]), get_noc_addr(receiver_noc_x, receiver_noc_y, buffer_addrs[1])};

    const auto& get_and_send_data = [&](uint32_t& page_idx, uint32_t& curr_idx, uint32_t& row_idx, uint32_t& col_idx, const uint32_t num_pages, const uint32_t num_bytes, const uint32_t num_bytes_per_send, const uint32_t num_bytes_per_send_word_size, uint32_t& curr_buffer_idx, uint32_t& next_buffer_idx) __attribute__((always_inline)) {
        uint32_t local_eth_l1_curr_src_addr = buffer_addrs[curr_buffer_idx] + 32;
        const uint32_t end_read_idx = page_idx + num_pages;
        for (; page_idx < end_read_idx; ++page_idx) {
            noc_async_read_tile(page_idx, s, local_eth_l1_curr_src_addr);
            local_eth_l1_curr_src_addr += page_size;
        }

        local_eth_l1_curr_src_addr = buffer_addrs[curr_buffer_idx] + 32;

        volatile tt_l1_ptr uint32_t * start_idx_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_addrs[curr_buffer_idx]);
        start_idx_addr[0] = curr_idx;
        start_idx_addr[1] = col_idx;
        start_idx_addr[2] = row_idx;
        eth_noc_async_read_barrier();

        for (uint32_t i = 0; i < num_pages; ++i) {
            noc_async_write_tile(curr_idx, d, local_eth_l1_curr_src_addr);
            local_eth_l1_curr_src_addr += page_size;
            curr_idx++;
            col_idx++;
            if (col_idx == num_cols) {
                curr_idx += col_offset;
                col_idx = 0;
                row_idx++;
                if (row_idx == num_rows) {
                    row_idx = 0;
                    curr_idx += row_offset;
                }
            }
        }
        eth_send_bytes(buffer_addrs[curr_buffer_idx], buffer_addrs[curr_buffer_idx], num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
        eth_wait_for_remote_receiver_done_and_get_local_receiver_data<true>(
            sender_semaphore_addr_ptr,
            receiver_semaphore_noc_addr,
            receiver_buffer_addrs[curr_buffer_idx],
            buffer_addrs[next_buffer_idx],
            num_bytes
        );
        std::swap(curr_buffer_idx, next_buffer_idx);

        // num_transfers = num_devices - 1
        for (uint32_t i = 0; i < num_transfers - 1; ++i) {
            eth_send_bytes(buffer_addrs[curr_buffer_idx], buffer_addrs[curr_buffer_idx], num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
            eth_wait_for_remote_receiver_done_and_get_local_receiver_data<false>(
                sender_semaphore_addr_ptr,
                receiver_semaphore_noc_addr,
                receiver_buffer_addrs[curr_buffer_idx],
                buffer_addrs[next_buffer_idx],
                num_bytes
            );
            std::swap(curr_buffer_idx, next_buffer_idx);
        }

    };

    // TODO: Are these necessary?
    constexpr uint32_t num_bytes_per_send = num_bytes;
    constexpr uint32_t num_bytes_per_send_word_size = num_bytes_per_send >> 4;

    uint32_t page_idx = 0;
    uint32_t curr_idx = global_start_idx;
    uint32_t col_idx = 0;
    uint32_t row_idx = 0;

    uint32_t curr_buffer_idx = 0, next_buffer_idx = 1;
    // How many chunks we split our local device data into
    for (uint32_t i = 0; i < num_full_chunks; ++i) {
        get_and_send_data(page_idx, curr_idx, row_idx, col_idx, num_pages, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size, curr_buffer_idx, next_buffer_idx);
    }

    if constexpr (rem_num_pages > 0) {
        // TODO: Are these necessary?
        constexpr uint32_t rem_num_bytes_per_send = rem_num_bytes;
        constexpr uint32_t rem_num_bytes_per_send_word_size = rem_num_bytes_per_send >> 4;
        get_and_send_data(page_idx, curr_idx, row_idx, col_idx, rem_num_pages, rem_num_bytes, rem_num_bytes_per_send, rem_num_bytes_per_send_word_size, curr_buffer_idx, next_buffer_idx);
    }
}
