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

    constexpr uint32_t num_transfers = get_compile_time_arg_val(4);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t page_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_pages = get_compile_time_arg_val(7);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(9);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t input_start_idx = get_compile_time_arg_val(11);
    constexpr uint32_t output_start_idx = get_compile_time_arg_val(12);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(13);
    constexpr uint32_t output_start_offset = get_compile_time_arg_val(14);
    constexpr uint32_t out_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t row_offset = get_compile_time_arg_val(16);
    constexpr uint32_t num_rows = get_compile_time_arg_val(17);

    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_size};
    const InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_offset, .page_size = out_page_size};

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t receiver_semaphore_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, sem_addr);

    uint32_t buffer_addrs[2] = {buffer0_addr, buffer1_addr};
    uint64_t receiver_buffer_addrs[2] = {get_noc_addr(receiver_noc_x, receiver_noc_y, buffer_addrs[0]), get_noc_addr(receiver_noc_x, receiver_noc_y, buffer_addrs[1])};
    volatile tt_l1_ptr uint32_t * header_addrs[2] = {reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_addrs[0]), reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_addrs[1])};

    const auto& get_and_send_data = [&](uint32_t& input_page_idx, uint32_t& output_page_idx, uint32_t& row_idx, const uint32_t num_pages, const uint32_t num_bytes, const uint32_t num_bytes_per_send, const uint32_t num_bytes_per_send_word_size, uint32_t& curr_buffer_idx, uint32_t& next_buffer_idx) __attribute__((always_inline)) {
        uint32_t local_eth_l1_curr_src_addr = buffer_addrs[curr_buffer_idx] + 32;
        const uint32_t end_read_idx = input_page_idx + num_pages;
        for (; input_page_idx < end_read_idx; ++input_page_idx) {
            uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
            noc_async_read(src_noc_addr, local_eth_l1_curr_src_addr, page_size);
            local_eth_l1_curr_src_addr += page_size;
        }

        volatile tt_l1_ptr uint32_t * header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_addrs[curr_buffer_idx]);
        header[0] = output_page_idx;
        header[1] = output_start_offset;
        header[2] = row_idx;
        eth_noc_async_read_barrier();

        local_eth_l1_curr_src_addr = buffer_addrs[curr_buffer_idx] + 32;

        for (uint32_t i = 0; i < num_pages; ++i) {
            uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
            noc_async_write(local_eth_l1_curr_src_addr, dst_noc_addr, page_size);
            local_eth_l1_curr_src_addr += page_size;
            output_page_idx++;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
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

    uint32_t input_page_idx = input_start_idx;
    uint32_t output_page_idx = output_start_idx;
    uint32_t row_idx = row_start_idx;

    uint32_t curr_buffer_idx = 0, next_buffer_idx = 1;

    // How many chunks we split our local device data into
    for (uint32_t i = 0; i < num_full_chunks; ++i) {
        get_and_send_data(input_page_idx, output_page_idx, row_idx, num_pages, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size, curr_buffer_idx, next_buffer_idx);
    }

    if constexpr (rem_num_pages > 0) {
        // TODO: Are these necessary?
        constexpr uint32_t rem_num_bytes_per_send = rem_num_bytes;
        constexpr uint32_t rem_num_bytes_per_send_word_size = rem_num_bytes_per_send >> 4;
        get_and_send_data(input_page_idx, output_page_idx, row_idx, rem_num_pages, rem_num_bytes, rem_num_bytes_per_send, rem_num_bytes_per_send_word_size, curr_buffer_idx, next_buffer_idx);
    }
}
