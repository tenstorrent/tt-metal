// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(2);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_pages = get_compile_time_arg_val(6);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(7);
    constexpr uint32_t input_start_idx = get_compile_time_arg_val(8);
    constexpr uint32_t output_start_idx = get_compile_time_arg_val(9);
    constexpr uint32_t output_start_addr_offset = get_compile_time_arg_val(10);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(11);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(12);
    constexpr uint32_t row_offset = get_compile_time_arg_val(13);
    constexpr uint32_t col_offset = get_compile_time_arg_val(14);
    constexpr uint32_t num_rows = get_compile_time_arg_val(15);
    constexpr uint32_t num_cols = get_compile_time_arg_val(16);
    constexpr uint32_t last_output_page_offset = get_compile_time_arg_val(17);
    constexpr uint32_t output_page_offset = get_compile_time_arg_val(18);
    constexpr uint32_t last_output_addr_offset = get_compile_time_arg_val(19);
    constexpr uint32_t output_addr_offset = get_compile_time_arg_val(20);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(21);
    constexpr uint32_t sem_addr = get_compile_time_arg_val(22);
    constexpr bool is_clockwise_direction = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(24);
    static_assert(half_cb_n_pages > rem_num_pages, "half_cb_n_pages must be greater than or equal to rem_num_pages");

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    #ifdef RM_INTERLEAVED
    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_size};
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
    #elif defined TILE_INTERLEAVED
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = page_size,
        .data_format = in0_df
    };

    InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = output_page_size,
        .data_format = in0_df
    };
    #endif
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t input_page_idx = input_start_idx;
    uint32_t output_base_page_idx = output_start_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;

    if constexpr(num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            read_chunk_from_input_tensor(input_page_idx, cb_id_in0, s, num_pages, page_size);
        }
    }
    if constexpr(rem_num_pages > 0) {
        read_chunk_from_input_tensor(input_page_idx, cb_id_in0, s, rem_num_pages, page_size);
        ASSERT(num_pages == 0 || num_pages > rem_num_pages);
        ASSERT(half_cb_n_pages > rem_num_pages);
        push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
    }

    uint32_t sem_idx = 1;

    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
        if (is_clockwise_direction) {
            if (input_ring_idx == 0) {
                input_ring_idx = num_transfers;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address += last_output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx += last_output_page_offset;
                }
            } else {
                input_ring_idx--;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address -= output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx -= output_page_offset;
                }
            }
        } else {
            if (input_ring_idx == num_transfers) {//0) {
                input_ring_idx = 0;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address -= last_output_addr_offset;
                    // d.bank_base_address = last_output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx -= last_output_page_offset;
                    // output_base_page_idx = last_output_page_offset;
                }
            } else {
                input_ring_idx++;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address += output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx += output_page_offset;
                }
            }
        }
        output_page_idx = output_base_page_idx;
        col_idx = col_start_idx;
        row_idx = row_start_idx;
        if constexpr(num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
                sem_idx++;
                read_chunk_from_output_tensor(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
            }
        }
        if constexpr(rem_num_pages > 0) {
            noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
            sem_idx++;
            read_chunk_from_output_tensor(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }
    }

}
