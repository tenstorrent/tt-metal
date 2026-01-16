// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

/**
 * Simplified in1 reader for DRAM streaming matmul.
 *
 * Reads in1 from DRAM one K subblock at a time (subblock_k tiles).
 * in1 is pre-shuffled on host so K tiles are contiguous for each N column.
 *
 * Uses transaction IDs for pipelining DRAM reads (triple buffering).
 * Reads are broken into pages that fit within NOC max burst size.
 * Output CB is tensor-backed - just waits for compute to finish.
 */
void kernel_main() {
    // Compile time args
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_addr = get_compile_time_arg_val(2);
    constexpr uint32_t in1_page_size = get_compile_time_arg_val(3);  // Full page size (16KB aligned)
    constexpr uint32_t in1_num_pages = get_compile_time_arg_val(4);  // Total pages including last
    constexpr uint32_t subblock_k = get_compile_time_arg_val(5);     // tiles per K subblock
    constexpr uint32_t per_core_N = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t out_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_subblocks_k = get_compile_time_arg_val(9);
    constexpr uint32_t in1_last_page_size = get_compile_time_arg_val(10);  // 0 if all pages are full size

    constexpr uint32_t num_iterations = num_subblocks_k * per_core_N;
    // Number of full-size pages (excluding potential partial last page)
    constexpr uint32_t in1_num_full_pages = (in1_last_page_size == 0) ? in1_num_pages : (in1_num_pages - 1);

    // Runtime args (per-core values)
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);

    // Setup DRAM read for in1
    uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_tensor_addr);
    uint32_t l1_write_addr_in1;
    uint32_t l1_read_addr_in1 = 0;

    // Set up NOC state for page reads
    noc_async_read_one_packet_set_state<true>(in1_base_addr, in1_page_size, vc);

    // Multi-buffering with transaction IDs for pipelining
    // Buffer 3 * num_subblocks_k blocks so compute doesn't block reader for full K dimension
    constexpr uint32_t total_num_blocks_in_buffer = 3 * num_subblocks_k;
    // Max blocks in flight before we need to wait (leave 1 slot free for flexibility)
    constexpr uint32_t max_blocks_in_flight = 2;
    uint32_t num_blocks_in_flight = 0;  // Blocks read but not yet pushed
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    cb_reserve_back(cb_id_in1, subblock_k * total_num_blocks_in_buffer);
    uint32_t l1_write_addr_in1_offset = 0;
    uint32_t l1_write_addr_in1_start = get_write_ptr(cb_id_in1);
    l1_write_addr_in1 = l1_write_addr_in1_start;

    // Read in1: for each N column, read num_subblocks_k K subblocks
    for (uint32_t n = 0; n < num_iterations; ++n) {
        noc_async_read_set_trid(curr_block_trid);

        // Read full-size pages for this K subblock
        for (uint32_t p = 0; p < in1_num_full_pages; p++) {
            noc_async_read_one_packet_with_state_with_trid(
                in1_base_addr, l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        // Read last page if it's a partial page (smaller than full page size)
        if constexpr (in1_last_page_size > 0) {
            // Set state for last page size
            noc_async_read_one_packet_set_state<true>(in1_base_addr, in1_last_page_size, vc);
            noc_async_read_one_packet_with_state_with_trid(
                in1_base_addr, l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
            l1_read_addr_in1 += in1_last_page_size;
            l1_write_addr_in1 += in1_last_page_size;
            // Restore state for full page size
            noc_async_read_one_packet_set_state<true>(in1_base_addr, in1_page_size, vc);
        }

        num_blocks_in_flight += 1;

        // When we reach max blocks in flight, wait for oldest block and push it
        if (num_blocks_in_flight == max_blocks_in_flight) {
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            cb_push_back(cb_id_in1, subblock_k);
            block_trid_to_wait = (block_trid_to_wait == total_num_blocks_in_buffer) ? 1 : (block_trid_to_wait + 1);
            num_blocks_in_flight -= 1;
        }

        if (curr_block_trid == total_num_blocks_in_buffer) {
            l1_write_addr_in1_offset = 0;
            curr_block_trid = 1;
        } else {
            l1_write_addr_in1_offset += in1_block_size_bytes;
            curr_block_trid += 1;
        }
        l1_write_addr_in1 = l1_write_addr_in1_start + l1_write_addr_in1_offset;
    }

    // Push all remaining blocks in flight
    while (num_blocks_in_flight > 0) {
        noc_async_read_barrier_with_trid(block_trid_to_wait);
        cb_push_back(cb_id_in1, subblock_k);
        block_trid_to_wait = (block_trid_to_wait == total_num_blocks_in_buffer) ? 1 : (block_trid_to_wait + 1);
        num_blocks_in_flight -= 1;
    }

    // Wait for compute to finish writing all output tiles
    // CB4 is backed by output tensor - data goes directly there
    cb_wait_front(cb_id_out, out_num_tiles);
}
