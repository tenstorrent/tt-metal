// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    constexpr uint32_t in1_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_pages = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_k = get_compile_time_arg_val(5);  // tiles per K subblock
    constexpr uint32_t per_core_N = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t out_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_subblocks_k = get_compile_time_arg_val(9);

    constexpr uint32_t num_iterations = num_subblocks_k * per_core_N;

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
    // Use 3 * num_subblocks_k buffers - must stay within NOC_MAX_TRANSACTION_ID (0xF = 15)
    // Assert on host side ensures num_buffers <= 15
    constexpr uint32_t num_buffers = 3 * num_subblocks_k;
    constexpr uint32_t extra_blocks_in_flight = 2;
    uint32_t num_free_blocks_in_buffer = num_buffers;
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    cb_reserve_back(cb_id_in1, num_free_blocks_in_buffer);
    uint32_t l1_write_addr_in1_offset = 0;
    uint32_t l1_write_addr_in1_start = get_write_ptr(cb_id_in1);
    l1_write_addr_in1 = l1_write_addr_in1_start;

    // Read in1: for each N column, read num_subblocks_k K subblocks
    for (uint32_t n = 0; n < num_iterations; ++n) {
        // Set transaction ID for this block's reads
        noc_async_read_set_trid(curr_block_trid);

        // Read pages for this K subblock
        for (uint32_t p = 0; p < in1_num_pages; p++) {
            noc_async_read_one_packet_with_state_with_trid(
                in1_base_addr, l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        // When down to 1 free buffer (1 extra in flight), wait for oldest and push it
        if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            cb_push_back(cb_id_in1, subblock_k);
            block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
            // Reserve 2 blocks for next iterations
            cb_reserve_back(cb_id_in1, subblock_k * (extra_blocks_in_flight + 1));
        } else {
            num_free_blocks_in_buffer -= 1;
        }

        // Advance write pointer and transaction ID (circular within num_buffers)
        if (curr_block_trid == num_buffers) {
            l1_write_addr_in1_offset = 0;
            curr_block_trid = 1;
        } else {
            l1_write_addr_in1_offset += in1_block_size_bytes;
            curr_block_trid += 1;
        }
        l1_write_addr_in1 = l1_write_addr_in1_start + l1_write_addr_in1_offset;
    }

    // Push the remaining blocks (extra_blocks_in_flight blocks still pending)
    for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
        noc_async_read_barrier_with_trid(block_trid_to_wait);
        cb_push_back(cb_id_in1, subblock_k);
        block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
    }

    // Wait for compute to finish writing all output tiles
    // CB4 is backed by output tensor - data goes directly there
    cb_wait_front(cb_id_out, out_num_tiles);
}
