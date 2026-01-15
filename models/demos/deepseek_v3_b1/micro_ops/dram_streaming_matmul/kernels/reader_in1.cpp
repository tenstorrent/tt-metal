// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

/**
 * Simple in1 reader kernel for DRAM streaming.
 *
 * Reads weights (in1) from DRAM and bias if present.
 * Output CB (CB4) is backed directly by output tensor - no copy needed.
 */
void kernel_main() {
    // Compile time args - CB IDs first, then buffer addresses and sizes
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in3 = get_compile_time_arg_val(1);  // bias CB
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t in1_tensor_addr = get_compile_time_arg_val(3);
    constexpr uint32_t in3_tensor_addr = get_compile_time_arg_val(4);  // bias addr, 0 if no bias
    constexpr uint32_t in1_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t in1_num_pages = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(10);

#ifdef FUSE_BIAS
    constexpr uint32_t in3_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t in3_num_pages = get_compile_time_arg_val(12);
#endif

    // Runtime args (per-core values)
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    // Read in1 from DRAM with triple buffering
    uint32_t l1_write_addr_in1;
    uint32_t l1_read_addr_in1 = 0;

    uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_tensor_addr);
    noc_async_read_one_packet_set_state<true>(in1_base_addr, in1_page_size, vc);

    constexpr uint32_t total_num_blocks_in_buffer = 3;
    uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    cb_reserve_back(cb_id_in1, in1_block_num_tiles);
    uint32_t l1_write_addr_in1_offset = 0;
    uint32_t l1_write_addr_in1_start = get_write_ptr(cb_id_in1);
    l1_write_addr_in1 = l1_write_addr_in1_start;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        noc_async_read_set_trid(curr_block_trid);

        for (uint32_t h = 0; h < in1_num_pages; ++h) {
            noc_async_read_one_packet_with_state_with_trid(
                in1_base_addr, l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        if (num_free_blocks_in_buffer == 2) {
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            cb_push_back(cb_id_in1, in1_block_num_tiles);
            block_trid_to_wait = block_trid_to_wait == 3 ? 1 : (block_trid_to_wait + 1);
            cb_reserve_back(cb_id_in1, in1_block_num_tiles * 2);
        } else {
            num_free_blocks_in_buffer -= 1;
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

    // Wait for last block
    noc_async_read_barrier_with_trid(block_trid_to_wait);
    cb_push_back(cb_id_in1, in1_block_num_tiles);

#ifdef FUSE_BIAS
    // Read bias from DRAM
    cb_reserve_back(cb_id_in3, in1_block_w);
    uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3);
    uint32_t l1_read_addr_in3 = 0;

    uint64_t in3_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in3_tensor_addr);
    noc_async_read_one_packet_set_state<true>(in3_base_addr, in3_page_size, vc);

    for (uint32_t h = 0; h < in3_num_pages; ++h) {
        noc_async_read_one_packet_with_state<true, true>(in3_base_addr + l1_read_addr_in3, l1_write_addr_in3, vc);
        l1_read_addr_in3 += in3_page_size;
        l1_write_addr_in3 += in3_page_size;
    }

    noc_async_read_barrier();
    cb_push_back(cb_id_in3, in1_block_w);
#endif

    // Wait for compute to finish writing to CB4
    // CB4 is backed by output tensor - data goes directly there
    cb_wait_front(cb_id_out, out_block_num_tiles);
}
