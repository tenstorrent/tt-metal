// SPDX-License-Identifier: Apache-2.0
// EXP2 reader (BRISC/NOC0): peak contiguous DRAM bank read (one_packet_with_state + triple-buffer TRID),
// pushing each block into cb0 for the NCRISC mcast to consume. cb_reserve_back provides backpressure from
// the mcast so we can detect the read funneling. block = num_pages packets of `page_bytes` each.
// use_cb=0 => decoupled: no cb backpressure (manual 3-slot ring), for isolating on-core read vs mcast.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);    // 16KB packets per block
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);   // 16384
    constexpr uint32_t block_tiles = get_compile_time_arg_val(3);  // tiles per block (for cb reserve/push)
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t use_cb = get_compile_time_arg_val(5);  // 0 = decoupled (no cb backpressure)

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);

    uint64_t src_base = get_noc_addr_from_bank_id<true>(bank_id, input_addr);
    noc_async_read_one_packet_set_state<true>(src_base, page_bytes, vc);
    uint32_t l1_read_addr = 0;  // running byte offset into the bank shard
    const uint32_t base_l1 = get_write_ptr(cb_id);
    constexpr uint32_t block_bytes = num_pages * page_bytes;

    constexpr uint32_t total_num_blocks_in_buffer = 3;
    uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        uint32_t l1_write_addr;
        if constexpr (use_cb) {
            cb_reserve_back(cb_id, block_tiles);
            l1_write_addr = get_write_ptr(cb_id);
        } else {
            l1_write_addr = base_l1 + (block % total_num_blocks_in_buffer) * block_bytes;
        }
        noc_async_read_set_trid(curr_block_trid);
        for (uint32_t h = 0; h < num_pages; ++h) {
            noc_async_read_one_packet_with_state_with_trid(src_base, l1_read_addr, l1_write_addr, curr_block_trid);
            l1_read_addr += page_bytes;
            l1_write_addr += page_bytes;
        }
        if (num_free_blocks_in_buffer == 2) {
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            block_trid_to_wait = block_trid_to_wait == 3 ? 1 : (block_trid_to_wait + 1);
        } else {
            num_free_blocks_in_buffer -= 1;
        }
        if constexpr (use_cb) {
            cb_push_back(cb_id, block_tiles);
        }
        curr_block_trid = curr_block_trid == total_num_blocks_in_buffer ? 1 : (curr_block_trid + 1);
    }
    noc_async_read_barrier_with_trid(block_trid_to_wait);
}
