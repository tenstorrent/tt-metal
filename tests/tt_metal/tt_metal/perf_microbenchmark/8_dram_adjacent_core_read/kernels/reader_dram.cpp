// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

template <uint32_t bank_base_address, uint32_t page_size, bool use_vc>
FORCE_INLINE void noc_async_read_tile_dram_sharded(
    uint32_t src_addr, uint32_t dest_addr, uint32_t bank_id = 0, const uint32_t vc = 0) {
    uint32_t src_addr_;
    uint32_t src_noc_xy;

    src_addr_ = src_addr + bank_base_address;
    src_addr_ += bank_to_dram_offset[bank_id];
    src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];

    WAYPOINT("NRTW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_index, get_noc_addr_helper(src_noc_xy, src_addr_), dest_addr, page_size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    WAYPOINT("NRTD");

    if constexpr (use_vc) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    }

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr_);           // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, page_size);              // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc_index] += 1;
}

void kernel_main() {
    constexpr uint32_t input_addr = get_compile_time_arg_val(0);
    constexpr uint32_t input_start_tile_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;

    // uint32_t src_base_addr = noc_async_read_tile_dram_sharded_set_state<true>(input_addr, page_size, bank_id, vc);
    uint64_t src_base_addr = get_noc_addr_from_bank_id<true>(bank_id, input_addr);
    noc_async_read_one_packet_set_state<true>(src_base_addr, page_size, vc);
    uint32_t l1_read_addr = 0;

    constexpr uint32_t total_num_blocks_in_buffer = 3;
    constexpr uint32_t total_num_trid = 4;
    uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    cb_reserve_back(cb_id, block_num_tiles);
    for (uint32_t block = 0; block < num_blocks; ++block) {
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        noc_async_read_tile_dram_sharded_set_trid(curr_block_trid);

        for (uint32_t h = 0; h < num_pages; ++h) {
            // noc_async_read_tile_dram_sharded_with_state_with_trid(
            //     src_base_addr, l1_read_addr, l1_write_addr, curr_block_trid);
            noc_async_read_tile_dram_sharded_with_state_with_trid(
                src_base_addr, l1_read_addr, l1_write_addr, curr_block_trid);
            l1_read_addr += page_size;
            l1_write_addr += page_size;
        }

        if (num_free_blocks_in_buffer == 2) {
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            // wait for next block trid
            block_trid_to_wait = block_trid_to_wait == 3 ? 1 : (block_trid_to_wait + 1);
        } else {
            num_free_blocks_in_buffer -= 1;
        }

        if (curr_block_trid == total_num_blocks_in_buffer) {
            curr_block_trid = 1;
        } else {
            curr_block_trid += 1;
        }
    }
    // last block to wait
    noc_async_read_barrier_with_trid(block_trid_to_wait);
}
