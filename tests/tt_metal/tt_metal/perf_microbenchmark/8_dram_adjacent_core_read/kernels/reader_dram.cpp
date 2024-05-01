// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

template <uint32_t bank_base_address, uint32_t page_size, bool use_vc>
FORCE_INLINE
void noc_async_read_tile_dram_sharded(uint32_t src_addr, uint32_t dest_addr, uint32_t bank_id = 0, const uint32_t vc = 0) {
    uint32_t src_addr_;
    uint32_t src_noc_xy;

    src_addr_ = src_addr + bank_base_address;
    src_addr_ += bank_to_dram_offset[bank_id];
    src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];

    DEBUG_STATUS('N', 'R', 'T', 'W');
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(get_noc_addr_helper(src_noc_xy, src_addr_), dest_addr, page_size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS('N', 'R', 'T', 'D');

    if constexpr(use_vc) {
        uint32_t noc_rd_cmd_field = NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    }

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr_);      // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy);   // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, page_size);  // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc_index] += 1;
}

void kernel_main() {
    constexpr uint32_t input_addr = get_compile_time_arg_val(0);
    constexpr uint32_t input_start_tile_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t block_w = get_compile_time_arg_val(3);
    constexpr uint32_t block_h = get_compile_time_arg_val(4);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t page_size = get_compile_time_arg_val(6);

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;

    cb_reserve_back(cb_id, block_num_tiles);
    uint32_t l1_read_addr = 0;
    for (uint32_t block = 0; block < num_blocks; ++block) {

        uint32_t l1_write_addr = get_write_ptr(cb_id);


        for(uint32_t h = 0; h < block_h; ++h) {

            noc_async_read_tile_dram_sharded<input_addr, page_size, true>(l1_read_addr, l1_write_addr, bank_id, vc);
            l1_read_addr += page_size;
            l1_write_addr += page_size;

        }

        noc_async_read_barrier();

    }
}
