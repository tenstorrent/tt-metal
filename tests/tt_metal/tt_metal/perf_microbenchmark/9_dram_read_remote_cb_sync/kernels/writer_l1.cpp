// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

ALWI CBInterface setup_remote_cb_read_write_interface(uint32_t tt_l1_ptr *sync_cb_config) {

    uint32_t fifo_addr = sync_cb_config[0];
    uint32_t fifo_size = sync_cb_config[1];
    uint32_t fifo_num_pages = sync_cb_config[2];
    uint32_t fifo_page_size = sync_cb_config[3];
    uint32_t fifo_limit = fifo_addr + fifo_size;

    CBInterface cb_interface;

    cb_interface.fifo_limit = fifo_limit;  // to check if we need to wrap
    cb_interface.fifo_wr_ptr = fifo_addr;
    cb_interface.fifo_rd_ptr = fifo_addr;
    cb_interface.fifo_size = fifo_size;
    cb_interface.tiles_acked_received_init = 0;
    cb_interface.fifo_num_pages = fifo_num_pages;
    cb_interface.fifo_page_size = fifo_page_size;
    cb_interface.fifo_wr_tile_ptr = 0;

    return cb_interface;
}

inline void remote_cb_reserve_back(const uint32_t operand, const uint32_t num_tiles, const uint32_t fifo_num_pages, CBInterface& cb_interface) {
    // TODO(MO): Manually uncomment until issue #6619 is resolved
    //DeviceZoneScopedSumN2("CB-COMPUTE-RESERVE-BACK");
    std::uint32_t output = operand;

    volatile tt_reg_ptr std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    volatile tt_reg_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);

    uint32_t tiles_received = cb_interface.tiles_received;

    std::int32_t free_tiles;
    do {
        std::uint16_t tiles_acked = cb_interface.tiles_acked;
        std::uint32_t free_tiles_wrap = cb_interface.fifo_num_pages - (tiles_received - tiles_acked);
        free_tiles = (std::int32_t) free_tiles_wrap;
    } while (free_tiles < num_tiles);
}

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);

    const uint32_t noc_x = get_arg_val<uint32_t>(0);
    const uint32_t noc_y = get_arg_val<uint32_t>(1);

    uint32_t tt_l1_ptr *sync_cb_config = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    CBInterface sync_cb_interface = setup_remote_cb_read_write_interface(sync_cb_config);

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t sync_cb_id = 1;

    uint32_t l1_write_addr = get_write_ptr(cb_id);
    const uint64_t l1_noc_write_addr = get_noc_addr(noc_x, noc_y, l1_write_addr);

    noc_async_write_one_packet_set_state(l1_noc_write_addr, page_size);

    for (uint32_t block = 0; block < num_blocks; ++block) {

        remote_cb_reserve_back(cb_id, block_num_tiles);
        auto remote_l1_write_addr = l1_noc_write_addr;

        cb_wait_front(cb_id, block_num_tiles);
        auto l1_read_addr = get_read_ptr(cb_id);

        for (uint32_t h = 0; h < num_pages; ++h) {
            noc_async_write_one_packet_with_state(l1_read_addr, remote_l1_write_addr);
            l1_read_addr += page_size;
            remote_l1_write_addr += page_size;
        }

        noc_async_write_barrier();

        remote_cb_push_back(cb_id, block_num_tiles);

        cb_pop_front(cb_id, block_num_tiles);

    }


}
