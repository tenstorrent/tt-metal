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

inline void remote_cb_wait_front(const uint32_t operand, const uint32_t num_tiles, CBInterface& cb_interface) {

    volatile tt_l1_ptr std::uint32_t * tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    uint16_t num_tiles_recv;
    do {
        tiles_received = cb_interface.tiles_received;
        num_tiles_recv = tiles_received - cb_interface.tiles_acked;
    } while (num_tiles_recv < num_tiles_u);

}

inline void remote_cb_pop_front(
    const uint32_t operand, const uint32_t num_tiles, const uint32_t num_words, CBInterface& cb_interface) {

    cb_interface.tiles_acked += num_tiles;
    cb_interface.fifo_rd_ptr += num_words;

    if (cb_interface.fifo_rd_ptr >= cb_interface.fifo_limit) {
        cb_interface.fifo_rd_ptr -= cb_interface.fifo_size;
    }
}

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);

    const uint32_t noc_x = get_arg_val<uint32_t>(0);
    const uint32_t noc_y = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t sync_cb_id = 1;

    uint32_t tt_l1_ptr *sync_cb_config = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    CBInterface sync_cb_interface = setup_remote_cb_read_write_interface(sync_cb_config);

    for (uint32_t block = 0; block < num_blocks; ++block) {

        remote_cb_wait_front(cb_id, block_num_tiles, sync_cb_interface);
        auto l1_read_addr = get_read_ptr(cb_id);

        // do sth here

        remote_cb_pop_front(cb_id, block_num_tiles, page_size*num_pages, sync_cb_interface);
    }


}
