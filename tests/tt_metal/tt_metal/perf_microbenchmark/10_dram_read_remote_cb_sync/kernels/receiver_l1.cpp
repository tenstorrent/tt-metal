// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "debug/dprint.h"

constexpr uint32_t ALIGNED_PAGE_SIZE = 16;

constexpr uint32_t cb_start_addr = get_compile_time_arg_val(0);
constexpr uint32_t cb_rd_ptr = get_compile_time_arg_val(0);
constexpr uint32_t cb_size = get_compile_time_arg_val(1);
constexpr uint32_t num_layers = get_compile_time_arg_val(2);

uint32_t rt_args_idx = 0;
uint32_t vc;
uint32_t noc_x;
uint32_t noc_y;
uint32_t pages_acked_semaphore_addr;
uint32_t pages_sent_semaphore_addr;
tt_l1_ptr uint32_t* page_size;
tt_l1_ptr uint32_t* num_blocks;
tt_l1_ptr uint32_t* block_num_tiles;

uint32_t start_page_size;

struct RemoteReceiverCBInterface {
    volatile tt_l1_ptr uint32_t* pages_acked;
    volatile tt_l1_ptr uint32_t* pages_sent;

    uint32_t fifo_size;
    uint32_t fifo_limit;
    uint32_t fifo_limit_page_aligned;

    uint32_t fifo_page_size;
    uint32_t fifo_aligned_num_pages;

    uint32_t fifo_rd_ptr;

    uint32_t fifo_start_addr;

    uint32_t aligned_page_size;
};

RemoteReceiverCBInterface remote_cb_interface;

template<uint32_t aligned_page_size>
FORCE_INLINE void setup_remote_receiver_cb_interface() {
    uint32_t num_pages = cb_size / start_page_size;
    uint32_t cb_size_page_aligned = num_pages * start_page_size;

    remote_cb_interface.fifo_size = cb_size;
    remote_cb_interface.fifo_limit = cb_size + cb_start_addr;
    remote_cb_interface.fifo_limit_page_aligned = cb_size_page_aligned + cb_start_addr;


    remote_cb_interface.fifo_page_size = start_page_size;
    remote_cb_interface.fifo_aligned_num_pages = num_pages * start_page_size / aligned_page_size;

    remote_cb_interface.fifo_rd_ptr = cb_rd_ptr;

    remote_cb_interface.fifo_start_addr = cb_start_addr;

    remote_cb_interface.pages_acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(pages_acked_semaphore_addr));
    remote_cb_interface.pages_sent = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(pages_sent_semaphore_addr));

    remote_cb_interface.aligned_page_size = aligned_page_size;
}

FORCE_INLINE void setup_remote_cb_page_size(uint32_t page_size) {
    uint32_t num_pages = remote_cb_interface.fifo_size / page_size;
    uint32_t cb_size_page_aligned = num_pages * page_size;

    remote_cb_interface.fifo_limit_page_aligned = cb_size_page_aligned + remote_cb_interface.fifo_start_addr;
    remote_cb_interface.fifo_page_size = page_size;
    remote_cb_interface.fifo_aligned_num_pages = num_pages * page_size / remote_cb_interface.aligned_page_size;

    uint32_t curr_fifo_rd_ptr = remote_cb_interface.fifo_rd_ptr;
    bool fifo_rd_ptr_exceed_fifo_limit = curr_fifo_rd_ptr > remote_cb_interface.fifo_limit_page_aligned;
    uint32_t num_pages_till_fifo_limit = (remote_cb_interface.fifo_limit_page_aligned - curr_fifo_rd_ptr) / page_size;
    remote_cb_interface.fifo_rd_ptr = fifo_rd_ptr_exceed_fifo_limit ?
        remote_cb_interface.fifo_start_addr : remote_cb_interface.fifo_limit_page_aligned - num_pages_till_fifo_limit * page_size;
}

FORCE_INLINE void remote_cb_wait_front(uint32_t num_pages) {
    uint32_t len_bytes = num_pages * remote_cb_interface.fifo_page_size;
    uint32_t num_pages_wait = len_bytes / remote_cb_interface.aligned_page_size;
    volatile uint32_t num_pages_recv = 0;
    uint32_t pages_acked = 0;
    uint32_t pages_sent = 0;

    do {

        pages_acked = (uint32_t)reg_read((uint32_t)remote_cb_interface.pages_acked);
        pages_sent = (uint32_t)reg_read((uint32_t)remote_cb_interface.pages_sent);
        num_pages_recv = pages_sent - pages_acked;
    } while (num_pages_recv < num_pages_wait);
}

FORCE_INLINE void remote_cb_pop_front(uint32_t num_pages, uint32_t remote_noc_x, uint32_t remote_noc_y, uint8_t noc = noc_index) {
    uint32_t len_bytes = num_pages * remote_cb_interface.fifo_page_size;
    uint32_t num_aligned_pages = len_bytes / remote_cb_interface.aligned_page_size;

    // DPRINT << "remote_cb_interface.fifo_rd_ptr " << remote_cb_interface.fifo_rd_ptr - remote_cb_interface.fifo_start_addr <<ENDL();
    // DPRINT << "remote_cb_interface.fifo_limit_page_aligned " << remote_cb_interface.fifo_limit_page_aligned <<ENDL();
    // DPRINT << "len_bytes " << len_bytes <<ENDL();

    *remote_cb_interface.pages_acked += num_aligned_pages;
    remote_cb_interface.fifo_rd_ptr += len_bytes;

    if (remote_cb_interface.fifo_rd_ptr >= remote_cb_interface.fifo_limit_page_aligned) {
        remote_cb_interface.fifo_rd_ptr = remote_cb_interface.fifo_start_addr;
    }

    uint64_t remote_ack_ptr_addr = get_noc_addr(remote_noc_x, remote_noc_y, (uint32_t)remote_cb_interface.pages_acked, noc);
    noc_semaphore_inc(remote_ack_ptr_addr, num_aligned_pages, noc);

    // DPRINT << "remote_cb_interface.fifo_start_addr " << remote_cb_interface.fifo_start_addr <<ENDL();
}


void kernel_main() {

    uint32_t rt_args_idx = 0;
    vc = get_arg_val<uint32_t>(rt_args_idx++);
    noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    pages_acked_semaphore_addr = get_arg_val<uint32_t>(rt_args_idx++);
    pages_sent_semaphore_addr = get_arg_val<uint32_t>(rt_args_idx++);

    page_size = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    num_blocks = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    block_num_tiles = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));

    start_page_size = page_size[0];

    constexpr uint32_t cb_id = 0;

    setup_remote_receiver_cb_interface<ALIGNED_PAGE_SIZE>();

    // DPRINT << "start" <<ENDL();

    for (uint32_t l = 0; l < num_layers; ++l) {
        uint32_t curr_page_size = page_size[l];
        uint32_t curr_num_blocks = num_blocks[l];
        uint32_t curr_block_num_tiles = block_num_tiles[l];

        DPRINT << "curr_num_blocks "  << curr_num_blocks << ENDL();
        // DPRINT << "curr_block_num_tiles " <<  curr_block_num_tiles<< ENDL();
        // DPRINT << "curr_page_size " <<  curr_page_size<< ENDL();

        setup_remote_cb_page_size(curr_page_size);

        for (uint32_t block = 0; block < curr_num_blocks; ++block) {
            remote_cb_wait_front(curr_block_num_tiles);

            // if (l == 2)
            //     DPRINT  << TSLICE(cb_id, 0, SliceRange{ .h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }, true) << ENDL();

            remote_cb_pop_front(curr_block_num_tiles, noc_x, noc_y);
        }
    }

    // DPRINT << "done" <<ENDL();

}
