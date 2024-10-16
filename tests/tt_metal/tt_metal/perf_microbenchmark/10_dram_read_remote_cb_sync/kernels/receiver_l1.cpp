// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

constexpr uint32_t ALIGNED_PAGE_SIZE = 16;

constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
constexpr uint32_t block_num_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_start_addr = get_compile_time_arg_val(2);
constexpr uint32_t cb_rd_ptr = get_compile_time_arg_val(2);
constexpr uint32_t cb_size = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);

uint32_t rt_args_idx = 0;
uint32_t vc;
uint32_t noc_x;
uint32_t noc_y;
uint32_t pages_acked_semaphore_addr;
uint32_t pages_sent_semaphore_addr;


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
    uint32_t num_pages = cb_size / page_size;
    uint32_t cb_size_page_aligned = num_pages * page_size;

    remote_cb_interface.fifo_size = cb_size;
    remote_cb_interface.fifo_limit = cb_size + cb_start_addr;
    remote_cb_interface.fifo_limit_page_aligned = cb_size_page_aligned + cb_start_addr;


    remote_cb_interface.fifo_page_size = page_size;
    remote_cb_interface.fifo_aligned_num_pages = num_pages * page_size / aligned_page_size;

    remote_cb_interface.fifo_rd_ptr = cb_rd_ptr;

    remote_cb_interface.fifo_start_addr = cb_start_addr;

    remote_cb_interface.pages_acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(pages_acked_semaphore_addr));
    remote_cb_interface.pages_sent = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(pages_sent_semaphore_addr));

    remote_cb_interface.aligned_page_size = aligned_page_size;
}

FORCE_INLINE void setup_remote_cb_page_size(uint32_t page_size) {
    uint32_t num_pages = remote_cb_interface.fifo_size / page_size;
    uint32_t cb_size_page_aligned = num_pages * page_size;

    remote_cb_interface.fifo_aligned_num_pages = num_pages * page_size / remote_cb_interface.aligned_page_size;
    remote_cb_interface.fifo_limit_page_aligned = cb_size_page_aligned + remote_cb_interface.fifo_start_addr;
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

    *remote_cb_interface.pages_acked += num_aligned_pages;
    remote_cb_interface.fifo_rd_ptr += len_bytes;

    if ((remote_cb_interface.fifo_rd_ptr + len_bytes) >= remote_cb_interface.fifo_limit_page_aligned) {
        remote_cb_interface.fifo_rd_ptr = remote_cb_interface.fifo_start_addr;
    }

    uint64_t remote_ack_ptr_addr = get_noc_addr(remote_noc_x, remote_noc_y, (uint32_t)remote_cb_interface.pages_acked, noc);
    noc_semaphore_inc(remote_ack_ptr_addr, num_aligned_pages, noc);
}


void kernel_main() {

    uint32_t rt_args_idx = 0;
    vc = get_arg_val<uint32_t>(rt_args_idx++);
    noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    pages_acked_semaphore_addr = get_arg_val<uint32_t>(rt_args_idx++);
    pages_sent_semaphore_addr = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t cb_id = 0;

    setup_remote_receiver_cb_interface<ALIGNED_PAGE_SIZE>();

    for (uint32_t block = 0; block < num_blocks; ++block) {
        remote_cb_wait_front(block_num_tiles);
        remote_cb_pop_front(block_num_tiles, noc_x, noc_y);
    }

}
