// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "debug/dprint.h"

constexpr uint32_t ALIGNED_PAGE_SIZE = 16;

constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
constexpr uint32_t block_num_tiles = get_compile_time_arg_val(1);
constexpr uint32_t noc = get_compile_time_arg_val(2);
constexpr uint32_t cb_start_addr = get_compile_time_arg_val(3);
constexpr uint32_t cb_wr_ptr = get_compile_time_arg_val(3);
constexpr uint32_t cb_size = get_compile_time_arg_val(4);
constexpr uint32_t coalesced_num_pages = get_compile_time_arg_val(5);
constexpr uint32_t coalesced_page_size = get_compile_time_arg_val(6);
constexpr uint32_t page_size = get_compile_time_arg_val(7);
constexpr uint32_t num_receivers = get_compile_time_arg_val(8);
constexpr uint32_t num_tile_rows = get_compile_time_arg_val(9);
constexpr uint32_t receiver_block_num_tiles = get_compile_time_arg_val(10);

tt_l1_ptr uint32_t* noc_x;
tt_l1_ptr uint32_t* noc_y;
tt_l1_ptr uint32_t* pages_acked_semaphore_addr;
tt_l1_ptr uint32_t* pages_sent_semaphore_addr;

template<uint32_t num_recv_cbs>
struct RemoteSenderCBInterface {
    uint32_t num_receivers;

    volatile tt_l1_ptr uint32_t* pages_acked[num_recv_cbs];
    volatile tt_l1_ptr uint32_t* pages_sent[num_recv_cbs];

    uint32_t fifo_size;
    uint32_t fifo_limit;
    uint32_t fifo_limit_page_aligned;

    uint32_t fifo_page_size;
    uint32_t fifo_aligned_num_pages;

    uint32_t fifo_wr_ptr;

    uint32_t fifo_start_addr;

    uint32_t aligned_page_size;
};

RemoteSenderCBInterface<num_receivers> remote_cb_interface;

template<uint32_t aligned_page_size>
FORCE_INLINE void setup_remote_sender_cb_interface() {
    uint32_t num_pages = cb_size / page_size;
    uint32_t cb_size_page_aligned = num_pages * page_size;

    remote_cb_interface.fifo_size = cb_size;
    remote_cb_interface.fifo_limit = cb_size + cb_start_addr;
    remote_cb_interface.fifo_limit_page_aligned = cb_size_page_aligned + cb_start_addr;

    remote_cb_interface.fifo_page_size = page_size;
    remote_cb_interface.fifo_aligned_num_pages = num_pages * page_size / aligned_page_size;

    remote_cb_interface.fifo_wr_ptr = cb_wr_ptr;

    remote_cb_interface.fifo_start_addr = cb_start_addr;

    remote_cb_interface.num_receivers = num_receivers;

    for (uint32_t i=0; i < num_receivers; ++i) {
        remote_cb_interface.pages_acked[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(pages_acked_semaphore_addr[i]));
        remote_cb_interface.pages_sent[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(pages_sent_semaphore_addr[i]));
    }

    remote_cb_interface.aligned_page_size = aligned_page_size;

}

FORCE_INLINE void setup_remote_cb_page_size(uint32_t page_size) {
    uint32_t num_pages = remote_cb_interface.fifo_size / page_size;
    uint32_t cb_size_page_aligned = num_pages * page_size;

    remote_cb_interface.fifo_aligned_num_pages = num_pages * page_size / remote_cb_interface.aligned_page_size;
    remote_cb_interface.fifo_limit_page_aligned = cb_size_page_aligned + remote_cb_interface.fifo_start_addr;
}

FORCE_INLINE void remote_cb_reserve_back(uint32_t num_pages) {
    uint32_t len_bytes = num_pages * remote_cb_interface.fifo_page_size;
    uint32_t num_pages_wait = len_bytes / remote_cb_interface.aligned_page_size;
    uint32_t free_pages;

    for (uint32_t i=0; i < remote_cb_interface.num_receivers; ++i) {
        do {
            uint32_t pages_acked = (uint32_t)reg_read((uint32_t)remote_cb_interface.pages_acked[0]);
            uint32_t pages_sent = (uint32_t)reg_read((uint32_t)remote_cb_interface.pages_sent[0]);
            free_pages = remote_cb_interface.fifo_aligned_num_pages - (pages_sent - pages_acked);
        } while (free_pages < num_pages_wait);
    }
}

// unused for now, but we might need to use this one if we want to transfer the maximum noc packet
FORCE_INLINE void remote_cb_push_back_and_write_pages_(uint32_t local_cb_addr, uint32_t num_pages, uint32_t remote_noc_x, uint32_t remote_noc_y, uint8_t noc = noc_index) {
    uint32_t len_bytes = num_pages * remote_cb_interface.fifo_page_size;
    uint32_t pages_sent = len_bytes / remote_cb_interface.aligned_page_size;

    uint32_t local_fifo_rd_ptr = local_cb_addr;
    uint32_t remote_fifo_wr_ptr = remote_cb_interface.fifo_wr_ptr;

    uint32_t src_addr = local_cb_addr;
    uint32_t dest_addr = remote_cb_interface.fifo_wr_ptr;
    uint32_t remote_noc_xy = uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, remote_noc_x), DYNAMIC_NOC_Y(noc, remote_noc_y)));
    uint64_t dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);


    while (len_bytes > NOC_MAX_BURST_SIZE) {

        src_addr = local_fifo_rd_ptr;
        dest_addr = remote_fifo_wr_ptr;
        dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

        // split one write to two chunks
        if ((dest_addr + NOC_MAX_BURST_SIZE) >= remote_cb_interface.fifo_limit_page_aligned) {
            uint32_t first_len_bytes = remote_cb_interface.fifo_limit_page_aligned - dest_addr;
            uint32_t second_len_bytes = NOC_MAX_BURST_SIZE - first_len_bytes;

            // issue first write transfer
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            ncrisc_noc_fast_write(noc, write_cmd_buf, src_addr, dest_noc_addr, first_len_bytes, NOC_UNICAST_WRITE_VC, false, false, 1, true);
            src_addr += first_len_bytes;
            dest_addr = remote_cb_interface.fifo_start_addr;
            dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

            if (second_len_bytes != 0) {
                // issue second write transfer
                while (!noc_cmd_buf_ready(noc, write_cmd_buf));
                ncrisc_noc_fast_write(noc, write_cmd_buf, src_addr, dest_noc_addr, second_len_bytes, NOC_UNICAST_WRITE_VC, false, false, 1, true);
                src_addr += second_len_bytes;
                dest_addr += second_len_bytes;
            }

        } else { // issue write in one request
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            ncrisc_noc_fast_write(noc, write_cmd_buf, src_addr, dest_noc_addr, NOC_MAX_BURST_SIZE, NOC_UNICAST_WRITE_VC, false, false, 1, true);
            src_addr += NOC_MAX_BURST_SIZE;
            dest_addr += NOC_MAX_BURST_SIZE;
        }

        // update local and remote pointers
        local_fifo_rd_ptr = src_addr;
        remote_fifo_wr_ptr = dest_addr;

        len_bytes -= NOC_MAX_BURST_SIZE;
    }

    dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);
    // split one write to two chunks for last write
    if ((dest_addr + len_bytes) >= remote_cb_interface.fifo_limit_page_aligned) {

        uint32_t first_len_bytes = remote_cb_interface.fifo_limit_page_aligned - dest_addr;
        uint32_t second_len_bytes = len_bytes - first_len_bytes;

        // issue first write transfer
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        ncrisc_noc_fast_write(noc, write_cmd_buf, src_addr, dest_noc_addr, first_len_bytes, NOC_UNICAST_WRITE_VC, false, false, 1, true);
        src_addr += first_len_bytes;
        dest_addr = remote_cb_interface.fifo_start_addr;
        dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

        if (second_len_bytes != 0) {
            // issue second write transfer
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            ncrisc_noc_fast_write(noc, write_cmd_buf, src_addr, dest_noc_addr, second_len_bytes, NOC_UNICAST_WRITE_VC, false, false, 1, true);
            src_addr += second_len_bytes;
            dest_addr += second_len_bytes;
        }

    } else { // issue write in one request
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        ncrisc_noc_fast_write(noc, write_cmd_buf, src_addr, dest_noc_addr, len_bytes, NOC_UNICAST_WRITE_VC, false, false, 1, true);
        src_addr += len_bytes;
        dest_addr += len_bytes;
    }

    *remote_cb_interface.pages_sent += pages_sent;
    remote_cb_interface.fifo_wr_ptr = dest_addr;

    uint64_t remote_ack_ptr_addr = get_noc_addr_helper(remote_noc_xy, (uint32_t)remote_cb_interface.pages_sent);
    noc_semaphore_inc(remote_ack_ptr_addr, pages_sent, noc);
}

FORCE_INLINE void remote_cb_push_back_and_write_pages(uint32_t local_cb_addr, uint32_t num_pages, uint32_t num_rows, uint32_t coalesced_num_pages_per_row, uint32_t coalesced_page_size, uint32_t* remote_noc_x, uint32_t* remote_noc_y, uint8_t noc = noc_index) {
    uint32_t len_bytes = num_pages * remote_cb_interface.fifo_page_size;
    uint32_t pages_sent = len_bytes / remote_cb_interface.aligned_page_size;

    uint32_t next_receiver_start_addr_stride = coalesced_num_pages_per_row * coalesced_page_size;
    uint32_t next_block_row_stride = next_receiver_start_addr_stride * remote_cb_interface.num_receivers;

    uint32_t dest_addr;

    uint32_t next_receiver_start_addr_offset = 0;
    for (uint32_t i=0; i < remote_cb_interface.num_receivers; ++i) {

        uint32_t src_addr = local_cb_addr + next_receiver_start_addr_offset;
        dest_addr = remote_cb_interface.fifo_wr_ptr;
        uint32_t remote_noc_xy = uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, remote_noc_x[i]), DYNAMIC_NOC_Y(noc, remote_noc_y[i])));
        uint64_t dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

        noc_async_write_one_packet_set_state(dest_noc_addr, coalesced_page_size, noc);

        for (uint32_t h = 0; h < num_rows; ++h) {
            uint32_t prev_src_addr = src_addr;
            for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                if ((dest_addr + coalesced_page_size) > remote_cb_interface.fifo_limit_page_aligned) {

                    uint32_t first_len_bytes = remote_cb_interface.fifo_limit_page_aligned - dest_addr;
                    uint32_t second_len_bytes = coalesced_page_size - first_len_bytes;

                    if (first_len_bytes != 0) {
                        noc_async_write_one_packet(src_addr, dest_noc_addr, first_len_bytes, noc);
                        src_addr += first_len_bytes;
                    }

                    dest_addr = remote_cb_interface.fifo_start_addr;
                    dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                    noc_async_write_one_packet(src_addr, dest_noc_addr, second_len_bytes, noc);

                    src_addr += second_len_bytes;
                    dest_addr += second_len_bytes;
                    dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                    noc_async_write_one_packet_set_state(dest_noc_addr, coalesced_page_size, noc);

                } else {
                    noc_async_write_one_packet_with_state(src_addr, dest_noc_addr, noc);

                    src_addr += coalesced_page_size;
                    dest_addr += coalesced_page_size;
                }
            }
            src_addr = prev_src_addr + next_block_row_stride;
        }
        next_receiver_start_addr_offset += next_receiver_start_addr_stride;

        *remote_cb_interface.pages_sent[i] += pages_sent;

        uint64_t remote_ack_ptr_addr = get_noc_addr_helper(remote_noc_xy, (uint32_t)remote_cb_interface.pages_sent[i]);
        noc_semaphore_inc(remote_ack_ptr_addr, pages_sent, noc);
    }

    remote_cb_interface.fifo_wr_ptr = dest_addr;

}

void kernel_main() {

    uint32_t rt_args_idx = 0;
    noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_receivers)));
    noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_receivers)));
    pages_acked_semaphore_addr = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_receivers)));
    pages_sent_semaphore_addr = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_receivers)));

    constexpr uint32_t cb_id = 0;

    setup_remote_sender_cb_interface<ALIGNED_PAGE_SIZE>();

    for (uint32_t block = 0; block < num_blocks; ++block) {

        cb_wait_front(cb_id, block_num_tiles);

        uint32_t local_cb_addr = get_read_ptr(cb_id);
        remote_cb_reserve_back(receiver_block_num_tiles);
        remote_cb_push_back_and_write_pages(local_cb_addr, receiver_block_num_tiles, num_tile_rows, coalesced_num_pages, coalesced_page_size, noc_x, noc_y, noc);

        cb_pop_front(cb_id, block_num_tiles);

    }

}
