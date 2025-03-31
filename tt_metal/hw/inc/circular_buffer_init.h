// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "circular_buffer.h"
#include "circular_buffer_constants.h"
#include "remote_circular_buffer_api.h"
#include "risc_attribs.h"

// NCRISC and BRISC setup read and write
// TRISC sets up read or write
FORCE_INLINE void setup_local_cb_read_write_interfaces(
    uint32_t tt_l1_ptr* cb_l1_base,
    uint32_t start_cb_index,
    uint32_t max_cb_index,
    bool read,
    bool write,
    bool init_wr_tile_ptr) {
    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr =
        cb_l1_base + start_cb_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;

    for (uint32_t cb_id = start_cb_index; cb_id < max_cb_index; cb_id++) {
        // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
        uint32_t fifo_addr = circular_buffer_config_addr[0] >> cb_addr_shift;
        uint32_t fifo_size = circular_buffer_config_addr[1] >> cb_addr_shift;
        uint32_t fifo_num_pages = circular_buffer_config_addr[2];
        uint32_t fifo_page_size = circular_buffer_config_addr[3] >> cb_addr_shift;
        uint32_t fifo_limit = fifo_addr + fifo_size;

        LocalCBInterface& local_interface = get_local_cb_interface(cb_id);
        local_interface.fifo_limit = fifo_limit;  // to check if we need to wrap
        if (write) {
            local_interface.fifo_wr_ptr = fifo_addr;
        }
        if (read) {
            local_interface.fifo_rd_ptr = fifo_addr;
        }
        local_interface.fifo_size = fifo_size;
        local_interface.tiles_acked_received_init = 0;
        if (write) {
            local_interface.fifo_num_pages = fifo_num_pages;
        }
        local_interface.fifo_page_size = fifo_page_size;

        if (init_wr_tile_ptr) {
            local_interface.fifo_wr_tile_ptr = 0;
        }

        circular_buffer_config_addr += UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;
    }
}

namespace experimental {

template <bool update_remote_over_noc = false>
inline void setup_remote_cb_interfaces(
    uint32_t tt_l1_ptr* cb_l1_base, uint32_t start_cb_index, uint8_t noc, uint8_t nm, bool posted, uint8_t cmd_buf) {
    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr = cb_l1_base;

    for (uint32_t cb_id = NUM_CIRCULAR_BUFFERS - 1, end_id = start_cb_index - 1; cb_id != end_id; cb_id--) {
        uint32_t config_addr = circular_buffer_config_addr[0];
        uint32_t page_size = circular_buffer_config_addr[1];
        volatile tt_l1_ptr uint32_t* l1_remote_cb_config_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_addr);
        const bool is_sender = l1_remote_cb_config_addr[0];
        uint32_t num_receivers = l1_remote_cb_config_addr[1];
        uint32_t fifo_start_addr = l1_remote_cb_config_addr[2];
        uint32_t fifo_size = l1_remote_cb_config_addr[3];
        uint32_t fifo_ptr = l1_remote_cb_config_addr[4];
        uint32_t remote_noc_xy_addr = l1_remote_cb_config_addr[5];
        uint32_t aligned_pages_sent_addr = l1_remote_cb_config_addr[6];
        if (is_sender) {
            RemoteSenderCBInterface& sender_cb_interface = get_remote_sender_cb_interface(cb_id);
            sender_cb_interface.config_ptr = config_addr;
            sender_cb_interface.fifo_start_addr = fifo_start_addr;
            sender_cb_interface.fifo_wr_ptr = fifo_ptr;
            sender_cb_interface.receiver_noc_xy_ptr = remote_noc_xy_addr;
            sender_cb_interface.aligned_pages_sent_ptr = aligned_pages_sent_addr;
            sender_cb_interface.num_receivers = num_receivers;
            // Using posted semaphore inc
            resize_remote_sender_cb_interface<update_remote_over_noc>(cb_id, page_size, noc, nm, posted, cmd_buf);
        } else {
            uint32_t aligned_pages_acked_addr = aligned_pages_sent_addr + L1_ALIGNMENT;
            uint32_t sender_noc_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr)[0];
            uint32_t sender_noc_y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr)[1];
            RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
            receiver_cb_interface.config_ptr = config_addr;
            receiver_cb_interface.fifo_start_addr = fifo_start_addr;
            receiver_cb_interface.fifo_rd_ptr = fifo_ptr;
            receiver_cb_interface.sender_noc_x = sender_noc_x;
            receiver_cb_interface.sender_noc_y = sender_noc_y;
            receiver_cb_interface.aligned_pages_acked_ptr = aligned_pages_acked_addr;
            // Using posted semaphore inc
            resize_remote_receiver_cb_interface<update_remote_over_noc>(cb_id, page_size, noc, nm, posted, cmd_buf);
        }
        circular_buffer_config_addr += UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG;
    }
}

}  // namespace experimental
