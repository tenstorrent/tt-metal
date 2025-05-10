// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Config Buffer on Sender Core will be populated as follows
struct sender_socket_md {
    // Standard Config Entries
    uint32_t bytes_acked = 0;
    uint32_t write_ptr = 0;
    uint32_t bytes_sent = 0;

    // Downstream Socket Metadata
    uint32_t downstream_mesh_id = 0;
    uint32_t downstream_chip_id = 0;
    uint32_t downstream_noc_y = 0;
    uint32_t downstream_noc_x = 0;
    uint32_t downstream_bytes_sent_addr = 0;
    uint32_t downstream_fifo_addr = 0;
    uint32_t downstream_fifo_total_size = 0;

    uint32_t is_sender = 0;
};

// Config Buffer on Receiver Cores will be populated as follows
struct receiver_socket_md {
    // Standard Config Entries
    uint32_t bytes_sent = 0;
    uint32_t bytes_acked = 0;
    uint32_t read_ptr = 0;
    uint32_t fifo_addr = 0;
    uint32_t fifo_total_size = 0;

    // Upstream Socket Metadata
    uint32_t upstream_mesh_id = 0;
    uint32_t upstream_chip_id = 0;
    uint32_t upstream_noc_y = 0;
    uint32_t upstream_noc_x = 0;
    uint32_t upstream_bytes_acked_addr = 0;

    uint32_t is_sender = 0;
};

struct SocketSenderInterface {
    uint32_t config_addr;
    uint32_t write_ptr;
    uint32_t bytes_sent;
    uint32_t bytes_acked_addr;
    uint32_t page_size;

    // Downstream Socket Metadata
    uint32_t downstream_mesh_id;
    uint32_t downstream_chip_id;
    uint32_t downstream_noc_y;
    uint32_t downstream_noc_x;
    uint32_t downstream_bytes_sent_addr;
    uint32_t downstream_fifo_addr;
    uint32_t downstream_fifo_total_size;
    uint32_t downstream_fifo_curr_size;
};

struct SocketReceiverInterface {
    uint32_t config_addr;
    uint32_t read_ptr;
    uint32_t bytes_acked;
    uint32_t bytes_sent_addr;
    uint32_t page_size;
    uint32_t fifo_addr;
    uint32_t fifo_total_size;
    uint32_t fifo_curr_size;

    // Upstream Socket Metadata
    uint32_t upstream_mesh_id;
    uint32_t upstream_chip_id;
    uint32_t upstream_noc_y;
    uint32_t upstream_noc_x;
    uint32_t upstream_bytes_acked_addr;
};
