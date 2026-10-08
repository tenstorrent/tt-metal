// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

struct d2h_sender_socket_md {
    uint32_t bytes_sent_addr_hi;
    uint32_t data_addr_hi;
    uint32_t pcie_xy_enc;
};

struct d2d_sender_socket_md {
    uint32_t downstream_mesh_id;
    uint32_t downstream_chip_id;
    uint32_t downstream_noc_y;
    uint32_t downstream_noc_x;
};

union sender_downstream_encoding {
    d2h_sender_socket_md d2h;
    d2d_sender_socket_md d2d;
} __attribute__((packed));

// Config Buffer on Sender Core will be populated as follows. Metadata size based on number of downstream receivers.
struct sender_socket_md {
    // Standard Config Entries
    uint32_t bytes_sent = 0;
    uint32_t num_downstreams = 0;
    uint32_t write_ptr = 0;
    uint32_t downstream_bytes_sent_addr = 0;
    uint32_t downstream_fifo_addr = 0;
    uint32_t downstream_fifo_total_size = 0;
    uint32_t is_d2h = 0;
};
// After the metadata, the buffer contains the following arrays:
// uint32_t bytes_acked_array[num_downstreams]
// sender_downstream_encoding[num_downstreams]

// Config Buffer on Receiver Cores will be populated as follows

struct h2d_socket_md {
    uint32_t bytes_acked_addr_lo;
    uint32_t bytes_acked_addr_hi;
    uint32_t data_addr_lo;
    uint32_t data_addr_hi;
    uint32_t pcie_xy_enc;
};

struct d2d_recv_socket_md {
    uint32_t upstream_mesh_id;
    uint32_t upstream_chip_id;
    uint32_t upstream_noc_y;
    uint32_t upstream_noc_x;
    uint32_t upstream_bytes_acked_addr;
};

// A lane is one sender writing in order along one route (one NoC, or one fabric link). Lanes do not order against
// each other, so when a page is split over several lanes one shared counter cannot tell the receiver it has all of
// it: one lane's shares of pages n and n+1 can add up to a page before another lane's share of page n lands. Each
// lane therefore has its own bytes_sent word. Lane 0's is bytes_sent; lanes 1..num_lanes-1 follow the struct. After
// its share of a page, a lane adds the page's bytes_sent increment to its word, so every word tracks the sender's
// bytes_sent once that lane's share has landed, and the receiver takes a page once every word covers it. num_lanes
// 0 or 1 is the single-counter socket.
struct receiver_socket_md {
    uint32_t bytes_sent;
    uint32_t read_ptr;
    uint32_t fifo_addr;
    uint32_t fifo_total_size;
    uint32_t bytes_acked;
    uint32_t is_h2d;
    union {
        h2d_socket_md h2d;
        d2d_recv_socket_md d2d;
    } __attribute__((packed));
    uint32_t num_lanes;
};

// Receiver config size: the struct, then a bytes_sent word for each lane after lane 0.
constexpr uint32_t receiver_socket_config_size_bytes(uint32_t num_lanes) {
    return sizeof(receiver_socket_md) + (num_lanes > 1 ? num_lanes - 1 : 0) * sizeof(uint32_t);
}

struct H2DSocketInterface {
    uint32_t bytes_acked_addr_lo;
    uint32_t bytes_acked_addr_hi;
    uint32_t data_addr_lo;
    uint32_t data_addr_hi;
    uint32_t pcie_xy_enc;
};

struct D2DSocketRecvInterface {
    uint32_t upstream_mesh_id;
    uint32_t upstream_chip_id;
    uint32_t upstream_noc_y;
    uint32_t upstream_noc_x;
    uint32_t upstream_bytes_acked_addr;
};

struct D2HSocketInterface {
    uint32_t bytes_sent_addr_hi;
    uint32_t data_addr_hi;
    uint32_t pcie_xy_enc;
};

struct D2DSocketSendInterface {
    uint32_t downstream_enc_base_addr;
};

struct SocketSenderInterface {
    uint32_t config_addr;
    uint32_t write_ptr;
    uint32_t bytes_sent;
    uint32_t bytes_acked_base_addr;
    uint32_t page_size;
    uint32_t num_downstreams;
    uint32_t downstream_fifo_total_size;
    uint32_t downstream_fifo_curr_size;
    uint32_t downstream_fifo_addr;
    uint32_t downstream_bytes_sent_addr;
    uint32_t is_d2h;
    union {
        D2HSocketInterface d2h;
        D2DSocketSendInterface d2d;
    } __attribute__((packed));
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
    uint32_t is_h2d;
    union {
        H2DSocketInterface h2d;
        D2DSocketRecvInterface d2d;
    } __attribute__((packed));
    uint32_t num_lanes;
    uint32_t extra_lanes_bytes_sent_addr;
};
