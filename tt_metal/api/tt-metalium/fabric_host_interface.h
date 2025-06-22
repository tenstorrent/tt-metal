// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// TODO: move routing table here
namespace tt::tt_fabric {

using chan_id_t = std::uint8_t;
using routing_plane_id_t = std::uint8_t;

static constexpr std::uint32_t DEFAULT_ROUTER_RX_QUEUE_SIZE_BYTES = 0x8000;  // maximum queue (power of 2);

static constexpr std::uint32_t MAX_MESH_SIZE = 1024;
static constexpr std::uint32_t MAX_NUM_MESHES = 1024;

static constexpr std::uint32_t NUM_CHANNELS_PER_UINT32 = sizeof(std::uint32_t) / sizeof(chan_id_t);
static constexpr std::uint32_t LOG_BASE_2_NUM_CHANNELS_PER_UINT32 = 2;
static constexpr std::uint32_t MODULO_LOG_BASE_2 = (1 << LOG_BASE_2_NUM_CHANNELS_PER_UINT32) - 1;
static constexpr std::uint32_t NUM_TABLE_ENTRIES = MAX_MESH_SIZE >> LOG_BASE_2_NUM_CHANNELS_PER_UINT32;

static_assert(MAX_MESH_SIZE == MAX_NUM_MESHES, "MAX_MESH_SIZE must be equal to MAX_NUM_MESHES");
static_assert(
    (sizeof(std::uint32_t) / sizeof(chan_id_t)) == NUM_CHANNELS_PER_UINT32,
    "LOG_BASE_2_NUM_CHANNELS_PER_UINT32 must be equal to log2(sizeof(std::uint32_t) / sizeof(chan_id_t))");

static constexpr std::uint32_t CLIENT_INTERFACE_SIZE = 3280;
static constexpr std::uint32_t CLIENT_HEADER_BUFFER_ENTRIES = 4;
static constexpr std::uint32_t GATEKEEPER_INFO_SIZE = 848;
static constexpr std::uint32_t PULL_CLIENT_INTERFACE_SIZE = 304;
static constexpr std::uint32_t PUSH_CLIENT_INTERFACE_SIZE = 288;
static constexpr std::uint32_t PACKET_WORD_SIZE_BYTES = 16;
static constexpr std::uint32_t PACKET_HEADER_SIZE_BYTES = 48;
static constexpr std::uint32_t PACKET_HEADER_SIZE_WORDS = PACKET_HEADER_SIZE_BYTES / PACKET_WORD_SIZE_BYTES;

enum packet_session_command : std::uint32_t {
    ASYNC_WR = (0x1 << 0),
    ASYNC_WR_RESP = (0x1 << 1),
    ASYNC_RD = (0x1 << 2),
    ASYNC_RD_RESP = (0x1 << 3),
    DSOCKET_WR = (0x1 << 4),
    SSOCKET_WR = (0x1 << 5),
    ATOMIC_INC = (0x1 << 6),
    ATOMIC_READ_INC = (0x1 << 7),
    SOCKET_OPEN = (0x1 << 8),
    SOCKET_CLOSE = (0x1 << 9),
    SOCKET_CONNECT = (0x1 << 10),
};

enum eth_chan_magic_values {
    INVALID_DIRECTION = 0xDD,
    INVALID_ROUTING_TABLE_ENTRY = 0xFF,
};

enum eth_chan_directions {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    COUNT = 4,
};

struct routing_table_t {
    chan_id_t dest_entry[MAX_MESH_SIZE];
};

struct port_direction_t {
    chan_id_t directions[eth_chan_directions::COUNT];
};

struct fabric_router_l1_config_t {
    routing_table_t intra_mesh_table;
    routing_table_t inter_mesh_table;
    port_direction_t port_direction;
    std::uint16_t my_mesh_id;  // Do we need this if we tag routing tables with magic values for outbound eth channels
                               // and route to local NOC?
    std::uint16_t my_device_id;
    std::uint16_t east_dim;
    std::uint16_t north_dim;
    std::uint8_t padding[4];  // pad to 16-byte alignment.
} __attribute__((packed));

}  // namespace tt::tt_fabric
