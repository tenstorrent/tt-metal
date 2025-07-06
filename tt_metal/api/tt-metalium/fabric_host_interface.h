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

enum eth_chan_magic_values : std::uint8_t {
    INVALID_DIRECTION = 0xDD,
    INVALID_ROUTING_TABLE_ENTRY = 0xFF,
};

enum eth_chan_directions : std::uint8_t {
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

struct tensix_routing_l1_info_t {
    uint32_t mesh_id;           // Current mesh ID
    uint32_t device_id;         // Current device ID

    eth_chan_directions intra_mesh_routing_table[MAX_MESH_SIZE];
    eth_chan_directions inter_mesh_routing_table[MAX_NUM_MESHES];
    std::uint8_t padding[8];  // pad to 16-byte alignment
} __attribute__((packed));

// MEM_TENSIX_ROUTING_TABLE_SIZE
static_assert(sizeof(tensix_routing_l1_info_t) == 2064, "Struct size mismatch!");

struct fabric_connection_info_t {
    uint32_t edm_direction;
    uint32_t edm_noc_xy;  // packed x,y coordinates
    uint32_t edm_buffer_base_addr;
    uint32_t num_buffers_per_channel;
    uint32_t edm_l1_sem_addr;
    uint32_t edm_connection_handshake_addr;
    uint32_t edm_worker_location_info_addr;
    uint32_t buffer_size_bytes;
    uint32_t buffer_index_semaphore_id;
} __attribute__((packed));

// Fabric connection metadata stored in worker L1
// 16 for WH, 12 for BH
struct tensix_fabric_connections_l1_info_t {
    static constexpr uint8_t MAX_FABRIC_ENDPOINTS = 16;
    // Each index corresponds to ethernet channel index
    fabric_connection_info_t connections[MAX_FABRIC_ENDPOINTS];
    uint32_t valid_connections_mask;  // bit mask indicating which connections are valid
    uint8_t padding[12];              // pad to cache line alignment
} __attribute__((packed));

static_assert(sizeof(tensix_fabric_connections_l1_info_t) == 592, "Struct size mismatch!");

}  // namespace tt::tt_fabric
