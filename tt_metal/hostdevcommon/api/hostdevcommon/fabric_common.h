// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

namespace tt::tt_fabric {

using chan_id_t = std::uint8_t;
using routing_plane_id_t = std::uint8_t;

static constexpr std::uint32_t CLIENT_INTERFACE_SIZE = 3280;
static constexpr std::uint32_t GATEKEEPER_INFO_SIZE = 848;
static constexpr std::uint32_t PULL_CLIENT_INTERFACE_SIZE = 304;
static constexpr std::uint32_t PACKET_WORD_SIZE_BYTES = 16;
static constexpr std::uint32_t PACKET_HEADER_SIZE_BYTES = 48;
static constexpr std::uint32_t PACKET_HEADER_SIZE_WORDS = PACKET_HEADER_SIZE_BYTES / PACKET_WORD_SIZE_BYTES;

// Constants for fabric mesh configuration
static constexpr std::uint32_t MAX_MESH_SIZE = 1024;
static constexpr std::uint32_t MAX_NUM_MESHES = 1024;
static_assert(MAX_MESH_SIZE == MAX_NUM_MESHES, "MAX_MESH_SIZE must be equal to MAX_NUM_MESHES");

constexpr std::uint8_t USE_DYNAMIC_CREDIT_ADDR = 255;

// Magic values for ethernet channel directions
enum eth_chan_magic_values : std::uint8_t {
    INVALID_DIRECTION = 0xDD,
    INVALID_ROUTING_TABLE_ENTRY = 0xFF,
};

// Ethernet channel directions
enum eth_chan_directions : std::uint8_t {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    COUNT = 4,
};

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

// 3 bit expression
enum class compressed_routing_values : std::uint8_t {
    COMPRESSED_EAST = 0,
    COMPRESSED_WEST = 1,
    COMPRESSED_NORTH = 2,
    COMPRESSED_SOUTH = 3,
    COMPRESSED_INVALID_DIRECTION = 4,            // Maps to INVALID_DIRECTION (0xDD)
    COMPRESSED_INVALID_ROUTING_TABLE_ENTRY = 5,  // Maps to INVALID_ROUTING_TABLE_ENTRY (0xFF)
};

// Compressed routing table base structure using 3 bits
template <std::uint32_t ArraySize>
struct __attribute__((packed)) compressed_routing_table_t {
    static constexpr std::uint32_t BITS_PER_COMPRESSED_ENTRY = 3;
    static constexpr std::uint8_t COMPRESSED_ENTRY_MASK = 0x7;                // 3-bit mask (2^3 - 1)
    static constexpr std::uint32_t BITS_PER_BYTE = sizeof(std::uint8_t) * 8;  // 8 bits in a byte
    static_assert(
        (ArraySize * BITS_PER_COMPRESSED_ENTRY) % BITS_PER_BYTE == 0,
        "ArraySize * BITS_PER_COMPRESSED_ENTRY must be divisible by BITS_PER_BYTE for optimal packing");

    // 3 bits per entry, so 8 entries per 3 bytes (24 bits)
    // For 1024 entries: 1024 * 3 / 8 = 384 bytes
    std::uint8_t packed_directions[ArraySize * BITS_PER_COMPRESSED_ENTRY / BITS_PER_BYTE];  // 384 bytes

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    // Host-side methods (declared here, implemented in compressed_routing_table.cpp):
    void set_direction(std::uint16_t index, std::uint8_t direction);
    std::uint8_t compress_value(std::uint8_t original_value) const;
    void set_original_direction(std::uint16_t index, std::uint8_t original_direction);
#else
    // Device-side methods (declared here, implemented in fabric_routing_table_interface.h):
    inline std::uint8_t get_direction(std::uint16_t index) const;
    inline std::uint8_t decompress_value(std::uint8_t compressed_value) const;
    inline std::uint8_t get_original_direction(std::uint16_t index) const;
#endif
};

struct tensix_routing_l1_info_t {
    uint32_t mesh_id;  // Current mesh ID
    // NOTE: Compressed version has additional overhead (2x slower) to read values,
    //       but raw data is too huge (2048 bytes) to fit in L1 memory.
    //       Need to evaluate once actual workloads are available
    compressed_routing_table_t<MAX_MESH_SIZE> intra_mesh_routing_table;   // 384 bytes
    compressed_routing_table_t<MAX_NUM_MESHES> inter_mesh_routing_table;  // 384 bytes
    uint8_t padding[12];                                                  // pad to 16-byte alignment
} __attribute__((packed));

struct fabric_connection_info_t {
    uint8_t edm_direction;
    uint8_t edm_noc_x;
    uint8_t edm_noc_y;
    uint32_t edm_buffer_base_addr;
    uint8_t num_buffers_per_channel;
    uint32_t edm_l1_sem_addr;
    uint32_t edm_connection_handshake_addr;
    uint32_t edm_worker_location_info_addr;
    uint16_t buffer_size_bytes;
    uint32_t buffer_index_semaphore_id;
} __attribute__((packed));

static_assert(sizeof(fabric_connection_info_t) == 26, "Struct size mismatch!");

struct fabric_aligned_connection_info_t {
    // 16-byte aligned semaphore address for flow control
    uint32_t worker_flow_control_semaphore;
    uint32_t padding_0[3];
};

struct tensix_fabric_connections_l1_info_t {
    static constexpr uint8_t MAX_FABRIC_ENDPOINTS = 16;
    // Each index corresponds to ethernet channel index
    fabric_connection_info_t read_only[MAX_FABRIC_ENDPOINTS];
    uint32_t valid_connections_mask;  // bit mask indicating which connections are valid
    uint32_t padding_0[3];            // pad to 16-byte alignment
    fabric_aligned_connection_info_t read_write[MAX_FABRIC_ENDPOINTS];
};

}  // namespace tt::tt_fabric

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "fabric/hw/inc/fabric_routing_table_interface.h"
#endif
