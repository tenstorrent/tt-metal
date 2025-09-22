// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <type_traits>

namespace tt::tt_fabric {

using chan_id_t = std::uint8_t;
using routing_plane_id_t = std::uint8_t;

static constexpr std::uint32_t CLIENT_INTERFACE_SIZE = 3280;
static constexpr std::uint32_t PACKET_WORD_SIZE_BYTES = 16;

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

// Compressed routing entry structures using manual bit packing
struct __attribute__((packed)) compressed_route_2d_t {
    // 16 bits total: ns_hops(5) + ew_hops(5) + ns_dir(1) + ew_dir(1) + turn_point(4)
    uint16_t data;

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    void set(uint8_t ns_hops, uint8_t ew_hops, uint8_t ns_dir, uint8_t ew_dir, uint8_t turn_point) {
        data = (ns_hops & 0x1F) | ((ew_hops & 0x1F) << 5) | ((ns_dir & 0x1) << 10) | ((ew_dir & 0x1) << 11) |
               ((turn_point & 0xF) << 12);
    }
#else
    uint8_t get_ns_hops() const { return data & 0x1F; }              // bits 0-4
    uint8_t get_ew_hops() const { return (data >> 5) & 0x1F; }       // bits 5-9
    uint8_t get_ns_direction() const { return (data >> 10) & 0x1; }  // bit 10
    uint8_t get_ew_direction() const { return (data >> 11) & 0x1; }  // bit 11
    uint8_t get_turn_point() const { return (data >> 12) & 0xF; }    // bits 12-15
#endif
};

static_assert(sizeof(compressed_route_2d_t) == 2, "2D route must be 2 bytes");

static const uint16_t MAX_CHIPS_LOWLAT_1D = 16;
static const uint16_t MAX_CHIPS_LOWLAT_2D = 256;
static const uint16_t SINGLE_ROUTE_SIZE_1D = 4;
static const uint16_t SINGLE_ROUTE_SIZE_2D = 32;

template <uint8_t dim, bool compressed>
struct __attribute__((packed)) routing_path_t {
    static_assert(dim == 1 || dim == 2, "dim must be 1 or 2");

    // For 1D: Create LowLatencyPacketHeader pattern
    static const uint32_t FIELD_WIDTH = 2;
    static const uint32_t WRITE_ONLY = 0b01;
    static const uint32_t FORWARD_ONLY = 0b10;
    static const uint32_t FWD_ONLY_FIELD = 0xAAAAAAAA;

    static const uint8_t NOOP = 0b0000;
    static const uint8_t FORWARD_EAST = 0b0001;
    static const uint8_t FORWARD_WEST = 0b0010;
    static const uint8_t FORWARD_NORTH = 0b0100;
    static const uint8_t FORWARD_SOUTH = 0b1000;
    static const uint8_t WRITE_AND_FORWARD_EAST = 0b0001;
    static const uint8_t WRITE_AND_FORWARD_WEST = 0b0010;
    static const uint8_t WRITE_AND_FORWARD_NORTH = 0b0100;
    static const uint8_t WRITE_AND_FORWARD_SOUTH = 0b1000;

    // Compressed routing uses much smaller encoding
    // 1D: 0 byte (num_hops passed from caller is the compressed info)
    static const uint16_t COMPRESSED_ROUTE_SIZE_1D = 0;
    // 2D: 2 bytes (ns_hops:5bits, ew_hops:5bits, ns_dir:1bit, ew_dir:1bit, turn_point:4bits)
    static const uint16_t COMPRESSED_ROUTE_SIZE_2D = sizeof(compressed_route_2d_t);

    static constexpr uint16_t MAX_CHIPS_LOWLAT = (dim == 1) ? MAX_CHIPS_LOWLAT_1D : MAX_CHIPS_LOWLAT_2D;
    static constexpr uint16_t COMPRESSED_ROUTE_SIZE = (dim == 1) ? COMPRESSED_ROUTE_SIZE_1D : COMPRESSED_ROUTE_SIZE_2D;
    static constexpr uint16_t SINGLE_ROUTE_SIZE =
        compressed ? ((dim == 1) ? COMPRESSED_ROUTE_SIZE_1D : COMPRESSED_ROUTE_SIZE_2D)
                   : ((dim == 1) ? SINGLE_ROUTE_SIZE_1D : SINGLE_ROUTE_SIZE_2D);

    typename std::conditional<
        !compressed,
        std::uint8_t[MAX_CHIPS_LOWLAT * SINGLE_ROUTE_SIZE],  // raw for uncompressed
        typename std::conditional<
            dim == 1,
            std::uint8_t[0],                         // empty for compressed 1D (0 bytes)
            compressed_route_2d_t[MAX_CHIPS_LOWLAT]  // two for compressed 2D
            >::type>::type paths = {};

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    // Routing calculation methods
    void calculate_chip_to_all_routing_fields(uint16_t src_chip_id, uint16_t num_chips, uint16_t ew_dim = 0);
#else
    // Device-side methods (declared here, implemented in fabric_routing_path_interface.h):
    inline bool decode_route_to_buffer(uint16_t dst_chip_id, volatile uint8_t* out_route_buffer) const;
#endif
};
// 16 chips * 4 bytes = 64
static_assert(sizeof(routing_path_t<1, false>) == 64, "1D uncompressed routing path must be 64 bytes");
static_assert(sizeof(routing_path_t<1, true>) == 0, "1D compressed routing path must be 0 bytes");
// 256 chips * 2 bytes = 512
static_assert(sizeof(routing_path_t<2, true>) == 512, "2D compressed routing path must be 512 bytes");

struct tensix_routing_l1_info_t {
    // TODO: https://github.com/tenstorrent/tt-metal/issues/28534
    //       these fabric node ids should be another struct as really commonly used data
    uint16_t my_mesh_id;    // Current mesh ID
    uint16_t my_device_id;  // Current chip ID
    // NOTE: Compressed version has additional overhead (2x slower) to read values,
    //       but raw data is too huge (2048 bytes) to fit in L1 memory.
    //       Need to evaluate once actual workloads are available
    compressed_routing_table_t<MAX_MESH_SIZE> intra_mesh_routing_table;   // 384 bytes
    compressed_routing_table_t<MAX_NUM_MESHES> inter_mesh_routing_table;  // 384 bytes
    uint8_t padding[12];                                                  // pad to 16-byte alignment
} __attribute__((packed));

struct fabric_connection_info_t {
    uint32_t edm_buffer_base_addr;
    uint32_t edm_connection_handshake_addr;
    uint32_t edm_worker_location_info_addr;
    uint32_t buffer_index_semaphore_id;
    uint16_t buffer_size_bytes;
    uint8_t edm_direction;
    uint8_t edm_noc_x;
    uint8_t edm_noc_y;
    uint8_t num_buffers_per_channel;
    uint16_t worker_free_slots_stream_id;
} __attribute__((packed));

static_assert(sizeof(fabric_connection_info_t) == 24, "Struct size mismatch!");
// NOTE: This assertion can be removed once "non device-init fabric"
//       is completely removed
static_assert(sizeof(fabric_connection_info_t) % 4 == 0, "Struct size must be 4-byte aligned");

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
#include "fabric/hw/inc/fabric_routing_path_interface.h"
#endif
