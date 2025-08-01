// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

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

// 3 bit expression
enum class compressed_routing_values : std::uint8_t {
    COMPRESSED_EAST = 0,
    COMPRESSED_WEST = 1,
    COMPRESSED_NORTH = 2,
    COMPRESSED_SOUTH = 3,
    COMPRESSED_INVALID_DIRECTION = 4,            // Maps to INVALID_DIRECTION (0xDD)
    COMPRESSED_INVALID_ROUTING_TABLE_ENTRY = 5,  // Maps to INVALID_ROUTING_TABLE_ENTRY (0xFF)
};

// Compressed routing table using 3 bits
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

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)  // SW (Host)
    void set_direction(std::uint16_t index, std::uint8_t direction) {
        std::uint32_t bit_index = index * BITS_PER_COMPRESSED_ENTRY;
        std::uint32_t byte_index = bit_index / BITS_PER_BYTE;
        std::uint32_t bit_offset = bit_index % BITS_PER_BYTE;

        if (bit_offset <= 5) {
            // All 3 bits are in the same byte
            packed_directions[byte_index] &= ~(COMPRESSED_ENTRY_MASK << bit_offset);             // Clear bits
            packed_directions[byte_index] |= (direction & COMPRESSED_ENTRY_MASK) << bit_offset;  // Set bits
        } else {
            // Bits span across two bytes
            std::uint8_t bits_in_first_byte = BITS_PER_BYTE - bit_offset;
            std::uint8_t bits_in_second_byte = BITS_PER_COMPRESSED_ENTRY - bits_in_first_byte;

            // Clear and set bits in first byte
            packed_directions[byte_index] &= ~(((1 << bits_in_first_byte) - 1) << bit_offset);
            packed_directions[byte_index] |= (direction & ((1 << bits_in_first_byte) - 1)) << bit_offset;

            // Clear and set bits in second byte
            packed_directions[byte_index + 1] &= ~((1 << bits_in_second_byte) - 1);
            packed_directions[byte_index + 1] |= (direction >> bits_in_first_byte) & ((1 << bits_in_second_byte) - 1);
        }
    }

    std::uint8_t compress_value(std::uint8_t original_value) const {
        switch (original_value) {
            case eth_chan_directions::EAST:
                return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_EAST);
            case eth_chan_directions::WEST:
                return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_WEST);
            case eth_chan_directions::NORTH:
                return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_NORTH);
            case eth_chan_directions::SOUTH:
                return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_SOUTH);
            case eth_chan_magic_values::INVALID_DIRECTION:
                return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_INVALID_DIRECTION);
            case eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY:
                return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_INVALID_ROUTING_TABLE_ENTRY);
            default:
                return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_INVALID_ROUTING_TABLE_ENTRY);
        }
    }

    void set_original_direction(std::uint16_t index, std::uint8_t original_direction) {
        set_direction(index, compress_value(original_direction));
    }
#else   // HW (Device)
    inline std::uint8_t get_direction(std::uint16_t index) const {
        std::uint32_t bit_index = index * BITS_PER_COMPRESSED_ENTRY;
        std::uint32_t byte_index = bit_index / BITS_PER_BYTE;
        std::uint32_t bit_offset = bit_index % BITS_PER_BYTE;

        if (bit_offset <= 5) {
            // All 3 bits are in the same byte
            return (packed_directions[byte_index] >> bit_offset) & COMPRESSED_ENTRY_MASK;
        } else {
            // Bits span across two bytes
            std::uint8_t low_bits =
                (packed_directions[byte_index] >> bit_offset) & ((1 << (BITS_PER_BYTE - bit_offset)) - 1);
            std::uint8_t high_bits = (packed_directions[byte_index + 1] &
                                      ((1 << (BITS_PER_COMPRESSED_ENTRY - (BITS_PER_BYTE - bit_offset))) - 1))
                                     << (BITS_PER_BYTE - bit_offset);
            return low_bits | high_bits;
        }
    }
    inline std::uint8_t decompress_value(std::uint8_t compressed_value) const {
        switch (static_cast<compressed_routing_values>(compressed_value)) {
            case compressed_routing_values::COMPRESSED_EAST: return eth_chan_directions::EAST;
            case compressed_routing_values::COMPRESSED_WEST: return eth_chan_directions::WEST;
            case compressed_routing_values::COMPRESSED_NORTH: return eth_chan_directions::NORTH;
            case compressed_routing_values::COMPRESSED_SOUTH: return eth_chan_directions::SOUTH;
            case compressed_routing_values::COMPRESSED_INVALID_DIRECTION:
                return eth_chan_magic_values::INVALID_DIRECTION;
            case compressed_routing_values::COMPRESSED_INVALID_ROUTING_TABLE_ENTRY:
                return eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY;
            default: return eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY;
        }
    }

    inline std::uint8_t get_original_direction(std::uint16_t index) const {
        return decompress_value(get_direction(index));
    }
#endif  // KERNEL_BUILD or FW_BUILD
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
    // NOTE: Compressed version has additional overhead (2x slower) to read values,
    //       but raw data is too huge (2048 bytes) to fit in L1 memory.
    //       Need to evaluate once actual workloads are available
    compressed_routing_table_t<MAX_MESH_SIZE> intra_mesh_routing_table;   // 384 bytes
    compressed_routing_table_t<MAX_NUM_MESHES> inter_mesh_routing_table;  // 384 bytes
    uint8_t padding[12];                                  // pad to 16-byte alignment
} __attribute__((packed));

constexpr std::uint8_t USE_DYNAMIC_CREDIT_ADDR = 255;

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
