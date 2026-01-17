// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <type_traits>

namespace tt::tt_fabric {

// Forward declaration to avoid including heavy host-only headers here
class FabricNodeId;

using chan_id_t = std::uint8_t;
using routing_plane_id_t = std::uint8_t;

static constexpr std::uint32_t CLIENT_INTERFACE_SIZE = 3280;
static constexpr std::uint32_t PACKET_WORD_SIZE_BYTES = 16;

// Constants for fabric mesh configuration
static constexpr std::uint32_t MAX_MESH_SIZE = 256;
static constexpr std::uint32_t MAX_NUM_MESHES = 1024;

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
    Z = 4,
    COUNT = 5,
};

template <size_t ArraySize>
struct routing_table_t {
    chan_id_t dest_entry[ArraySize];
};

struct port_direction_t {
    chan_id_t directions[eth_chan_directions::COUNT];
};

// 3 bit expression
enum class compressed_routing_values : std::uint8_t {
    COMPRESSED_EAST = 0,
    COMPRESSED_WEST = 1,
    COMPRESSED_NORTH = 2,
    COMPRESSED_SOUTH = 3,
    COMPRESSED_Z = 4,
    COMPRESSED_INVALID_DIRECTION = 5,            // Maps to INVALID_DIRECTION (0xDD)
    COMPRESSED_INVALID_ROUTING_TABLE_ENTRY = 6,  // Maps to INVALID_ROUTING_TABLE_ENTRY (0xFF)
};

// Compressed routing table base structure using 3 bits
template <std::uint32_t ArraySize>
struct __attribute__((packed)) direction_table_t {
    static constexpr std::uint32_t BITS_PER_COMPRESSED_ENTRY = 3;
    static constexpr std::uint8_t COMPRESSED_ENTRY_MASK = 0x7;                // 3-bit mask (2^3 - 1)
    static constexpr std::uint32_t BITS_PER_BYTE = sizeof(std::uint8_t) * 8;  // 8 bits in a byte
    static_assert(
        (ArraySize * BITS_PER_COMPRESSED_ENTRY) % BITS_PER_BYTE == 0,
        "ArraySize * BITS_PER_COMPRESSED_ENTRY must be divisible by BITS_PER_BYTE for optimal packing");

    // 3 bits per entry, so 8 entries per 3 bytes (24 bits)
    // For 1024 entries: 1024 * 3 / 8 = 384 bytes
    std::uint8_t packed_directions[ArraySize * BITS_PER_COMPRESSED_ENTRY / BITS_PER_BYTE];

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    // Host-side methods (declared here, implemented in compressed_direction_table.cpp):
    void set_direction(std::uint16_t index, std::uint8_t direction);
    std::uint8_t compress_value(std::uint8_t original_value) const;
    void set_original_direction(std::uint16_t index, std::uint8_t original_direction);
#else
    // Device-side methods (declared here, implemented in fabric_direction_table_interface.h):
    inline std::uint8_t get_direction(std::uint16_t index) const;
    inline std::uint8_t decompress_value(std::uint8_t compressed_value) const;
    inline std::uint8_t get_original_direction(std::uint16_t index) const;
#endif
};

// Compressed routing entry structures using manual bit packing
// Note: Using uint32_t (4B) instead of packed uint16_t+uint8_t (3B) because union with 1D table
// makes both equivalent in memory (union = max(1024, 1024) = 1024B). Prioritizing performance.
// Can switch to 3B packed (uint16_t+uint8_t) if memory becomes critical in future.
struct __attribute__((packed)) compressed_route_2d_t {
    // Field widths (source of truth)
    static constexpr uint32_t NS_HOPS_WIDTH = 7;
    static constexpr uint32_t EW_HOPS_WIDTH = 7;
    static constexpr uint32_t NS_DIR_WIDTH = 1;
    static constexpr uint32_t EW_DIR_WIDTH = 1;
    static constexpr uint32_t TURN_POINT_WIDTH = 7;

    // Bit positions (derived from widths)
    static constexpr uint32_t NS_HOPS_SHIFT = 0;
    static constexpr uint32_t EW_HOPS_SHIFT = NS_HOPS_SHIFT + NS_HOPS_WIDTH;   // 7
    static constexpr uint32_t NS_DIR_SHIFT = EW_HOPS_SHIFT + EW_HOPS_WIDTH;    // 14
    static constexpr uint32_t EW_DIR_SHIFT = NS_DIR_SHIFT + NS_DIR_WIDTH;      // 15
    static constexpr uint32_t TURN_POINT_SHIFT = EW_DIR_SHIFT + EW_DIR_WIDTH;  // 16

    // Masks (derived from widths)
    static constexpr uint32_t NS_HOPS_MASK = (1U << NS_HOPS_WIDTH) - 1;        // 0x7F
    static constexpr uint32_t EW_HOPS_MASK = (1U << EW_HOPS_WIDTH) - 1;        // 0x7F
    static constexpr uint32_t NS_DIR_MASK = (1U << NS_DIR_WIDTH) - 1;          // 0x1
    static constexpr uint32_t EW_DIR_MASK = (1U << EW_DIR_WIDTH) - 1;          // 0x1
    static constexpr uint32_t TURN_POINT_MASK = (1U << TURN_POINT_WIDTH) - 1;  // 0x7F

    uint32_t data;

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    void set(uint8_t ns_hops, uint8_t ew_hops, uint8_t ns_dir, uint8_t ew_dir, uint8_t turn_point) {
        data = (ns_hops & NS_HOPS_MASK) | ((ew_hops & EW_HOPS_MASK) << EW_HOPS_SHIFT) |
               ((ns_dir & NS_DIR_MASK) << NS_DIR_SHIFT) | ((ew_dir & EW_DIR_MASK) << EW_DIR_SHIFT) |
               ((turn_point & TURN_POINT_MASK) << TURN_POINT_SHIFT);
    }
#else
    uint8_t get_ns_hops() const { return data & NS_HOPS_MASK; }
    uint8_t get_ew_hops() const { return (data >> EW_HOPS_SHIFT) & EW_HOPS_MASK; }
    uint8_t get_ns_direction() const { return (data >> NS_DIR_SHIFT) & NS_DIR_MASK; }
    uint8_t get_ew_direction() const { return (data >> EW_DIR_SHIFT) & EW_DIR_MASK; }
    uint8_t get_turn_point() const { return (data >> TURN_POINT_SHIFT) & TURN_POINT_MASK; }
#endif
};

static_assert(sizeof(compressed_route_2d_t) == 4, "2D route must be 4 bytes");

// ============================================================================
// Dynamic Packet Header Configuration
// ============================================================================

// Centralized build-time configuration for packet headers
struct FabricHeaderConfig {
    // 1D Routing Configuration
#ifdef FABRIC_1D_PKT_HDR_EXTENSION_WORDS
    static constexpr uint32_t LOW_LATENCY_EXTENSION_WORDS = FABRIC_1D_PKT_HDR_EXTENSION_WORDS;
#else
    // Default for host compilation or if not specified (Backward Compatibility)
    static constexpr uint32_t LOW_LATENCY_EXTENSION_WORDS = 1;
#endif

    // Derived Constants (Centralized Logic)
    static constexpr uint32_t LOW_LATENCY_NUM_WORDS = 1 + LOW_LATENCY_EXTENSION_WORDS;

    // 2D Routing Configuration
#ifdef FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE
    static constexpr uint32_t MESH_ROUTE_BUFFER_SIZE = FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE;
#else
    // Default: 35 bytes (96B header)
    static constexpr uint32_t MESH_ROUTE_BUFFER_SIZE = 35;
#endif

    // Validation (Fail fast)
    static_assert(LOW_LATENCY_EXTENSION_WORDS <= 3, "Only supports up to 3 extension words (64 hops)");
};

// Centralized routing field constants (single source of truth)
struct RoutingFieldsConstants {
    // 1D Constants (Low Latency)
    struct LowLatency {
        static constexpr uint32_t FIELD_WIDTH = 2;
        static constexpr uint32_t FIELD_MASK = 0b11;
        static constexpr uint32_t NOOP = 0b00;
        static constexpr uint32_t WRITE_ONLY = 0b01;
        static constexpr uint32_t FORWARD_ONLY = 0b10;
        static constexpr uint32_t WRITE_AND_FORWARD = 0b11;
        static constexpr uint32_t BASE_HOPS = 16;               // Hops per 32-bit word
        static constexpr uint32_t FWD_ONLY_FIELD = 0xAAAAAAAA;  // 32-bit pattern (all FORWARD_ONLY)
        static constexpr uint32_t WR_ONLY_FIELD = 0x55555555;   // 32-bit pattern (all WRITE_ONLY)
    };

    // 2D Constants (Mesh)
    struct Mesh {
        static constexpr uint32_t FIELD_WIDTH = 8;    // 8 bits per hop command
        static constexpr uint32_t FIELD_MASK = 0b1111;  // 4-bit mask

        // Basic direction commands (4-bit encoding for each direction)
        static constexpr uint8_t NOOP = 0b0000;
        static constexpr uint8_t FORWARD_EAST = 0b0001;
        static constexpr uint8_t FORWARD_WEST = 0b0010;
        static constexpr uint8_t FORWARD_NORTH = 0b0100;
        static constexpr uint8_t FORWARD_SOUTH = 0b1000;

        // Multicast combinations (OR of direction bits for write-and-forward)
        static constexpr uint8_t WRITE_AND_FORWARD_EW = FORWARD_EAST | FORWARD_WEST;    // 0b0011
        static constexpr uint8_t WRITE_AND_FORWARD_NS = FORWARD_NORTH | FORWARD_SOUTH;  // 0b1100
        static constexpr uint8_t WRITE_AND_FORWARD_NE = FORWARD_NORTH | FORWARD_EAST;   // 0b0101
        static constexpr uint8_t WRITE_AND_FORWARD_NW = FORWARD_NORTH | FORWARD_WEST;   // 0b0110
        static constexpr uint8_t WRITE_AND_FORWARD_SE = FORWARD_SOUTH | FORWARD_EAST;   // 0b1001
        static constexpr uint8_t WRITE_AND_FORWARD_SW = FORWARD_SOUTH | FORWARD_WEST;   // 0b1010
        static constexpr uint8_t WRITE_AND_FORWARD_NEW = FORWARD_NORTH | WRITE_AND_FORWARD_EW;          // 0b0111
        static constexpr uint8_t WRITE_AND_FORWARD_SEW = FORWARD_SOUTH | WRITE_AND_FORWARD_EW;          // 0b1011
        static constexpr uint8_t WRITE_AND_FORWARD_NSE = WRITE_AND_FORWARD_NS | FORWARD_EAST;           // 0b1101
        static constexpr uint8_t WRITE_AND_FORWARD_NSW = WRITE_AND_FORWARD_NS | FORWARD_WEST;           // 0b1110
        static constexpr uint8_t WRITE_AND_FORWARD_NSEW = WRITE_AND_FORWARD_NS | WRITE_AND_FORWARD_EW;  // 0b1111
    };
};

// Centralized routing encoding functions (stateless, buffer-based primitives)
namespace routing_encoding {

//=============================================================================
// 1D Routing Encoders
//=============================================================================

/**
 * Canonical 1D unicast routing pattern encoder
 *
 * Generates bit pattern where:
 *   - Each hop uses 2 bits (FIELD_WIDTH = 2)
 *   - FORWARD_ONLY (0b10) for transit hops
 *   - WRITE_ONLY (0b01) for final hop
 *
 * @param num_hops Number of hops (0 = self-route, 1-32 supported)
 * @param buffer Output buffer (uint32_t array)
 *        buffer[0] = value (active routing field)
 *        buffer[1..n] = route_buffer entries (if num_words > 1)
 * @param num_words Size of buffer (1 for ≤16 hops, 2 for ≤32 hops)
 *
 * Example: 3 hops with num_words=1
 *   Hop 0 (bits 0-1): FORWARD_ONLY = 0b10
 *   Hop 1 (bits 2-3): FORWARD_ONLY = 0b10
 *   Hop 2 (bits 4-5): WRITE_ONLY = 0b01
 *   Result: buffer[0] = 0b01'10'10 = 0x1A
 *
 * Router consumes fields LSB-first (hop 0 at bits 0-1, hop 1 at bits 2-3, etc.)
 */
inline void encode_1d_unicast(uint8_t num_hops, uint32_t* buffer, uint32_t num_words) {
    using LowLatencyFields = RoutingFieldsConstants::LowLatency;

    // Zero-initialize
    for (uint32_t i = 0; i < num_words; i++) {
        buffer[i] = 0;
    }

    if (num_hops == 0) {
        return;  // Self-route
    }

    // Logic: FWD_ONLY for (hops-1), then WRITE_ONLY
    const uint32_t write_hop_index = num_hops - 1;
    const uint32_t write_word_index = write_hop_index / LowLatencyFields::BASE_HOPS;
    const uint32_t write_bit_pos = (write_hop_index % LowLatencyFields::BASE_HOPS) * LowLatencyFields::FIELD_WIDTH;

    const uint32_t forward_mask = (1U << write_bit_pos) - 1;
    const uint32_t write_word_value =
        (LowLatencyFields::FWD_ONLY_FIELD & forward_mask) | (LowLatencyFields::WRITE_ONLY << write_bit_pos);

    for (uint32_t i = 0; i < num_words; i++) {
        if (i < write_word_index) {
            buffer[i] = LowLatencyFields::FWD_ONLY_FIELD;
        } else if (i == write_word_index) {
            buffer[i] = write_word_value;
        }
    }
}

/**
 * Canonical 1D multicast routing pattern encoder
 *
 * Generates bit pattern for multicast routing:
 *   - FORWARD_ONLY (0b10) before range
 *   - WRITE_AND_FORWARD (0b11) within range
 *   - WRITE_ONLY (0b01) at final hop
 *
 * @param start_hop First hop to start writing (1-indexed)
 * @param range_hops Number of hops in multicast range
 * @param buffer Output buffer (uint32_t array)
 * @param num_words Size of buffer (1 for ≤16 hops, 2 for ≤32 hops)
 *
 * Example: starting 3 hops away, multicasting to 2 chips (start_hop=3, range_hops=2)
 *   Hop 0 (bits 0-1): FORWARD_ONLY = 0b10
 *   Hop 1 (bits 2-3): FORWARD_ONLY = 0b10
 *   Hop 2 (bits 4-5): WRITE_AND_FORWARD = 0b11 (start of multicast range)
 *   Hop 3 (bits 6-7): WRITE_ONLY = 0b01 (end of range)
 *   Result: buffer[0] = 0b01'11'10'10 = 0x7A
 *
 * Router consumes fields LSB-first (hop 0 at bits 0-1, hop 1 at bits 2-3, etc.)
 */
inline void encode_1d_multicast(uint8_t start_hop, uint8_t range_hops, uint32_t* buffer, uint32_t num_words) {
    using LowLatencyFields = RoutingFieldsConstants::LowLatency;

    for (uint32_t i = 0; i < num_words; i++) {
        buffer[i] = 0;
    }

    // Last hop in the multicast range (inclusive, may be negative if range_hops == 0)
    //
    // Multicast pattern (start_hop=3, range_hops=2 example):
    //   Hop index:  0    1    2    3    4   ...
    //   Action:     FWD  FWD  W+F  WR   -
    //                          X----X           <- multicast range (writes to 2 chips)
    //                          ^    ^
    //                       start   last_hop
    //
    // Calculation: start_hop is 1-indexed -> convert to 0-indexed (hop 2)
    //              Add range_hops to get end position, then -1 for inclusive last index
    //              (3-1) + 2 - 1 = 3, simplified: (start_hop + range_hops) - 2
    const int last_hop = static_cast<int>(start_hop + range_hops) - 2;

    auto set_hop_field = [&](uint32_t hop_index, uint32_t field_value) {
        const uint32_t word_idx = hop_index / LowLatencyFields::BASE_HOPS;

        // Bounds check (replaces constexpr check from original method)
        if (word_idx < num_words) {
            const uint32_t bit_pos = (hop_index % LowLatencyFields::BASE_HOPS) * LowLatencyFields::FIELD_WIDTH;
            buffer[word_idx] |= (field_value << bit_pos);
        }
    };

    // 1. Prefix: Forward to start
    for (int hop = 0; hop < static_cast<int>(start_hop) - 1; hop++) {
        set_hop_field(hop, LowLatencyFields::FORWARD_ONLY);
    }

    // 2. Range: Write & Forward (for range_hops - 1 hops)
    for (int hop = static_cast<int>(start_hop) - 1; hop < last_hop; hop++) {
        set_hop_field(hop, LowLatencyFields::WRITE_AND_FORWARD);
    }

    // 3. Tail: Write Only (only if we have a valid last hop)
    if (last_hop >= 0) [[likely]] {
        set_hop_field(last_hop, LowLatencyFields::WRITE_ONLY);
    }
}

//=============================================================================
// 2D Routing Encoders
//=============================================================================

/**
 * Canonical 2D unicast routing pattern encoder
 *
 * Handles NS -> EW routing with proper write command selection.
 *
 * This matches the existing decode_route_to_buffer logic (fabric_routing_path_interface.h lines 45-75):
 * - Final hop uses opposite-direction bit (no forward) to stop packet
 * - If both NS and EW exist: emit (ns_hops - 1) NS forwards, then ew_hops EW forwards, then 1 EW opposite
 * - If only NS: emit (ns_hops - 1) NS forwards, then 1 NS opposite
 * - If only EW: emit (ew_hops - 1) EW forwards, then 1 EW opposite
 *
 * Example: Traveling South 2 hops, then East 1 hop:
 *   - Forward South (0b1000), Forward South (0b1000), Forward East (0b0001), Write North (0b0100 - opposite)
 *   - Final North bit stops the packet at destination
 *
 * @param ns_hops Number of North/South hops
 * @param ew_hops Number of East/West hops
 * @param ns_dir North/South direction (0=North, 1=South)
 * @param ew_dir East/West direction (0=West, 1=East)
 * @param buffer Output buffer (uint8_t array)
 * @param max_buffer_size Size of buffer (8/16/24/32 bytes)
 * @param prepend_one_hop If true, adds one extra forward hop at the start (used by routers)
 */
inline void encode_2d_unicast(
    uint8_t ns_hops,
    uint8_t ew_hops,
    uint8_t ns_dir,
    uint8_t ew_dir,
    uint8_t* buffer,
    uint32_t max_buffer_size,
    bool prepend_one_hop = false) {
    using MeshFields = RoutingFieldsConstants::Mesh;
    uint32_t idx = 0;

    // Forward commands based on direction
    const uint8_t ns_fwd = (ns_dir == 1) ? MeshFields::FORWARD_SOUTH : MeshFields::FORWARD_NORTH;
    const uint8_t ew_fwd = (ew_dir == 1) ? MeshFields::FORWARD_EAST : MeshFields::FORWARD_WEST;

    // Final hop uses OPPOSITE direction to stop packet (destination logic)
    // If traveling South (ns_dir=1), final command is North (opposite)
    // If traveling East (ew_dir=1), final command is West (opposite)
    const uint8_t ns_write = (ns_dir == 1) ? MeshFields::FORWARD_NORTH : MeshFields::FORWARD_SOUTH;
    const uint8_t ew_write = (ew_dir == 1) ? MeshFields::FORWARD_WEST : MeshFields::FORWARD_EAST;

    if (ns_hops > 0 && ew_hops > 0) {
        // NS -> EW turn: (ns_hops-1 + prepend) NS forwards, ew_hops EW forwards, 1 EW write
        for (auto i = 0; i < ns_hops - 1 + prepend_one_hop; ++i) {
            buffer[idx++] = ns_fwd;
        }
        for (auto i = 0; i < ew_hops; ++i) {
            buffer[idx++] = ew_fwd;
        }
        buffer[idx++] = ew_write;
    } else if (ns_hops > 0) {
        // Only NS: (ns_hops-1 + prepend) NS forwards, 1 NS write
        for (auto i = 0; i < ns_hops - 1 + prepend_one_hop; ++i) {
            buffer[idx++] = ns_fwd;
        }
        buffer[idx++] = ns_write;
    } else if (ew_hops > 0) {
        // Only EW: (ew_hops-1 + prepend) EW forwards, 1 EW write
        for (auto i = 0; i < ew_hops - 1 + prepend_one_hop; ++i) {
            buffer[idx++] = ew_fwd;
        }
        buffer[idx++] = ew_write;
    }

    // Fill remainder with NOOP
    while (idx < max_buffer_size) {
        buffer[idx++] = MeshFields::NOOP;
    }
}

}  // namespace routing_encoding

// ============================================================================

// Number of routing table entries (destinations), not hops.
// For 4×64 mesh: 64 entries, each storing a route up to 63 hops long.
static const uint16_t MAX_CHIPS_LOWLAT_1D = 64;
static const uint16_t MAX_CHIPS_LOWLAT_2D = 256;
// Size of each routing table entry in bytes
static const uint16_t SINGLE_ROUTE_SIZE_1D = 16;  // 4 words for 64 hops: base + 3 extension words
static const uint16_t SINGLE_ROUTE_SIZE_2D = 32;

template <uint8_t dim, bool compressed>
struct __attribute__((packed)) intra_mesh_routing_path_t {
    static_assert(dim == 1 || dim == 2, "dim must be 1 or 2");

    // Compressed routing uses much smaller encoding
    // 1D: 0 byte (num_hops passed from caller is the compressed info)
    static const uint16_t COMPRESSED_ROUTE_SIZE_1D = 0;
    // 2D: 2 bytes (ns_hops:5bits, ew_hops:5bits, ns_dir:1bit, ew_dir:1bit, turn_point:4bits)
    static const uint16_t COMPRESSED_ROUTE_SIZE_2D = sizeof(compressed_route_2d_t);

    static constexpr uint16_t MAX_CHIPS_LOWLAT = (dim == 1) ? MAX_CHIPS_LOWLAT_1D : MAX_CHIPS_LOWLAT_2D;
    static constexpr uint16_t COMPRESSED_ROUTE_SIZE = (dim == 1) ? COMPRESSED_ROUTE_SIZE_1D : COMPRESSED_ROUTE_SIZE_2D;
    static constexpr uint16_t UNCOMPRESSED_ROUTE_SIZE = (dim == 1) ? SINGLE_ROUTE_SIZE_1D : SINGLE_ROUTE_SIZE_2D;
    static constexpr uint16_t SINGLE_ROUTE_SIZE = compressed ? COMPRESSED_ROUTE_SIZE : UNCOMPRESSED_ROUTE_SIZE;

    std::conditional_t<
        !compressed,
        std::uint8_t[MAX_CHIPS_LOWLAT * SINGLE_ROUTE_SIZE],  // raw for uncompressed
        std::conditional_t<
            dim == 1,
            std::uint8_t[0],                         // empty for compressed 1D (0 bytes)
            compressed_route_2d_t[MAX_CHIPS_LOWLAT]  // two for compressed 2D
            >>
        paths = {};

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    // Routing calculation methods
    void calculate_chip_to_all_routing_fields(const FabricNodeId& src_fabric_node_id, uint16_t num_chips);
#else
    // Device-side methods (declared here, implemented in fabric_routing_path_interface.h):
    inline bool decode_route_to_buffer(
        uint16_t dst_chip_id, volatile uint8_t* out_route_buffer, bool prepend_one_hop = false) const;
#endif
};

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

struct routing_l1_info_t {
    // TODO: https://github.com/tenstorrent/tt-metal/issues/28534
    //       these fabric node ids should be another struct as really commonly used data
    uint16_t my_mesh_id = 0;    // Current mesh ID
    uint16_t my_device_id = 0;  // Current chip ID
    // NOTE: Compressed version has additional overhead (2x slower) to read values,
    //       but raw data is too huge (2048 bytes) to fit in L1 memory.
    //       Need to evaluate once actual workloads are available
    direction_table_t<MAX_MESH_SIZE> intra_mesh_direction_table{};   // 96 bytes
    direction_table_t<MAX_NUM_MESHES> inter_mesh_direction_table{};  // 384 bytes

    // Union overlaps 1D and 2D routing tables at same offset
    union __attribute__((packed)) {
        intra_mesh_routing_path_t<1, false> routing_path_table_1d;  // 1024 bytes
        intra_mesh_routing_path_t<2, true> routing_path_table_2d;   // 1024 bytes
    };

    std::uint8_t exit_node_table[MAX_NUM_MESHES] = {};               // 1024 bytes
    uint8_t padding[12] = {};                                        // pad to 16-byte alignment
} __attribute__((packed));

// 64 chips * 16 bytes = 1024
static_assert(
    sizeof(intra_mesh_routing_path_t<1, false>) == 1024,
    "1D uncompressed routing path must be 1024 bytes (64 entries x 16 bytes per route)");

static_assert(sizeof(intra_mesh_routing_path_t<1, true>) == 0, "1D compressed routing path must be 0 bytes");

// 256 chips * 4 bytes = 1024
static_assert(
    sizeof(intra_mesh_routing_path_t<2, true>) == 1024,
    "2D compressed routing path must be 1024 bytes (256 entries x 4 bytes per route)");

// Verify total struct size
static_assert(
    sizeof(routing_l1_info_t) == 2544,
    "routing_l1_info_t must be 2544 bytes: base(484) + union(1024) + exit(1024) + pad(12)");

struct worker_routing_l1_info_t {
    routing_l1_info_t routing_info{};
    tensix_fabric_connections_l1_info_t fabric_connections{};
};

struct fabric_routing_l1_info_t {
    routing_l1_info_t routing_info;
};

// Fabric connection synchronization region in L1
// Used for multi-RISC synchronization when opening fabric connections
// Memory layout: [lock(4) | initialized(4) | connection_object(128) | padding(8)] = 144 bytes
struct fabric_connection_sync_t {
    uint32_t lock;         // Spinlock for mutual exclusion (0 = unlocked, 1 = locked)
    uint32_t initialized;  // Flag indicating if fabric connection has been initialized (0 = not initialized, 1 =
                           // initialized)
    // Connection object storage follows at offset 8 (accessed via address calculation)
};
static_assert(sizeof(fabric_connection_sync_t) == 8, "fabric_connection_sync_t must be 8 bytes");

// Offset to connection object storage within the sync region
static constexpr uint32_t FABRIC_CONNECTION_OBJECT_OFFSET = 8;
// Size reserved for WorkerToFabricEdmSender object (verified by static_assert in tt_fabric_udm_impl.hpp)
static constexpr uint32_t FABRIC_CONNECTION_OBJECT_SIZE = 128;

}  // namespace tt::tt_fabric

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#if defined(COMPILE_FOR_ERISC)
#define ROUTING_PATH_BASE MEM_AERISC_FABRIC_ROUTING_PATH_BASE
#define ROUTING_PATH_BASE_1D MEM_AERISC_FABRIC_ROUTING_PATH_BASE_1D
#define ROUTING_PATH_BASE_2D MEM_AERISC_FABRIC_ROUTING_PATH_BASE_2D
#define ROUTING_TABLE_BASE MEM_AERISC_ROUTING_TABLE_BASE
#define EXIT_NODE_TABLE_BASE MEM_AERISC_EXIT_NODE_TABLE_BASE
#elif defined(COMPILE_FOR_IDLE_ERISC)
#define ROUTING_PATH_BASE MEM_IERISC_FABRIC_ROUTING_PATH_BASE
#define ROUTING_PATH_BASE_1D MEM_IERISC_FABRIC_ROUTING_PATH_BASE_1D
#define ROUTING_PATH_BASE_2D MEM_IERISC_FABRIC_ROUTING_PATH_BASE_2D
#define ROUTING_TABLE_BASE MEM_IERISC_ROUTING_TABLE_BASE
#define EXIT_NODE_TABLE_BASE MEM_IERISC_EXIT_NODE_TABLE_BASE
#else
#define ROUTING_PATH_BASE MEM_TENSIX_ROUTING_PATH_BASE
#define ROUTING_PATH_BASE_1D MEM_TENSIX_ROUTING_PATH_BASE_1D
#define ROUTING_PATH_BASE_2D MEM_TENSIX_ROUTING_PATH_BASE_2D
#define ROUTING_TABLE_BASE MEM_TENSIX_ROUTING_TABLE_BASE
#define EXIT_NODE_TABLE_BASE MEM_TENSIX_EXIT_NODE_TABLE_BASE
#endif

#include "fabric/hw/inc/fabric_direction_table_interface.h"
#include "fabric/hw/inc/fabric_routing_path_interface.h"
#endif
