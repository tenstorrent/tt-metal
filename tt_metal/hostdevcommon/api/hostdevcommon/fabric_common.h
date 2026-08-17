// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <type_traits>

#include "tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h"

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

// Sentinel stream-register value marking an inactive consumer, shared by host emission (which
// writes it into the named CT args for any consumer absent in a configuration) and the kernel's
// init path (which skips initialising any entry reading it). Deliberately out of the 0..31
// register range: no real register can ever collide with it, so every id in the file is ordinary.
static constexpr uint32_t k_unused_stream_id = 32;

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
        static constexpr uint32_t FIELD_WIDTH = 8;       // 8 bits per hop command
        static constexpr uint32_t FIELD_MASK = 0b1111;   // 4-bit mask

        // Basic direction commands (bit-per-direction encoding, matching eth_chan_directions)
        static constexpr uint8_t NOOP = 0b00000;
        static constexpr uint8_t FORWARD_EAST = 0b00001;
        static constexpr uint8_t FORWARD_WEST = 0b00010;
        static constexpr uint8_t FORWARD_NORTH = 0b00100;
        static constexpr uint8_t FORWARD_SOUTH = 0b01000;

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

// ============================================================================
// Indexed 2D route codec (destination-indexed ABI)
// ============================================================================
// Target ABI for 2D mesh routing. L1 holds
// destination-major, per-axis 2-bit action vectors (y_vectors[dst_y][cur_y], x_vectors[dst_x][cur_x]);
// packets carry the widened form: one action byte per logical coordinate, route_buffer_y[Y]
// immediately followed by route_buffer_x[X]. The control plane generates the 2-bit tables; workers
// widen them into packets at setup; the router kernel decodes action bytes using its compile-time
// coordinate.
struct IndexedMeshRoutingFields {
    // ---- Packet action byte -------------------------------------------------
    // One-hot per output port; bits 0..4 intentionally match eth_chan_directions so the action bit
    // for a direction is (1 << direction). Bit 5 requests local delivery at the current chip.
    // Bits 6..7 are reserved and must be zero (kernel fail-stops otherwise).
    static constexpr uint8_t ACTION_EAST = 0b00000001;
    static constexpr uint8_t ACTION_WEST = 0b00000010;
    static constexpr uint8_t ACTION_NORTH = 0b00000100;
    static constexpr uint8_t ACTION_SOUTH = 0b00001000;
    static constexpr uint8_t ACTION_Z = 0b00010000;
    static constexpr uint8_t ACTION_LOCAL_DELIVER = 0b00100000;
    static constexpr uint8_t ACTION_ETH_MASK = 0b00011111;
    static constexpr uint8_t ACTION_VALID_MASK = ACTION_ETH_MASK | ACTION_LOCAL_DELIVER;
    static constexpr uint8_t ACTION_RESERVED_MASK = 0b11000000;

    static constexpr uint8_t action_bit(eth_chan_directions dir) { return static_cast<uint8_t>(1u << dir); }

    // ---- 2-bit L1 vector encodings ------------------------------------------
    // Y axis: STOP means the destination Y is reached (or the row is not traversed); Z is the
    // intra-mesh express (skip) link.
    static constexpr uint8_t Y2_STOP = 0;
    static constexpr uint8_t Y2_NORTH = 1;
    static constexpr uint8_t Y2_SOUTH = 2;
    static constexpr uint8_t Y2_Z = 3;
    // X axis: encoding 3 has no meaning on X (no express dimension) and is reserved-invalid.
    static constexpr uint8_t X2_STOP = 0;
    static constexpr uint8_t X2_EAST = 1;
    static constexpr uint8_t X2_WEST = 2;
    static constexpr uint8_t X2_INVALID = 3;

    // ---- Packed-row helpers ---------------------------------------------------
    // Rows hold ceil(axis/4) bytes, 4 entries per byte, entry 0 at the LSBs of byte 0.
    static constexpr uint32_t BITS_PER_ACTION = 2;
    static constexpr uint32_t ACTIONS_PER_BYTE = 4;

    static constexpr uint32_t row_bytes(uint32_t axis_size) {
        return (axis_size + ACTIONS_PER_BYTE - 1) / ACTIONS_PER_BYTE;
    }
    // Destination-major table footprint: one packed row per destination coordinate.
    static constexpr uint32_t table_bytes(uint32_t axis_size) { return axis_size * row_bytes(axis_size); }

    static inline uint8_t get_action_2bit(const std::uint8_t* packed_row, uint32_t index) {
        const uint32_t byte_index = index / ACTIONS_PER_BYTE;
        const uint32_t shift = (index % ACTIONS_PER_BYTE) * BITS_PER_ACTION;
        return static_cast<uint8_t>((packed_row[byte_index] >> shift) & 0b11);
    }
    static inline void set_action_2bit(std::uint8_t* packed_row, uint32_t index, uint8_t action_2bit) {
        const uint32_t byte_index = index / ACTIONS_PER_BYTE;
        const uint32_t shift = (index % ACTIONS_PER_BYTE) * BITS_PER_ACTION;
        packed_row[byte_index] =
            static_cast<std::uint8_t>((packed_row[byte_index] & ~(0b11u << shift)) | ((action_2bit & 0b11u) << shift));
    }

    // ---- Widen (2-bit -> one-hot action byte) -----------------------------------
    // Used by workers (packet setup) and host generation/validation. STOP and X2_INVALID widen to 0
    // (no one-hot output); the caller pokes ACTION_LOCAL_DELIVER at its own coordinate afterwards.
    static constexpr uint8_t widen_y(uint8_t action_2bit) {
        switch (action_2bit) {
            case Y2_NORTH: return ACTION_NORTH;
            case Y2_SOUTH: return ACTION_SOUTH;
            case Y2_Z: return ACTION_Z;
            default: return 0;
        }
    }
    static constexpr uint8_t widen_x(uint8_t action_2bit) {
        switch (action_2bit) {
            case X2_EAST: return ACTION_EAST;
            case X2_WEST: return ACTION_WEST;
            default: return 0;
        }
    }

    // ---- L1 region sizing -------------------------------------------------------
    // Compatibility bound [Y,X] = [64,4]: Y table 64*16 = 1024 B, X table 4*1 = 4 B, total 1028 B
    // (reuses the legacy 1024 B 2D union slot plus 4 B of trailing padding). Actual
    // tables are sized by the live mesh shape via table_bytes().
    static constexpr uint32_t MAX_INDEXED_MESH_Y = 64;
    static constexpr uint32_t MAX_INDEXED_MESH_X = 4;
    // Expanded inline: Clang does not treat in-class constexpr member functions as defined for
    // constant evaluation within the class body, so table_bytes() cannot be called here.
    static constexpr uint32_t INDEXED_VECTOR_TABLE_BYTES =
        MAX_INDEXED_MESH_Y * ((MAX_INDEXED_MESH_Y + ACTIONS_PER_BYTE - 1) / ACTIONS_PER_BYTE) +
        MAX_INDEXED_MESH_X * ((MAX_INDEXED_MESH_X + ACTIONS_PER_BYTE - 1) / ACTIONS_PER_BYTE);  // 1028

    // ---- Pack (host-side table generation) --------------------------------------
    // Fills the Y region [0, table_bytes(y_size)) followed by
    // the X region [table_bytes(y_size), vectors_region_bytes(...)), destination-major packed rows.
    // Action sources map (cur, dst) on their axis to an eth_chan_directions value; the diagonal is
    // always STOP and is never queried. Returns false on a non-representable action (X has no code
    // besides E/W/STOP and X2_INVALID must never reach L1) or on a shape exceeding the [64,4] bound.
    static constexpr uint32_t vectors_region_bytes(uint32_t y_size, uint32_t x_size) {
        return table_bytes(y_size) + table_bytes(x_size);
    }

    // ---- Multicast reverse-tree region (contract 5.7.1 storage, 6.2 placement) --
    // The hybrid layout: the destination-major vectors, then this chip's own reverse trees. The
    // vectors are mesh-identical, the trees are not -- each chip carries only T(its own row) and
    // T(its own column), so this region's contents differ per chip while its geometry does not.
    //
    // Placed at a shape-derived offset rather than a fixed one because the vectors are packed to the
    // live shape, so a fixed offset would either waste the [64,4] bound on every mesh or collide.
    // Both sides compute the offset the same way from the shape they already hold.
    static constexpr uint32_t MCAST_TREE_EDGE_BYTES = 2;
    // An arborescence over n rows has n-1 edges; a single-row axis has none.
    static constexpr uint32_t mcast_tree_edge_count(uint32_t axis_size) { return axis_size > 1 ? axis_size - 1 : 0; }
    static constexpr uint32_t mcast_tree_region_bytes(uint32_t y_size, uint32_t x_size) {
        return MCAST_TREE_EDGE_BYTES * (mcast_tree_edge_count(y_size) + mcast_tree_edge_count(x_size));
    }
    static constexpr uint32_t mcast_tree_offset_bytes(uint32_t y_size, uint32_t x_size) {
        return (vectors_region_bytes(y_size, x_size) + 3u) & ~3u;
    }
    // Whether vectors plus trees fit the existing union slot. False is not a defect: [64,4] needs the
    // struct growth recorded in contract 6.2, and the caller is expected to say so rather than pack
    // over the end of the slot.
    static constexpr bool hybrid_region_fits(uint32_t y_size, uint32_t x_size) {
        return mcast_tree_offset_bytes(y_size, x_size) + mcast_tree_region_bytes(y_size, x_size) <=
               INDEXED_VECTOR_TABLE_BYTES;
    }

    static constexpr uint32_t mcast_tree_y_offset(uint32_t y_size, uint32_t x_size) {
        return mcast_tree_offset_bytes(y_size, x_size);
    }
    static constexpr uint32_t mcast_tree_x_offset(uint32_t y_size, uint32_t x_size) {
        return mcast_tree_offset_bytes(y_size, x_size) + MCAST_TREE_EDGE_BYTES * mcast_tree_edge_count(y_size);
    }

    // Packed edge fields (contract 5.7.1). The Y <= 64 bound is what fixes the two 6-bit row fields;
    // parent_output holds this axis's 2-bit vector code, the same encoding a unicast vector uses, so
    // it widens through widen_y / widen_x rather than a private table.
    static constexpr int mcast_edge_child(std::uint16_t packed) { return packed & 0x3F; }
    static constexpr int mcast_edge_parent(std::uint16_t packed) { return (packed >> 6) & 0x3F; }
    static constexpr std::uint8_t mcast_edge_output(std::uint16_t packed) {
        return static_cast<std::uint8_t>((packed >> 12) & 0x3);
    }

    // Byte-wise so neither side has to reason about halfword alignment inside a packed union.
    static inline std::uint16_t get_mcast_tree_edge(const std::uint8_t* region, uint32_t index) {
        const uint32_t byte_index = index * MCAST_TREE_EDGE_BYTES;
        return static_cast<std::uint16_t>(
            region[byte_index] | (static_cast<std::uint16_t>(region[byte_index + 1]) << 8));
    }
    static inline void set_mcast_tree_edge(std::uint8_t* region, uint32_t index, std::uint16_t packed_edge) {
        const uint32_t byte_index = index * MCAST_TREE_EDGE_BYTES;
        region[byte_index] = static_cast<std::uint8_t>(packed_edge & 0xFF);
        region[byte_index + 1] = static_cast<std::uint8_t>((packed_edge >> 8) & 0xFF);
    }

    static inline std::uint8_t* y_row(std::uint8_t* table, uint32_t y_size, uint32_t dst_y) {
        return table + dst_y * row_bytes(y_size);
    }
    static inline const std::uint8_t* y_row(const std::uint8_t* table, uint32_t y_size, uint32_t dst_y) {
        return table + dst_y * row_bytes(y_size);
    }
    static inline std::uint8_t* x_row(std::uint8_t* table, uint32_t y_size, uint32_t x_size, uint32_t dst_x) {
        return table + table_bytes(y_size) + dst_x * row_bytes(x_size);
    }
    static inline const std::uint8_t* x_row(
        const std::uint8_t* table, uint32_t y_size, uint32_t x_size, uint32_t dst_x) {
        return table + table_bytes(y_size) + dst_x * row_bytes(x_size);
    }

    template <typename YActionSource, typename XActionSource>
    static inline bool pack_indexed_route_vectors(
        std::uint8_t* out, uint32_t y_size, uint32_t x_size, YActionSource&& y_action, XActionSource&& x_action) {
        if (y_size > MAX_INDEXED_MESH_Y || x_size > MAX_INDEXED_MESH_X ||
            vectors_region_bytes(y_size, x_size) > INDEXED_VECTOR_TABLE_BYTES) {
            return false;
        }
        const uint32_t region_bytes = vectors_region_bytes(y_size, x_size);
        for (uint32_t i = 0; i < region_bytes; ++i) {
            out[i] = 0;
        }
        for (uint32_t dst = 0; dst < y_size; ++dst) {
            std::uint8_t* row = y_row(out, y_size, dst);
            for (uint32_t cur = 0; cur < y_size; ++cur) {
                uint8_t action = Y2_STOP;
                if (cur != dst) {
                    switch (y_action(cur, dst)) {
                        case eth_chan_directions::NORTH: action = Y2_NORTH; break;
                        case eth_chan_directions::SOUTH: action = Y2_SOUTH; break;
                        case eth_chan_directions::Z: action = Y2_Z; break;
                        default: return false;
                    }
                }
                set_action_2bit(row, cur, action);
            }
        }
        for (uint32_t dst = 0; dst < x_size; ++dst) {
            std::uint8_t* row = x_row(out, y_size, x_size, dst);
            for (uint32_t cur = 0; cur < x_size; ++cur) {
                uint8_t action = X2_STOP;
                if (cur != dst) {
                    switch (x_action(cur, dst)) {
                        case eth_chan_directions::EAST: action = X2_EAST; break;
                        case eth_chan_directions::WEST: action = X2_WEST; break;
                        default: return false;
                    }
                }
                set_action_2bit(row, cur, action);
            }
        }
        return true;
    }

    // ---- Decode (packet-side action selection) -----------------------------------
    // Packet decode semantics: the router at logical (local_y, local_x) reads
    // its action byte from the packet's flat [Y | X] route buffer (route_buffer[0..y_size) = Y actions,
    // route_buffer[y_size..y_size+x_size) = X actions). E/W-facing routers consume X only; N/S/Z-facing
    // routers consume Y whenever the COMPLETE Y byte is nonzero — a set LOCAL_DELIVER or E/W tooth bit
    // must not fall through to X — and X otherwise. Transit performs no divide/mod and no L1 vector
    // reads: coordinates are the kernel's cached locals and y_size is its compile-time mesh shape.
    template <eth_chan_directions MY_DIR>
    static inline std::uint8_t decode_action(
        const volatile std::uint8_t* route_buffer, std::uint32_t local_y, std::uint32_t local_x, std::uint32_t y_size) {
        if constexpr (MY_DIR == eth_chan_directions::EAST || MY_DIR == eth_chan_directions::WEST) {
            return route_buffer[y_size + local_x];
        } else {
            const std::uint8_t action_y = route_buffer[local_y];
            if (action_y != 0) {
                return action_y;
            }
            return route_buffer[y_size + local_x];
        }
    }

    // The four non-self eth outputs available to a router facing MY_DIR, in the fixed slot order of the
    // packed dispatch key: base order {E, W, N, S, Z} with the self direction removed.
    // The self direction is never a valid output — no same-link return path exists.
    template <eth_chan_directions MY_DIR>
    static constexpr std::array<eth_chan_directions, 4> fwd_dirs() {
        if constexpr (MY_DIR == eth_chan_directions::EAST) {
            return {
                eth_chan_directions::WEST,
                eth_chan_directions::NORTH,
                eth_chan_directions::SOUTH,
                eth_chan_directions::Z};
        } else if constexpr (MY_DIR == eth_chan_directions::WEST) {
            return {
                eth_chan_directions::EAST,
                eth_chan_directions::NORTH,
                eth_chan_directions::SOUTH,
                eth_chan_directions::Z};
        } else if constexpr (MY_DIR == eth_chan_directions::NORTH) {
            return {
                eth_chan_directions::EAST,
                eth_chan_directions::WEST,
                eth_chan_directions::SOUTH,
                eth_chan_directions::Z};
        } else if constexpr (MY_DIR == eth_chan_directions::SOUTH) {
            return {
                eth_chan_directions::EAST,
                eth_chan_directions::WEST,
                eth_chan_directions::NORTH,
                eth_chan_directions::Z};
        } else {  // Z-facing: the express link lands into the local cardinal plane only
            return {
                eth_chan_directions::EAST,
                eth_chan_directions::WEST,
                eth_chan_directions::NORTH,
                eth_chan_directions::SOUTH};
        }
    }

    // Valid when reserved bits are clear, the self-facing bit is clear, and at least one output
    // (eth or LOCAL_DELIVER) is selected. A selected direction with no wired sender is rejected by
    // the kernel's dispatch, not by this predicate.
    template <eth_chan_directions MY_DIR>
    static constexpr bool action_is_valid(std::uint8_t action) {
        if (action & ACTION_RESERVED_MASK) {
            return false;
        }
        if (action & action_bit(MY_DIR)) {
            return false;
        }
        return action != 0;
    }

    // Pack the action's eth outputs through fwd_dirs<MY_DIR>() into the dense 4-bit dispatch key.
    // LOCAL_DELIVER is intentionally outside the key; the kernel handles it separately after the
    // eth fanout.
    template <eth_chan_directions MY_DIR>
    static constexpr std::uint8_t pack_fwd_key(std::uint8_t action) {
        constexpr auto dirs = fwd_dirs<MY_DIR>();
        std::uint8_t key = 0;
        for (std::uint32_t slot = 0; slot < 4; ++slot) {
            if (action & action_bit(dirs[slot])) {
                key = static_cast<std::uint8_t>(key | (1u << slot));
            }
        }
        return key;
    }

    // Intermesh exit test: the installed maps say "deliver here" (LOCAL_DELIVER
    // with no eth bits) but the packet's final mesh is elsewhere, so this chip is the mesh's selected
    // exit. Both halves are load-bearing — mesh-id inequality alone also matches packets merely
    // transiting a boundary-capable chip. The kernel evaluates this only on an
    // is_intramesh_router_on_edge channel, after decode and before the local-delivery fanout; an
    // intermesh carrier's installed maps always yield exactly LOCAL_DELIVER at the exit chip, so
    // the predicate is never ambiguous.
    static inline bool action_is_intermesh_exit(
        std::uint8_t action, std::uint16_t dst_mesh_id, std::uint16_t my_mesh_id) {
        return action == ACTION_LOCAL_DELIVER && dst_mesh_id != my_mesh_id;
    }
};

// Action bits 0..4 must line up with eth_chan_directions (E=0..Z=4): direction -> bit is (1 << dir).
static_assert(
    IndexedMeshRoutingFields::action_bit(eth_chan_directions::EAST) == IndexedMeshRoutingFields::ACTION_EAST,
    "indexed action bit mismatch for EAST");
static_assert(
    IndexedMeshRoutingFields::action_bit(eth_chan_directions::WEST) == IndexedMeshRoutingFields::ACTION_WEST,
    "indexed action bit mismatch for WEST");
static_assert(
    IndexedMeshRoutingFields::action_bit(eth_chan_directions::NORTH) == IndexedMeshRoutingFields::ACTION_NORTH,
    "indexed action bit mismatch for NORTH");
static_assert(
    IndexedMeshRoutingFields::action_bit(eth_chan_directions::SOUTH) == IndexedMeshRoutingFields::ACTION_SOUTH,
    "indexed action bit mismatch for SOUTH");
static_assert(
    IndexedMeshRoutingFields::action_bit(eth_chan_directions::Z) == IndexedMeshRoutingFields::ACTION_Z,
    "indexed action bit mismatch for Z");
static_assert(
    (IndexedMeshRoutingFields::ACTION_VALID_MASK | IndexedMeshRoutingFields::ACTION_RESERVED_MASK) == 0xFF &&
        (IndexedMeshRoutingFields::ACTION_VALID_MASK & IndexedMeshRoutingFields::ACTION_RESERVED_MASK) == 0,
    "indexed action byte valid/reserved masks must partition the byte");
static_assert(IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES == 1028, "indexed vector table must be 1028 B");

// Contract 6.2 hybrid footprints: [32,4] is 260 B of vectors plus a 68 B tree region and fits the
// existing slot; [64,4] is 1160 B and does not, which is the documented growth boundary.
static_assert(
    IndexedMeshRoutingFields::vectors_region_bytes(32, 4) == 260 &&
        IndexedMeshRoutingFields::mcast_tree_region_bytes(32, 4) == 68 &&
        IndexedMeshRoutingFields::hybrid_region_fits(32, 4),
    "[32,4] hybrid layout must fit the 2D union slot");
static_assert(
    !IndexedMeshRoutingFields::hybrid_region_fits(64, 4),
    "[64,4] hybrid layout is expected to exceed the slot until routing_l1_info_t grows (contract 6.2)");
static_assert(
    IndexedMeshRoutingFields::mcast_tree_x_offset(32, 4) == IndexedMeshRoutingFields::mcast_tree_y_offset(32, 4) + 62,
    "X tree must follow the Y tree's y_size-1 edges");

// ============================================================================
// Indexed multicast encode (contract 5.5 / 5.6 / 5.7.1)
// ============================================================================
// Shared by the worker producer and host validation so both run the identical arithmetic; the host
// test can then check it against the independent 5.6 path-trace reference without a device.
//
// No STL, no allocation, no Z-neighbor lookup: the reverse tree carries both endpoints and the
// parent's command, so encoding never follows a Z edge forward.

// Row bitmaps at the Y <= 64 bound.
inline constexpr std::uint32_t MCAST_ROW_BITS_WORDS = 2;

inline void mcast_set_row_bit(std::uint32_t* bits, std::uint32_t row) { bits[row >> 5] |= 1u << (row & 31); }
inline bool mcast_test_row_bit(const std::uint32_t* bits, std::uint32_t row) {
    return ((bits[row >> 5] >> (row & 31)) & 1u) != 0;
}

// One axis of the 5.7.1 reverse pass. `needed` enters holding the requested targets and leaves
// holding those plus every transit parent; an edge is taken exactly when its child subtree still
// holds something needed, which is what makes the result union(R(root, target)).
//
// LOCAL_DELIVER is deliberately not set here: 5.7.1 forbids deriving it from the expanded `needed`
// set, because a row can be needed purely as a transit parent.
inline void mcast_prune_axis(
    std::uint8_t* out_actions,
    const std::uint8_t* tree_region,
    std::uint32_t axis_len,
    std::uint32_t* needed,
    bool is_y_axis) {
    const std::uint32_t edge_count = IndexedMeshRoutingFields::mcast_tree_edge_count(axis_len);
    for (std::uint32_t i = 0; i < edge_count; ++i) {
        const std::uint16_t edge = IndexedMeshRoutingFields::get_mcast_tree_edge(tree_region, i);
        const std::uint32_t child = static_cast<std::uint32_t>(IndexedMeshRoutingFields::mcast_edge_child(edge));
        if (!mcast_test_row_bit(needed, child)) {
            continue;
        }
        const std::uint32_t parent = static_cast<std::uint32_t>(IndexedMeshRoutingFields::mcast_edge_parent(edge));
        const std::uint8_t code = IndexedMeshRoutingFields::mcast_edge_output(edge);
        out_actions[parent] |=
            is_y_axis ? IndexedMeshRoutingFields::widen_y(code) : IndexedMeshRoutingFields::widen_x(code);
        mcast_set_row_bit(needed, parent);
    }
}

// Fills route_buffer[0..y_size) with route_buffer_y and route_buffer[y_size..y_size+x_size) with
// route_buffer_x, from the reverse trees embedded in this chip's `vectors` table.
//
// Contract §5.0 keeps two chips apart that a same-mesh send lets you conflate:
//
//   anchor_{y,x}   the chip the client's N/S/E/W extents are measured from (range_anchor)
//   encode_root_x  the column where path tracing begins, which owns `vectors`
//
// They coincide for a worker sending inside its own mesh, and the overload below is that case. They
// differ at a destination-mesh landing, where the tree is T(landing) but the rectangle is still the
// one the original client asked for around the retained range_anchor. Passing the anchor where the
// encode root belongs would read the teeth off the wrong column and, when the two columns differ,
// silently drop LOCAL_DELIVER from every target row.
//
// There is no encode_root_y parameter because none is needed: the Y tree in `vectors` is already
// rooted at the encode root, and pruning walks from the targets up to it.
//
// N walks toward decreasing y and S toward increasing y, both modular: the axis is a ring, and an
// extent that wraps is a legal request rather than a clamped one.
//
// The anchor row is not itself a Y target -- an N/S range names the rows around the anchor, not the
// anchor -- but with no Y extent at all the operation is X-only and the anchor row is where the
// teeth attach, so it stands in as the single target row.
inline void encode_indexed_mcast_maps(
    std::uint8_t* route_buffer,
    const std::uint8_t* vectors,
    std::uint32_t y_size,
    std::uint32_t x_size,
    std::uint32_t anchor_y,
    std::uint32_t anchor_x,
    std::uint32_t encode_root_x,
    std::uint32_t n_hops,
    std::uint32_t s_hops,
    std::uint32_t e_hops,
    std::uint32_t w_hops) {
    const std::uint32_t root_y = anchor_y;
    const std::uint32_t root_x = anchor_x;
    std::uint8_t* out_y = route_buffer;
    std::uint8_t* out_x = route_buffer + y_size;
    for (std::uint32_t i = 0; i < y_size + x_size; ++i) {
        route_buffer[i] = 0;
    }

    std::uint32_t y_targets[MCAST_ROW_BITS_WORDS] = {0, 0};
    std::uint32_t x_targets[MCAST_ROW_BITS_WORDS] = {0, 0};

    if (n_hops == 0 && s_hops == 0) {
        mcast_set_row_bit(y_targets, root_y);
    } else {
        for (std::uint32_t k = 1; k <= n_hops; ++k) {
            mcast_set_row_bit(y_targets, (root_y + y_size - (k % y_size)) % y_size);
        }
        for (std::uint32_t k = 1; k <= s_hops; ++k) {
            mcast_set_row_bit(y_targets, (root_y + k) % y_size);
        }
    }

    // The anchor column is always a target: the spine rows deliver, not merely forward.
    mcast_set_row_bit(x_targets, root_x);
    for (std::uint32_t k = 1; k <= e_hops; ++k) {
        mcast_set_row_bit(x_targets, (root_x + k) % x_size);
    }
    for (std::uint32_t k = 1; k <= w_hops; ++k) {
        mcast_set_row_bit(x_targets, (root_x + x_size - (k % x_size)) % x_size);
    }

    const std::uint8_t* tree_y = vectors + IndexedMeshRoutingFields::mcast_tree_y_offset(y_size, x_size);
    const std::uint8_t* tree_x = vectors + IndexedMeshRoutingFields::mcast_tree_x_offset(y_size, x_size);

    std::uint32_t needed_x[MCAST_ROW_BITS_WORDS] = {x_targets[0], x_targets[1]};
    mcast_prune_axis(out_x, tree_x, x_size, needed_x, /*is_y_axis=*/false);
    for (std::uint32_t x = 0; x < x_size; ++x) {
        if (mcast_test_row_bit(x_targets, x)) {
            out_x[x] |= IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
        }
    }

    std::uint32_t needed_y[MCAST_ROW_BITS_WORDS] = {y_targets[0], y_targets[1]};
    mcast_prune_axis(out_y, tree_y, y_size, needed_y, /*is_y_axis=*/true);

    // Contract 5.5: every target row carries the ENCODE ROOT column's own E/W teeth, and delivers
    // only if that column is itself a target column. Read after the LOCAL_DELIVER pass above, which
    // is what makes that condition true when it should be. Indexed by encode_root_x and not by the
    // anchor: the teeth are what this chip has to launch, and at a landing this chip is not the
    // anchor.
    const std::uint8_t x_root_action = out_x[encode_root_x];
    const std::uint8_t teeth =
        x_root_action & (IndexedMeshRoutingFields::ACTION_EAST | IndexedMeshRoutingFields::ACTION_WEST);
    const std::uint8_t deliver = x_root_action & IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
    for (std::uint32_t y = 0; y < y_size; ++y) {
        if (mcast_test_row_bit(y_targets, y)) {
            out_y[y] |= teeth | deliver;
        }
    }
}

// Same-mesh source: the client's rectangle is measured from the very chip that roots the trees, so
// the anchor and the encode root are one chip and callers need not distinguish them.
inline void encode_indexed_mcast_maps(
    std::uint8_t* route_buffer,
    const std::uint8_t* vectors,
    std::uint32_t y_size,
    std::uint32_t x_size,
    std::uint32_t root_y,
    std::uint32_t root_x,
    std::uint32_t n_hops,
    std::uint32_t s_hops,
    std::uint32_t e_hops,
    std::uint32_t w_hops) {
    encode_indexed_mcast_maps(
        route_buffer, vectors, y_size, x_size, root_y, root_x, root_x, n_hops, s_hops, e_hops, w_hops);
}

// FWD_DIRS slot order: base {E, W, N, S, Z} with the self direction removed.
static_assert(
    IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::NORTH>()[0] == eth_chan_directions::EAST &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::NORTH>()[1] == eth_chan_directions::WEST &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::NORTH>()[2] == eth_chan_directions::SOUTH &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::NORTH>()[3] == eth_chan_directions::Z,
    "fwd_dirs<NORTH> must be {E, W, S, Z}");
static_assert(
    IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::EAST>()[0] == eth_chan_directions::WEST &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::EAST>()[3] == eth_chan_directions::Z,
    "fwd_dirs<EAST> must be {W, N, S, Z}");
static_assert(
    IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::Z>()[0] == eth_chan_directions::EAST &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::Z>()[1] == eth_chan_directions::WEST &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::Z>()[2] == eth_chan_directions::NORTH &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::Z>()[3] == eth_chan_directions::SOUTH,
    "fwd_dirs<Z> must be {E, W, N, S}");
static_assert(
    IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::SOUTH>()[2] == eth_chan_directions::NORTH &&
        IndexedMeshRoutingFields::fwd_dirs<eth_chan_directions::WEST>()[1] == eth_chan_directions::NORTH,
    "fwd_dirs<SOUTH>/<WEST> must exclude the self direction in slot order");

// Packing example: at a NORTH-facing router, action S|Z|LOCAL_DELIVER packs to 0b1100
// (LOCAL_DELIVER stays outside the key).
static_assert(
    IndexedMeshRoutingFields::pack_fwd_key<eth_chan_directions::NORTH>(
        IndexedMeshRoutingFields::ACTION_SOUTH | IndexedMeshRoutingFields::ACTION_Z |
        IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER) == 0b1100,
    "pack_fwd_key<NORTH>(S|Z|LOCAL_DELIVER) must be 0b1100");
static_assert(
    IndexedMeshRoutingFields::pack_fwd_key<eth_chan_directions::EAST>(
        IndexedMeshRoutingFields::ACTION_WEST | IndexedMeshRoutingFields::ACTION_NORTH) == 0b0011,
    "pack_fwd_key<EAST>(W|N) must select slots {W, N} -> 0b0011");

// Invalid-action checks.
static_assert(
    IndexedMeshRoutingFields::action_is_valid<eth_chan_directions::NORTH>(IndexedMeshRoutingFields::ACTION_SOUTH) &&
        IndexedMeshRoutingFields::action_is_valid<eth_chan_directions::Z>(IndexedMeshRoutingFields::ACTION_EAST),
    "ordinary non-self outputs must be valid");
static_assert(
    !IndexedMeshRoutingFields::action_is_valid<eth_chan_directions::NORTH>(IndexedMeshRoutingFields::ACTION_NORTH) &&
        !IndexedMeshRoutingFields::action_is_valid<eth_chan_directions::Z>(IndexedMeshRoutingFields::ACTION_Z) &&
        !IndexedMeshRoutingFields::action_is_valid<eth_chan_directions::EAST>(IndexedMeshRoutingFields::ACTION_EAST),
    "the self-facing bit must be invalid");
static_assert(
    !IndexedMeshRoutingFields::action_is_valid<eth_chan_directions::WEST>(0) &&
        !IndexedMeshRoutingFields::action_is_valid<eth_chan_directions::SOUTH>(0x80),
    "empty actions and reserved bits must be invalid");

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

/**
 * Canonical 1D sparse multicast routing pattern encoder
 *
 * Generates bit pattern for multicast routing based on a supplied hop mask
 * Each bit in the hop mask represents a single hop in the target direction
 * If the bit is 1, the router will perform the WRITE operation at that hop.
 * If the bit is 0, the router will simply forward the packet to the next hop.
 * This continues until the last set bit, which will perform a WRITE operation and not forward the packet any further.
 * For instance, a hop mask of 0b1010 means that devices 2 and 4 hops away from sender will have the data written to
 * them. This function converts the hop mask into a fabric multicast packet header routing field, following the 2-bit
 * encoding shown in encode_1d_multicast.
 *
 * @param hop_mask Bitmask of hops to write. Currently only supports uint16_t, tracked in #36581
 * @param buffer Output buffer (uint32_t, will be expanded into array in future, tracked in #36581)
 *
 * Example: hop mask 0b1010 would be converted into the following routing fields:
 *   - Hop 0 (0): FORWARD_ONLY (0b10)
 *   - Hop 1 (1): WRITE_AND_FORWARD (0b11)
 *   - Hop 2 (0): FORWARD_ONLY (0b10)
 *   - Hop 3 (1): WRITE_ONLY (0b01)
 *   Resulting routing field: 0b01'10'11'10
 *
 * Router consumes fields LSB-first (hop 0 at bits 0-1, hop 1 at bits 2-3, etc.)
 */
template <typename HopMaskType>
inline void encode_1d_sparse_multicast(HopMaskType hop_mask, uint32_t& buffer) {
    using LowLatencyFields = RoutingFieldsConstants::LowLatency;

// Currently, hop_mask must be an unsigned integer and currently only supports uint16_t, tracked in #36581
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
    ASSERT(std::is_unsigned_v<HopMaskType> && (sizeof(HopMaskType) == sizeof(uint16_t)));
#endif

    auto set_hop_field = [&](uint32_t hop_index, uint32_t field_value) {
        const uint32_t bit_pos = (hop_index % LowLatencyFields::BASE_HOPS) * LowLatencyFields::FIELD_WIDTH;
        buffer |= (field_value << bit_pos);
    };

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
    ASSERT(hop_mask > 0);
#endif

    buffer = 0;
    uint32_t hop_index = 0;
    // Treat hop_mask like a shift register, checking LSB each time
    while (hop_mask > 0) {
        // Case 1: We've arrived at the last hop. Write and stop.
        if (hop_mask == 1) {
            set_hop_field(hop_index, LowLatencyFields::WRITE_ONLY);
        }
        // Case 2: This hop involves a write operation. Write and forward.
        else if (hop_mask & 1) {
            set_hop_field(hop_index, LowLatencyFields::WRITE_AND_FORWARD);
        }
        // Case 3: This hop does not involve a write operation. Forward only.
        else {
            set_hop_field(hop_index, LowLatencyFields::FORWARD_ONLY);
        }
        hop_index++;
        hop_mask >>= 1;
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

    // Forward commands per dimension
    const uint8_t ns_fwd = (ns_dir == 1) ? MeshFields::FORWARD_SOUTH : MeshFields::FORWARD_NORTH;
    const uint8_t ew_fwd = (ew_dir == 1) ? MeshFields::FORWARD_EAST : MeshFields::FORWARD_WEST;

    // Final hop uses OPPOSITE direction to stop the packet at the destination
    const uint8_t ns_write = (ns_dir == 1) ? MeshFields::FORWARD_NORTH : MeshFields::FORWARD_SOUTH;
    const uint8_t ew_write = (ew_dir == 1) ? MeshFields::FORWARD_WEST : MeshFields::FORWARD_EAST;

    // Build the ordered hop-direction list in dimension order (all NS, then all EW).
    // Each entry is (forward_cmd, write_cmd) for that hop's dimension.
    uint8_t fwd_list[FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE];
    uint8_t wr_list[FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE];
    uint32_t num_hops = 0;
    const uint32_t total_cardinal = (uint32_t)ns_hops + (uint32_t)ew_hops;
    for (uint32_t c = 0; c < total_cardinal; ++c) {
        if (c < ns_hops) {
            fwd_list[num_hops] = ns_fwd;
            wr_list[num_hops] = ns_write;
        } else {
            fwd_list[num_hops] = ew_fwd;
            wr_list[num_hops] = ew_write;
        }
        ++num_hops;
    }

    if (num_hops == 0) {
        while (idx < max_buffer_size) {
            buffer[idx++] = MeshFields::NOOP;
        }
        return;
    }

    // buffer[i] is the action performed at the (i+1)-th chip after the source. The source leaves via
    // hop 0 (the "initial direction", not encoded here) unless prepend_one_hop is set (router usage),
    // in which case hop 0's forward is emitted too. Every intermediate chip forwards the *next* hop,
    // and the destination chip performs the write for the final hop.
    const uint32_t first_hop = prepend_one_hop ? 0 : 1;
    for (uint32_t k = first_hop; k < num_hops; ++k) {
        buffer[idx++] = fwd_list[k];
    }
    buffer[idx++] = wr_list[num_hops - 1];

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

enum class RouterCommand : std::uint32_t {
    // The main state where messages and credits are forwarded
    RUN = 0,

    // The router enters the pause state, which is the "hub" transitionary state to other states.
    // When paused, no messages/credits are processed by the router
    PAUSE = 1,

    // The router accepts messages but drops them instead of forwarding them.
    // The pipe to /dev/null of TT-Fabric
    DRAIN = 3,

    // Commands the router to make one link retrain attempt
    RETRAIN = 4
};

struct RouterStateManager {
    RouterState state;  // 4B, written by device, read by host
    uint8_t padding0[12];     //
    RouterCommand command;    // 4B, written by host, read by device
    uint8_t padding1[12];     //

    // template <bool ENABLE_RISC_CPU_DATA_CACHE>
    bool is_non_run_command_pending() const {
        // router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        return command != RouterCommand::RUN;
    }
};

// Destination-major indexed route vectors. Shares the 2D union slot with the legacy compressed
// table; the two interpretations are never live at once.
// Raw byte storage because row strides depend on the live mesh shape [Y,X]:
//   y_vectors: row dst_y at byte offset dst_y * ceil(Y/4), region [0, Y*ceil(Y/4))
//   x_vectors: row dst_x at byte offset dst_x * ceil(X/4), region [Y*ceil(Y/4), Y*ceil(Y/4) + X*ceil(X/4))
// Typed accessors and the host-side packer live on IndexedMeshRoutingFields
// (y_row/x_row, pack_indexed_route_vectors); the device-side widen helpers read rows through them.
struct __attribute__((packed)) indexed_route_vectors_t {
    std::uint8_t data[IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES];  // 1028

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    // Fills the destination-major 2-bit vector table for this mesh by probing ControlPlane's
    // first-hop relation along each axis and packing it with
    // IndexedMeshRoutingFields::pack_indexed_route_vectors (implemented in compressed_routing_path.cpp).
    // Deliberately same-named and same-signatured as
    // intra_mesh_routing_path_t<2, true>::calculate_chip_to_all_routing_fields so the ControlPlane
    // call site swap is a mechanical type + memcpy-target change. Asserts on representational
    // illegality (off-axis probe result or shape overflow).
    void calculate_chip_to_all_routing_fields(const FabricNodeId& src_fabric_node_id, uint16_t num_chips);
#endif
};
static_assert(
    sizeof(indexed_route_vectors_t) == IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES,
    "indexed route vectors must be exactly the [64,4] bound");

struct routing_l1_info_t {
    RouterStateManager state_manager{};  // 32 bytes
    uint16_t my_mesh_id = 0;           // Current mesh ID // 2 bytes
    uint16_t my_device_id = 0;         // Current chip ID // 2 bytes
    // NOTE: Compressed version has additional overhead (2x slower) to read values,
    //       but raw data is too huge (2048 bytes) to fit in L1 memory.
    //       Need to evaluate once actual workloads are available
    direction_table_t<MAX_MESH_SIZE> intra_mesh_direction_table{};   // 96 bytes
    direction_table_t<MAX_NUM_MESHES> inter_mesh_direction_table{};  // 384 bytes

    // Union overlaps 1D and 2D routing tables at same offset. indexed_route_vectors is the target
    // 2D layout; it and routing_path_table_2d are never live at once.
    union __attribute__((packed)) {
        intra_mesh_routing_path_t<1, false> routing_path_table_1d;  // 1024 bytes
        intra_mesh_routing_path_t<2, true> routing_path_table_2d;   // 1024 bytes
        indexed_route_vectors_t indexed_route_vectors;              // 1028 bytes
    };

    std::uint8_t exit_node_table[MAX_NUM_MESHES] = {};               // 1024 bytes
    // This chip's (y, x) coordinates within its mesh, row-major with my_device_id
    // (y = id / x_size, x = id % x_size). Populated host-side with the rest of the table.
    std::uint8_t my_mesh_coord_y = 0;
    std::uint8_t my_mesh_coord_x = 0;
    uint8_t padding[6] = {};  // pad to 16-byte alignment
};
static_assert(offsetof(routing_l1_info_t, routing_path_table_1d) == 516);
static_assert(
    offsetof(routing_l1_info_t, exit_node_table) == 516 + IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES,
    "exit_node_table must follow the 1028-byte 2D union slot");
static_assert(
    offsetof(routing_l1_info_t, my_mesh_coord_y) ==
        516 + IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES + MAX_NUM_MESHES,
    "my_mesh_coord_y must immediately follow exit_node_table");
static_assert(offsetof(routing_l1_info_t, state_manager) % 16 == 0);
static_assert(sizeof(routing_l1_info_t) % 16 == 0);

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
    sizeof(routing_l1_info_t) == 2576,
    "routing_l1_info_t must be 2576 bytes: base(516) + union(1028) + exit(1024) + coords(2) + pad(6)");

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
#elif defined(COMPILE_FOR_DISPATCH_ENGINE)
#define ROUTING_PATH_BASE MEM_DISPATCH_TENSIX_ROUTING_PATH_BASE
#define ROUTING_PATH_BASE_1D MEM_DISPATCH_TENSIX_ROUTING_PATH_BASE
#define ROUTING_PATH_BASE_2D MEM_DISPATCH_TENSIX_ROUTING_PATH_BASE
#define ROUTING_TABLE_BASE MEM_DISPATCH_TENSIX_ROUTING_TABLE_BASE
#define EXIT_NODE_TABLE_BASE MEM_DISPATCH_TENSIX_EXIT_NODE_TABLE_BASE
#else
#define ROUTING_PATH_BASE MEM_TENSIX_ROUTING_PATH_BASE
#define ROUTING_PATH_BASE_1D MEM_TENSIX_ROUTING_PATH_BASE_1D
#define ROUTING_PATH_BASE_2D MEM_TENSIX_ROUTING_PATH_BASE_2D
#define ROUTING_TABLE_BASE MEM_TENSIX_ROUTING_TABLE_BASE
#define EXIT_NODE_TABLE_BASE MEM_TENSIX_EXIT_NODE_TABLE_BASE
#endif

#if defined(COMPILE_FOR_DISPATCH_ENGINE)
#define FABRIC_CONNECTIONS_BASE MEM_DISPATCH_TENSIX_FABRIC_CONNECTIONS_BASE
#define FABRIC_CONNECTION_LOCK_BASE MEM_DISPATCH_FABRIC_CONNECTION_LOCK_BASE
#define FABRIC_COUNTER_BASE MEM_DISPATCH_FABRIC_COUNTER_BASE
#else
#define FABRIC_CONNECTIONS_BASE MEM_TENSIX_FABRIC_CONNECTIONS_BASE
#define FABRIC_CONNECTION_LOCK_BASE MEM_FABRIC_CONNECTION_LOCK_BASE
#define FABRIC_COUNTER_BASE MEM_FABRIC_COUNTER_BASE
#endif

#include "fabric/hw/inc/fabric_direction_table_interface.h"
#include "fabric/hw/inc/fabric_routing_path_interface.h"
#endif
