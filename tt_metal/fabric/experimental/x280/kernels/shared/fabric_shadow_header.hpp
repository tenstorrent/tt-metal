// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Concrete deployed fabric packet layout shared by lookup and forwarding.
 */
#ifndef X280_FABRIC_SHADOW_HEADER_HPP
#define X280_FABRIC_SHADOW_HEADER_HPP

#include <stddef.h>
#include <stdint.h>

#include "fabric_shadow.h"

struct fs_mesh_routing_fields_t {
    union {
        uint32_t value;
        struct {
            uint16_t hop_index;
            uint8_t branch_east_offset;
            uint8_t branch_west_offset;
        };
    };
};

struct __attribute__((packed, aligned(16))) fs_hybrid_mesh_packet_header_t {
    uint8_t base[44];
    fs_mesh_routing_fields_t routing_fields;
    uint8_t route_buffer[FS_ROUTE_BUFFER_CAPACITY];
    union {
        struct {
            uint16_t dst_start_chip_id;
            uint16_t dst_start_mesh_id;
        };
        uint32_t dst_start_node_id;
    };
    union {
        uint16_t mcast_params[4];
        uint64_t mcast_params_64;
    };
    uint8_t is_mcast_active;
};

static_assert(sizeof(fs_mesh_routing_fields_t) == 4u, "routing fields are one word");
static_assert(
    sizeof(fs_hybrid_mesh_packet_header_t) == FS_PACKET_HEADER_BYTES, "fixed route capacity produces a 96-byte header");
static_assert(alignof(fs_hybrid_mesh_packet_header_t) == 16u, "packet allocation bases are 16-byte aligned");
static_assert(
    offsetof(fs_hybrid_mesh_packet_header_t, routing_fields) == 44u, "routing fields retain their production offset");
static_assert(
    offsetof(fs_hybrid_mesh_packet_header_t, route_buffer) == 48u, "route buffer retains its production offset");
static_assert(
    offsetof(fs_hybrid_mesh_packet_header_t, dst_start_chip_id) == 83u,
    "destination chip remains unaligned at offset 83");
static_assert(
    offsetof(fs_hybrid_mesh_packet_header_t, dst_start_mesh_id) == 85u,
    "destination mesh remains unaligned at offset 85");

/* Initializes the complete fixed header with the four forwarding commands.
 * The caller must provide writable storage and publish the writes afterward. */
inline void fs_seed_forward_header(volatile fs_hybrid_mesh_packet_header_t* header) {
    volatile uint8_t* bytes = reinterpret_cast<volatile uint8_t*>(header);
    for (uint32_t i = 0u; i < FS_PACKET_HEADER_BYTES; ++i) {
        bytes[i] = 0u;
    }
    static constexpr uint8_t cycle[] = {
        FS_HOP_FORWARD_SOUTH,
        FS_HOP_FORWARD_EAST,
        FS_HOP_FORWARD_NORTH,
        FS_HOP_FORWARD_WEST,
    };
    for (uint32_t i = 0u; i < FS_ROUTE_BUFFER_CAPACITY; ++i) {
        header->route_buffer[i] = i < sizeof(cycle) ? cycle[i] : FS_HOP_NOOP;
    }
    header->routing_fields.value = 0u;
}

/* Initializes the complete 96-byte header as a zero-payload unicast packet
 * with dst_chip and dst_mesh. Emits volatile writes to prevent elision on
 * repeated staging buffer use. */
inline void fs_seed_unicast_header(
    volatile fs_hybrid_mesh_packet_header_t* header, uint16_t dst_chip, uint16_t dst_mesh) {
    volatile uint8_t* bytes = reinterpret_cast<volatile uint8_t*>(header);
    for (uint32_t i = 0u; i < FS_PACKET_HEADER_BYTES; ++i) {
        bytes[i] = 0u;
    }
    header->routing_fields.value = 0u;
    for (uint32_t i = 0u; i < FS_ROUTE_BUFFER_CAPACITY; ++i) {
        header->route_buffer[i] = FS_HOP_NOOP;
    }
    bytes[83] = static_cast<uint8_t>(dst_chip & 0xFFu);
    bytes[84] = static_cast<uint8_t>((dst_chip >> 8) & 0xFFu);
    bytes[85] = static_cast<uint8_t>(dst_mesh & 0xFFu);
    bytes[86] = static_cast<uint8_t>((dst_mesh >> 8) & 0xFFu);
}

/* Byte offsets of PacketHeaderBase fields inside `base`: payload_size_bytes
 * (uint16), noc_send_type, then src_ch_id, after the 40-byte NocCommandFields
 * (tt_metal/fabric/fabric_edm_packet_header.hpp). */
#define FS_PAYLOAD_SIZE_BYTES_OFFSET 40u
#define FS_SRC_CH_ID_OFFSET 43u

static_assert(
    FS_SRC_CH_ID_OFFSET < offsetof(fs_hybrid_mesh_packet_header_t, routing_fields),
    "src_ch_id lies inside the opaque base");

/** Writes payload_size_bytes, little-endian, as production lays it out. */
inline void fs_set_payload_size_bytes(volatile fs_hybrid_mesh_packet_header_t* header, uint16_t bytes_after_header) {
    volatile uint8_t* bytes = reinterpret_cast<volatile uint8_t*>(header);
    bytes[FS_PAYLOAD_SIZE_BYTES_OFFSET] = static_cast<uint8_t>(bytes_after_header & 0xFFu);
    bytes[FS_PAYLOAD_SIZE_BYTES_OFFSET + 1u] = static_cast<uint8_t>((bytes_after_header >> 8) & 0xFFu);
}

inline uint16_t fs_payload_size_bytes(const volatile fs_hybrid_mesh_packet_header_t* header) {
    const volatile uint8_t* bytes = reinterpret_cast<const volatile uint8_t*>(header);
    return static_cast<uint16_t>(bytes[FS_PAYLOAD_SIZE_BYTES_OFFSET] | (bytes[FS_PAYLOAD_SIZE_BYTES_OFFSET + 1u] << 8));
}

/** Writes src_ch_id, which production's EDM sender sets to its channel index. */
inline void fs_set_src_ch_id(volatile fs_hybrid_mesh_packet_header_t* header, uint8_t id) {
    reinterpret_cast<volatile uint8_t*>(header)[FS_SRC_CH_ID_OFFSET] = id;
}

inline uint8_t fs_src_ch_id(const volatile fs_hybrid_mesh_packet_header_t* header) {
    return reinterpret_cast<const volatile uint8_t*>(header)[FS_SRC_CH_ID_OFFSET];
}

#endif
