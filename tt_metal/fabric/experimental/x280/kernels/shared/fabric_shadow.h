// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Benchmark-local view of the deployed fabric layout. */
#ifndef X280_FABRIC_SHADOW_H
#define X280_FABRIC_SHADOW_H

#include <stddef.h>
#include <stdint.h>

#include "ubench_section_load.h"

/* The deployed 2D packet header: HybridMeshPacketHeaderT<35>, 96 bytes
 * (tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp). */
#define FS_PACKET_HEADER_BYTES 96u
#define FS_ROUTE_BUFFER_CAPACITY 35u

#define FS_MAX_MESH_SIZE 256u
#define FS_MAX_NUM_MESHES 1024u

#define FS_DIRECTION_EAST 0u
#define FS_DIRECTION_WEST 1u
#define FS_DIRECTION_NORTH 2u
#define FS_DIRECTION_SOUTH 3u
#define FS_DIRECTION_Z 4u
#define FS_INVALID_DIRECTION 0xDDu
#define FS_INVALID_ROUTING_TABLE_ENTRY 0xFFu

#define FS_COMPRESSED_EAST 0u
#define FS_COMPRESSED_WEST 1u
#define FS_COMPRESSED_NORTH 2u
#define FS_COMPRESSED_SOUTH 3u
#define FS_COMPRESSED_Z 4u
#define FS_COMPRESSED_INVALID_DIRECTION 5u
#define FS_COMPRESSED_INVALID_ROUTING_TABLE_ENTRY 6u

#define FS_BITS_PER_COMPRESSED_ENTRY 3u
#define FS_BITS_PER_BYTE 8u
#define FS_COMPRESSED_ENTRY_MASK 0x7u

typedef struct {
    uint8_t packed_directions[FS_MAX_MESH_SIZE * FS_BITS_PER_COMPRESSED_ENTRY / FS_BITS_PER_BYTE];
} fs_intra_mesh_direction_table_t;

typedef struct {
    uint8_t packed_directions[FS_MAX_NUM_MESHES * FS_BITS_PER_COMPRESSED_ENTRY / FS_BITS_PER_BYTE];
} fs_inter_mesh_direction_table_t;

/*
 * Decode one packed entry. Returns zero without reading the table when the
 * pointers are null or index is outside entry_count. The backing byte extent
 * remains the caller's contract.
 */
static inline int fs_packed_direction(
    const volatile uint8_t* packed_directions, uint16_t entry_count, uint16_t index, uint8_t* compressed_direction) {
    if (compressed_direction == NULL) {
        return 0;
    }
    *compressed_direction = FS_INVALID_ROUTING_TABLE_ENTRY;
    if (packed_directions == NULL || index >= entry_count) {
        return 0;
    }

    const uint32_t bit_index = (uint32_t)index * FS_BITS_PER_COMPRESSED_ENTRY;
    const uint32_t byte_index = bit_index / FS_BITS_PER_BYTE;
    const uint32_t bit_offset = bit_index % FS_BITS_PER_BYTE;

    if (bit_offset <= FS_BITS_PER_BYTE - FS_BITS_PER_COMPRESSED_ENTRY) {
        *compressed_direction = (uint8_t)((packed_directions[byte_index] >> bit_offset) & FS_COMPRESSED_ENTRY_MASK);
        return 1;
    }

    const uint32_t bits_in_first = FS_BITS_PER_BYTE - bit_offset;
    const uint8_t low = (uint8_t)((packed_directions[byte_index] >> bit_offset) & ((1u << bits_in_first) - 1u));
    const uint8_t high =
        (uint8_t)((packed_directions[byte_index + 1u] & ((1u << (FS_BITS_PER_COMPRESSED_ENTRY - bits_in_first)) - 1u))
                  << bits_in_first);
    *compressed_direction = (uint8_t)(low | high);
    return 1;
}

static inline int fs_original_direction(
    const volatile uint8_t* packed_directions, uint16_t entry_count, uint16_t index, uint8_t* original_direction) {
    if (original_direction == NULL) {
        return 0;
    }
    *original_direction = FS_INVALID_ROUTING_TABLE_ENTRY;
    uint8_t compressed_direction;
    if (!fs_packed_direction(packed_directions, entry_count, index, &compressed_direction)) {
        return 0;
    }
    switch (compressed_direction) {
        case FS_COMPRESSED_EAST: *original_direction = FS_DIRECTION_EAST; break;
        case FS_COMPRESSED_WEST: *original_direction = FS_DIRECTION_WEST; break;
        case FS_COMPRESSED_NORTH: *original_direction = FS_DIRECTION_NORTH; break;
        case FS_COMPRESSED_SOUTH: *original_direction = FS_DIRECTION_SOUTH; break;
        case FS_COMPRESSED_Z: *original_direction = FS_DIRECTION_Z; break;
        case FS_COMPRESSED_INVALID_DIRECTION: *original_direction = FS_INVALID_DIRECTION; break;
        case FS_COMPRESSED_INVALID_ROUTING_TABLE_ENTRY:
        default: *original_direction = FS_INVALID_ROUTING_TABLE_ENTRY; break;
    }
    return 1;
}

#define FS_HOP_NOOP 0x0u
#define FS_HOP_FORWARD_EAST 0x1u
#define FS_HOP_FORWARD_WEST 0x2u
#define FS_HOP_FORWARD_NORTH 0x4u
#define FS_HOP_FORWARD_SOUTH 0x8u

typedef struct __attribute__((aligned(16))) {
    uint8_t opaque_state[32];
    uint16_t my_mesh_id;
    uint16_t my_device_id;
    fs_intra_mesh_direction_table_t intra_mesh_direction_table;
    fs_inter_mesh_direction_table_t inter_mesh_direction_table;
    uint8_t opaque_tail[2060];
} fs_routing_l1_info_t;

UBENCH_LAYOUT_ASSERT(sizeof(fs_intra_mesh_direction_table_t) == 96u, "intra table is 256 packed 3-bit entries");
UBENCH_LAYOUT_ASSERT(sizeof(fs_inter_mesh_direction_table_t) == 384u, "inter table is 1024 packed 3-bit entries");
UBENCH_LAYOUT_ASSERT(
    offsetof(fs_routing_l1_info_t, my_mesh_id) == 32u, "local mesh identity retains the production offset");
UBENCH_LAYOUT_ASSERT(
    offsetof(fs_routing_l1_info_t, intra_mesh_direction_table) == 36u, "intra table retains the production offset");
UBENCH_LAYOUT_ASSERT(
    offsetof(fs_routing_l1_info_t, inter_mesh_direction_table) == 132u, "inter table retains the production offset");
UBENCH_LAYOUT_ASSERT(sizeof(fs_routing_l1_info_t) == 2576u, "full routing image retains the production size");

#endif
