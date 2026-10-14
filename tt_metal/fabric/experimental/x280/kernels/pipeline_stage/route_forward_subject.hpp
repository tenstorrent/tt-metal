// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_PIPELINE_STAGE_ROUTE_FORWARD_SUBJECT_HPP
#define X280_PIPELINE_STAGE_ROUTE_FORWARD_SUBJECT_HPP

#include <stddef.h>
#include <stdint.h>

#include "../shared/fabric_shadow.h"
#include "../shared/fabric_shadow_header.hpp"
#include "pipeline_stage_common.h"
#include "route_forward_args.h"

namespace pipeline_stage {

static inline __attribute__((always_inline)) uint32_t decode_hop(uint8_t command) {
    switch (command & 0xFu) {
        case FS_HOP_FORWARD_EAST: return PIPELINE_STAGE_DIR_EAST;
        case FS_HOP_FORWARD_WEST: return PIPELINE_STAGE_DIR_WEST;
        case FS_HOP_FORWARD_NORTH: return PIPELINE_STAGE_DIR_NORTH;
        case FS_HOP_FORWARD_SOUTH: return PIPELINE_STAGE_DIR_SOUTH;
        default: return PIPELINE_STAGE_DIR_COUNT;
    }
}

/* The hop after hop on a route of hop_count hops, wrapping to the first. */
static inline __attribute__((always_inline)) uint32_t next_hop(uint32_t& generator, uint32_t hop_count) {
    const uint32_t hop = generator;
    generator = hop + 1u < hop_count ? hop + 1u : 0u;
    return hop;
}

/* True when each of the first hop_count route_buffer commands decodes to a
 * direction; sentinel is then PIPELINE_STAGE_VERIFY_MATCH, otherwise
 * pipeline_stage_verify_mismatch of the first failing hop and its command.
 * hop_count is at most FS_ROUTE_BUFFER_CAPACITY. */
static inline bool verify_route(
    const volatile fs_hybrid_mesh_packet_header_t* header, uint32_t hop_count, uint32_t& sentinel) {
    for (uint32_t hop = 0u; hop < hop_count; ++hop) {
        const uint8_t command = header->route_buffer[hop];
        if (decode_hop(command) >= PIPELINE_STAGE_DIR_COUNT) {
            sentinel = pipeline_stage_verify_mismatch(hop, command);
            return false;
        }
    }
    sentinel = PIPELINE_STAGE_VERIFY_MATCH;
    return true;
}

/* hop indexes the caller-validated route buffer. */
static inline __attribute__((always_inline)) void stage_once(
    volatile fs_hybrid_mesh_packet_header_t* header, uint32_t hop, uint32_t& witness) {
    const uint8_t command = header->route_buffer[hop];
    const uint32_t direction = decode_hop(command);
    header->routing_fields.value = hop + 1u;
    if (direction >= PIPELINE_STAGE_DIR_COUNT) {
        return;
    }
    witness += direction;
}

}  // namespace pipeline_stage

#endif
