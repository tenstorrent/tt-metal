// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef PIPELINE_STAGE_ROUTE_COMPUTE_SUBJECT_HPP
#define PIPELINE_STAGE_ROUTE_COMPUTE_SUBJECT_HPP

#include <stddef.h>
#include <stdint.h>

#include "../shared/fabric_shadow.h"
#include "../shared/fabric_shadow_header.hpp"
#include "pipeline_stage_common.h"

namespace pipeline_stage {

struct fixture_header_t {
    uint16_t chip;
    uint16_t mesh;
};

/* PIPELINE_STAGE_LOOKUP_FIXTURE_V1: the routing image's own mesh, and each
 * header slot's destination. Slots 0-7 route within the mesh, 8-15 between
 * meshes. */
static constexpr uint16_t kFixtureMesh = 513u;
static constexpr fixture_header_t kFixture[PIPELINE_STAGE_LOOKUP_HEADER_COUNT] = {
    {0u, kFixtureMesh},
    {1u, kFixtureMesh},
    {2u, kFixtureMesh},
    {5u, kFixtureMesh},
    {7u, kFixtureMesh},
    {8u, kFixtureMesh},
    {254u, kFixtureMesh},
    {255u, kFixtureMesh},
    {255u, 0u},
    {254u, 1u},
    {8u, 2u},
    {7u, 5u},
    {5u, 8u},
    {2u, 1021u},
    {1u, 1022u},
    {0u, 1023u},
};

/* The direction the fixture's routing tables hold for slot (< 16). */
static constexpr uint8_t expected_direction(uint32_t slot) {
    return slot < 8u ? static_cast<uint8_t>(kFixture[slot].chip & 3u)
                     : static_cast<uint8_t>((kFixture[slot].mesh + 2u) & 3u);
}

/* Advances the xorshift32 generator and returns the header slot it selects.
 * A zero generator stays zero; run_nonce, the seed, is nonzero. */
static inline __attribute__((always_inline)) uint32_t next_slot(uint32_t& generator) {
    uint32_t value = generator;
    value ^= value << 13u;
    value ^= value >> 17u;
    value ^= value << 5u;
    generator = value;
    return value & (PIPELINE_STAGE_LOOKUP_HEADER_COUNT - 1u);
}

/* The prepared input at `input`: the routing image at its start, then
 * PIPELINE_STAGE_LOOKUP_HEADER_COUNT header slots at
 * X280_PINNED_INPUT_HEADERS_OFFSET. */
struct prepared_input_t {
    volatile uint8_t* input;

    inline __attribute__((always_inline)) const volatile fs_routing_l1_info_t* routing() const {
        return reinterpret_cast<const volatile fs_routing_l1_info_t*>(input);
    }

    /* slot is below PIPELINE_STAGE_LOOKUP_HEADER_COUNT. */
    inline __attribute__((always_inline)) const volatile fs_hybrid_mesh_packet_header_t* header(uint32_t slot) const {
        return reinterpret_cast<const volatile fs_hybrid_mesh_packet_header_t*>(
            input + X280_PINNED_INPUT_HEADERS_OFFSET + (uint64_t)slot * FS_PACKET_HEADER_BYTES);
    }
};

static inline __attribute__((always_inline)) uint8_t lookup_direction(const prepared_input_t& a, uint32_t slot) {
    const volatile fs_routing_l1_info_t* routing = a.routing();
    const volatile fs_hybrid_mesh_packet_header_t* header = a.header(slot);

    /* Offsets 83 and 85 forbid aligned casts. */
    const uint16_t destination_chip = header->dst_start_chip_id;
    const uint16_t destination_mesh = header->dst_start_mesh_id;
    uint8_t direction = FS_INVALID_ROUTING_TABLE_ENTRY;
    if (destination_mesh == routing->my_mesh_id) {
        (void)fs_original_direction(
            routing->intra_mesh_direction_table.packed_directions, FS_MAX_MESH_SIZE, destination_chip, &direction);
    } else {
        (void)fs_original_direction(
            routing->inter_mesh_direction_table.packed_directions, FS_MAX_NUM_MESHES, destination_mesh, &direction);
    }
    return direction;
}

static inline __attribute__((always_inline)) void stage_once(
    const prepared_input_t& a, uint32_t slot, uint32_t& witness) {
    witness += static_cast<uint32_t>(lookup_direction(a, slot)) + 1u;
}

/* True when the routing image and every header hold the fixture and every
 * slot looks up its expected direction; mismatch is then
 * PIPELINE_STAGE_VERIFY_MATCH. Otherwise mismatch names the first failure
 * (pipeline_stage_verify_mismatch of the slot, or of the mesh-id offset). */
static inline bool verify_fixture(const prepared_input_t& a, uint32_t& mismatch) {
    const volatile fs_routing_l1_info_t* routing = a.routing();
    if (routing->my_mesh_id != kFixtureMesh) {
        mismatch = pipeline_stage_verify_mismatch(
            offsetof(fs_routing_l1_info_t, my_mesh_id), static_cast<uint8_t>(routing->my_mesh_id));
        return false;
    }

    for (uint32_t slot = 0u; slot < PIPELINE_STAGE_LOOKUP_HEADER_COUNT; ++slot) {
        const volatile fs_hybrid_mesh_packet_header_t* header = a.header(slot);
        if (header->dst_start_chip_id != kFixture[slot].chip || header->dst_start_mesh_id != kFixture[slot].mesh) {
            mismatch = pipeline_stage_verify_mismatch(slot, static_cast<uint8_t>(header->dst_start_chip_id));
            return false;
        }
        const uint8_t produced = lookup_direction(a, slot);
        if (produced != expected_direction(slot)) {
            mismatch = pipeline_stage_verify_mismatch(slot, produced);
            return false;
        }
    }
    mismatch = PIPELINE_STAGE_VERIFY_MATCH;
    return true;
}

}  // namespace pipeline_stage

#endif  // PIPELINE_STAGE_ROUTE_COMPUTE_SUBJECT_HPP
