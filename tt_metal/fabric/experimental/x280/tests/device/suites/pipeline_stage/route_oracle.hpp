// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host routing oracle and prepared input builder for route_compute.

#ifndef X280_SUITES_PIPELINE_STAGE_ROUTE_ORACLE_HPP
#define X280_SUITES_PIPELINE_STAGE_ROUTE_ORACLE_HPP

#include "fabric_shadow_header.hpp"
#include "pipeline_stage_common.h"
#include "route_compute_subject.hpp"
#include "rt/x280_pinned_input.h"

#include <hostdevcommon/fabric_common.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace x280::suites::pipeline_stage::route_compute {

using ::pipeline_stage::expected_direction;
using ::pipeline_stage::kFixture;
using ::pipeline_stage::kFixtureMesh;

constexpr uint32_t kRoutingImageBytes = static_cast<uint32_t>(X280_PINNED_INPUT_ROUTING_BYTES);

static_assert(sizeof(tt::tt_fabric::routing_l1_info_t) == kRoutingImageBytes);
static_assert(offsetof(tt::tt_fabric::routing_l1_info_t, intra_mesh_direction_table) == 36u);
static_assert(offsetof(tt::tt_fabric::routing_l1_info_t, inter_mesh_direction_table) == 132u);

inline std::vector<uint8_t> build_input_image() {
    tt::tt_fabric::routing_l1_info_t routing{};
    routing.my_mesh_id = kFixtureMesh;
    routing.my_device_id = 0u;

    const uint8_t invalid = static_cast<uint8_t>(tt::tt_fabric::eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
    for (uint16_t chip = 0u; chip < tt::tt_fabric::MAX_MESH_SIZE; ++chip) {
        routing.intra_mesh_direction_table.set_original_direction(chip, invalid);
    }
    for (uint16_t mesh = 0u; mesh < tt::tt_fabric::MAX_NUM_MESHES; ++mesh) {
        routing.inter_mesh_direction_table.set_original_direction(mesh, invalid);
    }

    for (uint32_t slot = 0u; slot < 8u; ++slot) {
        routing.intra_mesh_direction_table.set_original_direction(kFixture[slot].chip, expected_direction(slot));
    }
    for (uint32_t slot = 8u; slot < PIPELINE_STAGE_LOOKUP_HEADER_COUNT; ++slot) {
        routing.inter_mesh_direction_table.set_original_direction(kFixture[slot].mesh, expected_direction(slot));
    }

    std::vector<uint8_t> input(X280_PINNED_INPUT_SPAN, 0u);
    std::memcpy(input.data(), &routing, sizeof(routing));

    for (uint32_t slot = 0u; slot < PIPELINE_STAGE_LOOKUP_HEADER_COUNT; ++slot) {
        fs_hybrid_mesh_packet_header_t header{};
        for (uint32_t i = 0u; i < sizeof(header.base); ++i) {
            header.base[i] = static_cast<uint8_t>(0x31u + slot * 7u + i);
        }
        header.routing_fields.value = 0x10000u + slot;
        for (uint32_t i = 0u; i < FS_ROUTE_BUFFER_CAPACITY; ++i) {
            header.route_buffer[i] = static_cast<uint8_t>(1u + ((slot + i) & 0xFu));
        }
        header.dst_start_chip_id = kFixture[slot].chip;
        header.dst_start_mesh_id = kFixture[slot].mesh;
        header.mcast_params_64 = 0xA500000000000000ull | static_cast<uint64_t>(slot);
        header.is_mcast_active = 0u;
        std::memcpy(
            input.data() + X280_PINNED_INPUT_HEADERS_OFFSET + slot * FS_PACKET_HEADER_BYTES, &header, sizeof(header));
    }
    return input;
}

inline uint32_t expected_witness(uint32_t seed, uint32_t count) {
    uint32_t witness = 0u;
    uint32_t generator = seed;
    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t slot = ::pipeline_stage::next_slot(generator);
        witness += static_cast<uint32_t>(expected_direction(slot)) + 1u;
    }
    return witness;
}

}  // namespace x280::suites::pipeline_stage::route_compute

#endif  // X280_SUITES_PIPELINE_STAGE_ROUTE_ORACLE_HPP
