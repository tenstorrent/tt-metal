// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric {

// Device-side compressed decoder function for 2D routing
template <>
inline bool intra_mesh_routing_path_t<2, true>::decode_route_to_buffer(
    uint16_t dst_chip_id, volatile uint8_t* out_route_buffer, bool prepend_one_hop) const {
    if (dst_chip_id >= MAX_CHIPS_LOWLAT) {
        // invalid chip
        out_route_buffer[0] = 0;
        ASSERT(false);  // caught only when watcher enabled. Otherwise make behavior consistent as returning false.
        return false;
    }

    // Get compressed route data (2 bytes: ns_hops, ew_hops, directions, turn_point)
    const auto& compressed_route = paths[dst_chip_id];
    uint8_t ns_hops = compressed_route.get_ns_hops();
    uint8_t ew_hops = compressed_route.get_ew_hops();
    uint8_t ns_dir = compressed_route.get_ns_direction();
    uint8_t ew_dir = compressed_route.get_ew_direction();

    if (ns_hops == 0 && ew_hops == 0) {
        // Noop to self
        out_route_buffer[0] = 0;
        out_route_buffer[1] = 0;
        return false;
    }

    // Use canonical 2D encoder to generate route buffer
    // Note: Buffer size is determined by packet header template
    constexpr uint32_t max_buffer_size = FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE;
    uint8_t temp_buffer[max_buffer_size];

    routing_encoding::encode_2d_unicast(
        ns_hops,
        ew_hops,
        ns_dir,
        ew_dir,
        temp_buffer,
        max_buffer_size,
        prepend_one_hop  // CRITICAL: Pass through prepend_one_hop for router usage
    );

    // Copy to volatile output
    for (uint32_t i = 0; i < max_buffer_size; i++) {
        out_route_buffer[i] = temp_buffer[i];
    }

    return true;
}

// Device-side decoder function for 1D routing (packed paths)
template <>
inline bool intra_mesh_routing_path_t<1, false>::decode_route_to_buffer(
    uint16_t dst_chip_id, volatile uint8_t* out_route_buffer, bool prepend_one_hop) const {
    // Determine number of words based on compile-time define
    constexpr uint32_t words_per_entry = FabricHeaderConfig::LOW_LATENCY_NUM_WORDS;

    // Bounds check
    if (dst_chip_id >= MAX_CHIPS_LOWLAT) {
        // Out of bounds - zero output
        volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(out_route_buffer);
        for (uint32_t i = 0; i < words_per_entry; i++) {
            out[i] = 0;
        }
        ASSERT(false && "dst_chip_id out of bounds");
        return false;
    }

    // Read from table with correct stride
    // Host populated this with the same stride (see compressed_routing_path.cpp)
    const uint32_t* table = reinterpret_cast<const uint32_t*>(&this->paths);
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(out_route_buffer);

    // Copy entry (words_per_entry words)
    const uint32_t* entry = &table[dst_chip_id * words_per_entry];
    for (uint32_t i = 0; i < words_per_entry; i++) {
        out[i] = entry[i];
    }

    return true;
}

// Device-side compressed decoder function for 1D routing
template <>
inline bool intra_mesh_routing_path_t<1, true>::decode_route_to_buffer(
    uint16_t hops, volatile uint8_t* out_route_buffer, bool prepend_one_hop) const {
    return true;
}

/**
 * Device-side on-the-fly generator for 1D routing (compressed=true path - DEFAULT)
 * Generates routing pattern directly without reading from a table.
 */
inline bool decode_route_to_buffer_by_hops(uint16_t hops, volatile uint8_t* out_route_buffer) {
    // Determine number of words based on compile-time define
    constexpr uint32_t num_words = FabricHeaderConfig::LOW_LATENCY_NUM_WORDS;
    constexpr uint32_t max_hops = num_words * RoutingFieldsConstants::LowLatency::BASE_HOPS;

    // Bounds check
    if (hops >= max_hops) {
        // Zero output
        volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(out_route_buffer);
        for (uint32_t i = 0; i < num_words; i++) {
            out[i] = 0;
        }
        ASSERT(false && "Hops exceeds max supported");
        return false;
    }

    // Generate routing pattern on-the-fly using canonical encoder
    uint32_t temp_buffer[num_words];
    routing_encoding::encode_1d_unicast(hops, temp_buffer, num_words);

    // Copy to volatile output
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(out_route_buffer);
    for (uint32_t i = 0; i < num_words; i++) {
        out[i] = temp_buffer[i];
    }

    return true;
}

// Device-side compressed decoder function for 1D routing
inline bool decode_route_to_buffer_by_dev(uint16_t dst_chip_id, volatile uint8_t* out_route_buffer) {
    tt_l1_ptr routing_l1_info_t* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    uint16_t my_device_id = routing_table->my_device_id;
    uint16_t hops = my_device_id > dst_chip_id ? my_device_id - dst_chip_id : dst_chip_id - my_device_id;

    return decode_route_to_buffer_by_hops(hops, out_route_buffer);
}

}  // namespace tt::tt_fabric
