// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "compressed_routing_path.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

// 1D uncompressed routing specialization
template <>
void intra_mesh_routing_path_t<1, false>::calculate_chip_to_all_routing_fields(
    const FabricNodeId& /*src_fabric_node_id*/, uint16_t num_chips) {
    // Zero-initialize entire 256-byte buffer
    std::memset(&paths, 0, sizeof(paths));

    // Query FabricContext to determine routing mode (16-hop vs 32-hop)
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();
    uint32_t extension_words = fabric_context.get_1d_pkt_hdr_extension_words();

    // Calculate words per entry and populate table
    // 16-hop mode: 1 word (4 bytes), 32-hop mode: 2 words (8 bytes)
    uint32_t words_per_entry = 1 + extension_words;
    uint32_t* buffer = reinterpret_cast<uint32_t*>(&paths);

    // Generate routing pattern for each chip
    for (uint16_t hops = 0; hops < num_chips; ++hops) {
        // Use canonical encoder with correct stride
        routing_encoding::encode_1d_unicast(
            hops,
            &buffer[hops * words_per_entry],  // Offset to this entry's location
            words_per_entry                   // Number of words to generate
        );
    }
}

// 1D compressed routing specialization. No-op
template <>
void intra_mesh_routing_path_t<1, true>::calculate_chip_to_all_routing_fields(
    const FabricNodeId& /*src_fabric_node_id*/, uint16_t /*num_chips*/) {
    // No-op
}

}  // namespace tt::tt_fabric
