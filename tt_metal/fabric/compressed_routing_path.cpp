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
// Builds the destination-major 2D action-map route table from first-hop directions. Axis decomposition
// follows DOR: while rows differ the first hop is a Y move (N/S/Z) probed same-column, and once rows
// match it is an X move (E/W) probed same-row.
void route_table_2d_t::calculate_chip_to_all_routing_fields(
    const FabricNodeId& src_fabric_node_id, uint16_t num_chips) {
    const auto mesh_id = src_fabric_node_id.mesh_id;
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Global mesh geometry: device tables are indexed by global chip ids and id->coord is row-major
    // (y = id / x_size, x = id % x_size), mirroring compute_and_embed_2d_routing_path_table.
    const MeshShape mesh_shape = control_plane.get_physical_mesh_shape(mesh_id, MeshScope::GLOBAL);
    const uint32_t y_size = mesh_shape[0];
    const uint32_t x_size = mesh_shape[1];
    TT_FATAL(
        y_size * x_size == num_chips && num_chips > 0,
        "2D route table: mesh {} shape {}x{} does not match {} chips",
        *mesh_id,
        y_size,
        x_size,
        num_chips);

    // The packer zeroes only the live [y_size,x_size] action-map region; clear the full 2D route-table
    // slot so the memcpy into L1 is deterministic.
    std::memset(data, 0, sizeof(data));

    auto probe = [&control_plane, mesh_id](uint32_t src_chip, uint32_t dst_chip) {
        const auto dir =
            control_plane.get_forwarding_direction(FabricNodeId(mesh_id, src_chip), FabricNodeId(mesh_id, dst_chip));
        // TT_FATAL, not TT_ASSERT: TT_ASSERT compiles to a no-op in Release, and dir.value() on an
        // empty optional is then undefined behaviour that yields a garbage direction, which packs a
        // wrong-but-plausible routing table instead of failing.
        TT_FATAL(
            dir.has_value() && dir.value() != RoutingDirection::NONE,
            "2D route table: no first-hop direction from chip {} to chip {}",
            src_chip,
            dst_chip);
        return control_plane.routing_direction_to_eth_direction(dir.value());
    };
    // Representative column 0 for Y probes, representative row 0 for X probes.
    auto y_action = [&](uint32_t cur_y, uint32_t dst_y) { return probe(cur_y * x_size, dst_y * x_size); };
    auto x_action = [&](uint32_t cur_x, uint32_t dst_x) { return probe(cur_x, dst_x); };

    const bool ok = Routing2DCodec::pack_route_vectors(data, y_size, x_size, y_action, x_action);
    // TT_FATAL, not TT_ASSERT. This is the load-bearing one: TT_ASSERT is a no-op in Release, `data`
    // is memset to zero above, and a failed pack therefore embeds an ALL-ZERO routing table. Every
    // route buffer then widens to zeros, every router decodes action 0, action_is_valid() rejects it,
    // and nothing forwards -- a silent cluster-wide hang with senders stuck at 0 packets, rather than
    // a diagnosable error. Same fail-loud reasoning as the multicast one-feeder gate (D9.1).
    TT_FATAL(
        ok,
        "2D route table: mesh {} shape {}x{} is not representable in the destination-major 2D action-map format "
        "(an axis probe returned an off-axis direction, or the shape exceeds the bound)",
        *mesh_id,
        y_size,
        x_size);
}

}  // namespace tt::tt_fabric
