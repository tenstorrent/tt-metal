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

// 2D compressed routing specialization: ControlPlane singleton-based implementation
template <>
void intra_mesh_routing_path_t<2, true>::calculate_chip_to_all_routing_fields(
    const FabricNodeId& src_fabric_node_id, uint16_t num_chips) {
    const auto& src_chip_id = src_fabric_node_id.chip_id;
    const auto& mesh_id = src_fabric_node_id.mesh_id;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    std::vector<chan_id_t> candidate_src_chan_ids;
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        auto chans = control_plane.get_active_fabric_eth_routing_planes_in_direction(src_fabric_node_id, dir);
        candidate_src_chan_ids.insert(candidate_src_chan_ids.end(), chans.begin(), chans.end());
    }

    for (uint16_t dst_chip_id = 0; dst_chip_id < num_chips; ++dst_chip_id) {
        if (dst_chip_id == src_chip_id) {
            paths[dst_chip_id].set(0, 0, 0, 0, 0);
            continue;
        }

        tt::tt_fabric::FabricNodeId dst_fabric_node_id(mesh_id, dst_chip_id);
        std::vector<uint16_t> best_chip_sequence;
        for (chan_id_t start_chan : candidate_src_chan_ids) {
            auto candidate_route = control_plane.get_fabric_route(src_fabric_node_id, dst_fabric_node_id, start_chan);
            if (candidate_route.empty()) {
                continue;
            }
            // Build chip sequence ("intra"-mesh only)
            std::vector<uint16_t> seq;
            seq.reserve(candidate_route.size());
            tt::tt_fabric::FabricNodeId last_added = src_fabric_node_id;
            for (const auto& step : candidate_route) {
                const tt::tt_fabric::FabricNodeId& node = step.first;
                if (node.mesh_id != mesh_id) {
                    break;  // ignore inter-mesh tail
                }
                if (node == last_added) {
                    continue;  // skip intra-chip channel change
                }
                seq.push_back(static_cast<uint16_t>(node.chip_id));
                last_added = node;
            }
            // pick up shortest path
            if (!seq.empty() && (best_chip_sequence.empty() || seq.size() < best_chip_sequence.size())) {
                best_chip_sequence.swap(seq);
            }
        }

        TT_ASSERT(
            !best_chip_sequence.empty(),
            "Failed to find intra-mesh route from chip {} to chip {} in mesh {}",
            src_chip_id,
            dst_chip_id,
            *mesh_id);

        uint8_t ns_hops = 0;
        uint8_t ew_hops = 0;
        uint8_t ns_direction = 0;
        uint8_t ew_direction = 0;
        // Z (intra-mesh skip link) is a dedicated dimension carrying at most one hop per route. Rather
        // than pin it to a specific axis, z_before records how many cardinal hops (NS then EW, in
        // dimension order) precede the Z hop. This lets the skip sit anywhere in the route: inside the
        // NS run (NS-axis skip) or inside the EW run (EW-axis skip, e.g. a wide 8x16 mesh). z_present
        // marks that the route contains a single Z hop.
        uint8_t z_present = 0;
        uint8_t z_before = 0;

        auto make_node = [mesh_id](uint16_t chip) { return tt::tt_fabric::FabricNodeId(mesh_id, chip); };
        auto next_dir = [&](uint16_t from_chip, uint16_t to_chip) {
            return control_plane.get_forwarding_direction(make_node(from_chip), make_node(to_chip));
        };

        bool seen_ns = false;
        bool seen_ew = false;
        bool seen_z = false;
        uint16_t prev_chip = src_chip_id;
        for (uint16_t curr_chip : best_chip_sequence) {
            auto dir_opt = next_dir(prev_chip, curr_chip);
            TT_ASSERT(
                dir_opt.has_value() && dir_opt.value() != RoutingDirection::NONE,
                "Invalid direction between chips {} and {}",
                prev_chip,
                curr_chip);
            const RoutingDirection d = dir_opt.value();
            if (d == RoutingDirection::N || d == RoutingDirection::S) {
                const uint8_t bit = (uint8_t)(d == RoutingDirection::S);
                if (!seen_ns) {
                    ns_direction = bit;
                    seen_ns = true;
                } else {
                    TT_ASSERT(
                        ns_direction == bit,
                        "Non-monotone NS traversal (with skip) is not supported: chip {} -> {}",
                        prev_chip,
                        curr_chip);
                }
                if (!seen_z) {
                    ++z_before;
                }
                ++ns_hops;
            } else if (d == RoutingDirection::E || d == RoutingDirection::W) {
                const uint8_t bit = (uint8_t)(d == RoutingDirection::E);
                if (!seen_ew) {
                    ew_direction = bit;
                    seen_ew = true;
                } else {
                    TT_ASSERT(
                        ew_direction == bit,
                        "Non-monotone EW traversal is not supported: chip {} -> {}",
                        prev_chip,
                        curr_chip);
                }
                if (!seen_z) {
                    ++z_before;
                }
                ++ew_hops;
            } else if (d == RoutingDirection::Z) {
                TT_ASSERT(!seen_z, "More than one Z (skip) hop per route is not supported: chip {}", curr_chip);
                z_present = 1;
                seen_z = true;
            } else {
                TT_ASSERT(false, "Unexpected routing direction between chips {} and {}", prev_chip, curr_chip);
            }
            prev_chip = curr_chip;
        }

        // turn_point marks the NS->EW turn position in the emitted route. A Z hop only shifts the turn
        // when it is spliced inside (or right after) the NS run, i.e. z_before <= ns_hops; an EW-axis
        // skip sits after the turn and does not move it.
        const uint8_t turn_point = (uint8_t)(ns_hops + ((z_present && z_before <= ns_hops) ? 1 : 0));
        paths[dst_chip_id].set(ns_hops, ew_hops, ns_direction, ew_direction, turn_point, z_present, z_before);
    }
}

}  // namespace tt::tt_fabric
