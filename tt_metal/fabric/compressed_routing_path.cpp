// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "compressed_routing_path.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/api/tt-metalium/control_plane.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

// 1D routing specialization
template <>
void intra_mesh_routing_path_t<1, false>::calculate_chip_to_all_routing_fields(
    const FabricNodeId& src_fabric_node_id, uint16_t num_chips) {
    uint32_t* route_ptr = reinterpret_cast<uint32_t*>(&paths);
    route_ptr[0] = 0;
    for (uint16_t hops = 1; hops < num_chips; ++hops) {
        route_ptr[hops] =
            (FWD_ONLY_FIELD & ((1 << (hops - 1) * FIELD_WIDTH) - 1)) | (WRITE_ONLY << (hops - 1) * FIELD_WIDTH);
    }
}

// 1D compressed routing specialization. No-op
template <>
void intra_mesh_routing_path_t<1, true>::calculate_chip_to_all_routing_fields(
    const FabricNodeId& src_fabric_node_id, uint16_t num_chips) {
    // No-op
}

// 2D compressed routing specialization: ControlPlane singleton-based implementation
template <>
void intra_mesh_routing_path_t<2, true>::calculate_chip_to_all_routing_fields(
    const FabricNodeId& src_fabric_node_id, uint16_t num_chips) {
    auto& src_chip_id = src_fabric_node_id.chip_id;
    auto& mesh_id = src_fabric_node_id.mesh_id;

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

        auto is_ns = [](RoutingDirection d) { return d == RoutingDirection::N || d == RoutingDirection::S; };
        auto is_ew = [](RoutingDirection d) { return d == RoutingDirection::E || d == RoutingDirection::W; };

        auto make_node = [mesh_id](uint16_t chip) { return tt::tt_fabric::FabricNodeId(mesh_id, chip); };
        auto next_dir = [&](uint16_t from_chip, uint16_t to_chip) {
            return control_plane.get_forwarding_direction(make_node(from_chip), make_node(to_chip));
        };
        auto ns_bit = [](RoutingDirection d) { return (uint8_t)(d == RoutingDirection::S); };
        auto ew_bit = [](RoutingDirection d) { return (uint8_t)(d == RoutingDirection::E); };
        auto it = best_chip_sequence.cbegin();
        uint16_t prev_chip = src_chip_id;
        auto consume_axis = [&](auto is_axis, auto dir_to_bit, uint8_t& hops, uint8_t& dir_bit) {
            if (it == best_chip_sequence.cend()) {
                return;
            }
            uint16_t curr_chip = *it;
            auto dir_opt = next_dir(prev_chip, curr_chip);
            if (!dir_opt.has_value() || dir_opt.value() == RoutingDirection::NONE) {
                TT_ASSERT(false, "Invalid direction between chips {} and {}", prev_chip, curr_chip);
            }
            if (!is_axis(*dir_opt)) {
                return;  // start of other axis
            }

            dir_bit = dir_to_bit(*dir_opt);
            do {
                ++hops;
                prev_chip = curr_chip;
                ++it;
                if (it == best_chip_sequence.cend()) {
                    break;
                }
                curr_chip = *it;
                dir_opt = next_dir(prev_chip, curr_chip);
                if (!dir_opt.has_value()) {
                    break;
                }
            } while (dir_opt.has_value() && is_axis(*dir_opt));
        };

        if (it != best_chip_sequence.cend()) {
            // Consume NS first (if present), then EW
            consume_axis(is_ns, ns_bit, ns_hops, ns_direction);
            consume_axis(is_ew, ew_bit, ew_hops, ew_direction);
        }

        paths[dst_chip_id].set(ns_hops, ew_hops, ns_direction, ew_direction, ns_hops);
    }
}

}  // namespace tt::tt_fabric
