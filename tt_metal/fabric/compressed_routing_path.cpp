// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "compressed_routing_path.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/api/tt-metalium/control_plane.hpp"

namespace tt::tt_fabric {

// 1D routing specialization
template <>
void intra_mesh_routing_path_t<1, false>::calculate_chip_to_all_routing_fields(
    FabricNodeId src_fabric_node_id, tt_metal::distributed::MeshShape& mesh_shape) {
    uint32_t* route_ptr = reinterpret_cast<uint32_t*>(&paths);
    route_ptr[0] = 0;
    for (uint16_t hops = 1; hops < MAX_CHIPS_LOWLAT_1D; ++hops) {
        route_ptr[hops] =
            (FWD_ONLY_FIELD & ((1 << (hops - 1) * FIELD_WIDTH) - 1)) | (WRITE_ONLY << (hops - 1) * FIELD_WIDTH);
    }
}

// 1D compressed routing specialization. No-op
template <>
void intra_mesh_routing_path_t<1, true>::calculate_chip_to_all_routing_fields(
    FabricNodeId src_fabric_node_id, tt_metal::distributed::MeshShape& mesh_shape) {
    // No-op
}

// 2D compressed routing specialization: ControlPlane singleton-based implementation
template <>
void intra_mesh_routing_path_t<2, true>::calculate_chip_to_all_routing_fields(
    FabricNodeId src_fabric_node_id, tt_metal::distributed::MeshShape& mesh_shape) {
    uint16_t num_chips = mesh_shape[0] * mesh_shape[1];
    uint8_t ew_dim = mesh_shape[1];
    uint8_t ns_dim = mesh_shape[0];
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

        if (best_chip_sequence.empty()) {
            paths[dst_chip_id].set(0, 0, 0, 0, 0);
            continue;
        }

        uint8_t ns_hops = 0;
        uint8_t ew_hops = 0;
        uint8_t ns_direction = 0;
        uint8_t ew_direction = 0;

        uint16_t prev_chip = src_chip_id;
        for (uint16_t curr_chip : best_chip_sequence) {
            uint16_t prev_row = prev_chip / ew_dim;
            uint16_t prev_col = prev_chip % ew_dim;
            uint16_t curr_row = curr_chip / ew_dim;
            uint16_t curr_col = curr_chip % ew_dim;

            int dr = static_cast<int>(curr_row) - static_cast<int>(prev_row);
            int dc = static_cast<int>(curr_col) - static_cast<int>(prev_col);

            if (dr != 0) {
                uint8_t step_dir = (dr == 1 || dr == -(static_cast<int>(ns_dim) - 1)) ? 1 : 0;
                if (ns_hops == 0) {
                    ns_direction = step_dir;
                }
                ++ns_hops;
            } else if (dc != 0) {
                uint8_t step_dir = (dc == 1 || dc == -(static_cast<int>(ew_dim) - 1)) ? 1 : 0;
                if (ew_hops == 0) {
                    ew_direction = step_dir;
                }
                ++ew_hops;
            }
            prev_chip = curr_chip;
        }

        uint8_t turn_after_ns = ns_hops;
        paths[dst_chip_id].set(ns_hops, ew_hops, ns_direction, ew_direction, turn_after_ns);
    }
}

}  // namespace tt::tt_fabric
