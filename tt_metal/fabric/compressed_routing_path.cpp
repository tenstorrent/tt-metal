// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compressed_routing_path.hpp"
#include <cstring>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

// 1D routing specialization
template <>
void intra_mesh_routing_path_t<1, false>::calculate_chip_to_all_routing_fields(
    uint16_t src_chip_id,
    tt_metal::distributed::MeshShape& mesh_shape,
    FabricType torus_type,
    const std::vector<std::vector<RoutingDirection>>* /*first_hop_table*/) {
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
    uint16_t src_chip_id,
    tt_metal::distributed::MeshShape& mesh_shape,
    FabricType torus_type,
    const std::vector<std::vector<RoutingDirection>>* /*first_hop_table*/) {
    // No-op
}

// 2D compressed routing specialization
template <>
void intra_mesh_routing_path_t<2, true>::calculate_chip_to_all_routing_fields(
    uint16_t src_chip_id,
    tt_metal::distributed::MeshShape& mesh_shape,
    FabricType torus_type,
    const std::vector<std::vector<RoutingDirection>>* first_hop_table) {
    TT_ASSERT(first_hop_table != nullptr, "intra_mesh first-hop table must be provided");
    uint16_t num_chips = mesh_shape[0] * mesh_shape[1];
    uint8_t ew_dim = mesh_shape[1];
    uint8_t ns_dim = mesh_shape[0];

    const bool torus_y = (torus_type == FabricType::TORUS_Y || torus_type == FabricType::TORUS_XY);
    const bool torus_x = (torus_type == FabricType::TORUS_X || torus_type == FabricType::TORUS_XY);

    for (uint16_t dst_chip_id = 0; dst_chip_id < num_chips; ++dst_chip_id) {
        if (src_chip_id == dst_chip_id) {
            paths[dst_chip_id].set(0, 0, 0, 0, 0);
            continue;
        }

        {
            // Follow precomputed first-hop table exactly to unify tie-breaks
            uint16_t cur = src_chip_id;
            uint8_t ns_hops = 0;
            uint8_t ew_hops = 0;
            uint8_t ns_direction = 0;  // 0=north, 1=south
            uint8_t ew_direction = 0;  // 0=west, 1=east
            bool ns_dir_set = false;
            bool ew_dir_set = false;

            while (cur != dst_chip_id) {
                RoutingDirection dir = (*first_hop_table)[cur][dst_chip_id];
                uint16_t cur_row = cur / ew_dim;
                uint16_t cur_col = cur % ew_dim;
                switch (dir) {
                    case RoutingDirection::N:
                        if (!ns_dir_set) {
                            ns_direction = 0;
                            ns_dir_set = true;
                        }
                        ns_hops++;
                        cur_row = torus_y ? static_cast<uint16_t>((cur_row + ns_dim - 1) % ns_dim)
                                          : static_cast<uint16_t>(cur_row - 1);
                        cur = static_cast<uint16_t>(cur_row * ew_dim + cur_col);
                        break;
                    case RoutingDirection::S:
                        if (!ns_dir_set) {
                            ns_direction = 1;
                            ns_dir_set = true;
                        }
                        ns_hops++;
                        cur_row = torus_y ? static_cast<uint16_t>((cur_row + 1) % ns_dim)
                                          : static_cast<uint16_t>(cur_row + 1);
                        cur = static_cast<uint16_t>(cur_row * ew_dim + cur_col);
                        break;
                    case RoutingDirection::E:
                        if (!ew_dir_set) {
                            ew_direction = 1;
                            ew_dir_set = true;
                        }
                        ew_hops++;
                        cur_col = torus_x ? static_cast<uint16_t>((cur_col + 1) % ew_dim)
                                          : static_cast<uint16_t>(cur_col + 1);
                        cur = static_cast<uint16_t>(cur_row * ew_dim + cur_col);
                        break;
                    case RoutingDirection::W:
                        if (!ew_dir_set) {
                            ew_direction = 0;
                            ew_dir_set = true;
                        }
                        ew_hops++;
                        cur_col = torus_x ? static_cast<uint16_t>((cur_col + ew_dim - 1) % ew_dim)
                                          : static_cast<uint16_t>(cur_col - 1);
                        cur = static_cast<uint16_t>(cur_row * ew_dim + cur_col);
                        break;
                    default:
                        // Should not happen for valid table entries
                        // Treat as no-op to avoid undefined behavior
                        cur = dst_chip_id;  // exit
                        break;
                }
            }
            uint8_t turn_after_ns = ns_hops;  // XY routing per table
            paths[dst_chip_id].set(ns_hops, ew_hops, ns_direction, ew_direction, turn_after_ns);
            continue;
        }
    }
}

}  // namespace tt::tt_fabric
