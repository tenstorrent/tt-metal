// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compressed_routing_path.hpp"
#include <cstring>

namespace tt::tt_fabric {

// 1D routing specialization
template <>
void intra_mesh_routing_path_t<1, false>::calculate_chip_to_all_routing_fields(
    uint16_t src_chip_id, tt_metal::distributed::MeshShape& mesh_shape, FabricType torus_type) {
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
    uint16_t src_chip_id, tt_metal::distributed::MeshShape& mesh_shape, FabricType torus_type) {
    // No-op
}

// 2D compressed routing specialization
template <>
void intra_mesh_routing_path_t<2, true>::calculate_chip_to_all_routing_fields(
    uint16_t src_chip_id, tt_metal::distributed::MeshShape& mesh_shape, FabricType torus_type) {
    // Calculate NS dimension size (assuming rectangular grid)
    uint16_t num_chips = mesh_shape[0] * mesh_shape[1];
    uint8_t ew_dim = mesh_shape[1];
    uint8_t ns_dim = mesh_shape[0];
    // Axis-aware torus handling
    const bool torus_y = (torus_type == FabricType::TORUS_Y || torus_type == FabricType::TORUS_XY);
    const bool torus_x = (torus_type == FabricType::TORUS_X || torus_type == FabricType::TORUS_XY);

    for (uint16_t dst_chip_id = 0; dst_chip_id < num_chips; ++dst_chip_id) {
        if (src_chip_id == dst_chip_id) {
            // Self route - no movement needed
            paths[dst_chip_id].set(0, 0, 0, 0, 0);
            continue;
        }

        // Calculate 2D coordinates (row-major: row = id / width, col = id % width)
        uint16_t src_row = src_chip_id / ew_dim;
        uint16_t src_col = src_chip_id % ew_dim;
        uint16_t dst_row = dst_chip_id / ew_dim;
        uint16_t dst_col = dst_chip_id % ew_dim;

        uint8_t ns_hops, ew_hops;
        uint8_t ns_direction, ew_direction;

        // North-South (Y axis, mesh_coord[0])
        {
            uint8_t ns_direct = (dst_row > src_row) ? (dst_row - src_row) : (src_row - dst_row);
            if (torus_y && ns_direct != 0) {
                uint8_t ns_wrap = ns_dim - ns_direct;
                if (ns_direct < ns_wrap) {
                    ns_hops = ns_direct;
                    ns_direction = dst_row > src_row;  // 0=north, 1=south
                } else if (ns_direct > ns_wrap) {
                    ns_hops = ns_wrap;
                    ns_direction = src_row > dst_row;  // Reverse direction for wrap
                } else {
                    // Equal distance: choose North to match host tie-break
                    ns_hops = ns_direct;  // same as ns_wrap
                    ns_direction = 0;     // 0=north
                }
            } else {
                ns_hops = ns_direct;
                ns_direction = dst_row > src_row;  // 0=north, 1=south
            }
        }

        // East-West (X axis, mesh_coord[1])
        {
            uint8_t ew_direct = (dst_col > src_col) ? (dst_col - src_col) : (src_col - dst_col);
            if (torus_x && ew_direct != 0) {
                uint8_t ew_wrap = ew_dim - ew_direct;
                if (ew_direct < ew_wrap) {
                    ew_hops = ew_direct;
                    ew_direction = dst_col > src_col;  // 0=west, 1=east
                } else if (ew_direct > ew_wrap) {
                    ew_hops = ew_wrap;
                    ew_direction = src_col > dst_col;  // Reverse direction for wrap
                } else {
                    // Equal distance: choose East to match host tie-break
                    ew_hops = ew_direct;  // same as ew_wrap
                    ew_direction = 1;     // 1=east
                }
            } else {
                ew_hops = ew_direct;
                ew_direction = dst_col > src_col;  // 0=west, 1=east
            }
        }

        uint8_t turn_after_ns = ns_hops;  // XY routing: complete NS first, then EW
        paths[dst_chip_id].set(ns_hops, ew_hops, ns_direction, ew_direction, turn_after_ns);
    }
}

}  // namespace tt::tt_fabric
