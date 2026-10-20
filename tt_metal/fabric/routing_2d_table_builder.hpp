// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

namespace routing_2d_table_builder_detail {

constexpr std::uint32_t action_vector_bytes(std::uint32_t y_size, std::uint32_t x_size) {
    return Routing2DCodec::table_bytes(y_size) + Routing2DCodec::table_bytes(x_size);
}

constexpr std::uint32_t mcast_tree_bytes(std::uint32_t y_size, std::uint32_t x_size) {
    return Routing2DCodec::MCAST_TREE_EDGE_BYTES *
           (Routing2DCodec::mcast_tree_edge_count(y_size) + Routing2DCodec::mcast_tree_edge_count(x_size));
}

}  // namespace routing_2d_table_builder_detail

// Host-side validation for the complete fixed-size 2D routing-table slot. Keep generation out of
// fabric_common.h: device code only consumes the packed representation.
constexpr bool is_valid_2d_route_table_shape(std::uint32_t y_size, std::uint32_t x_size) {
    if (y_size == 0 || x_size == 0 || y_size > Routing2DCodec::MAX_AXIS_SIZE ||
        x_size > Routing2DCodec::MAX_AXIS_SIZE) {
        return false;
    }

    // Divide before multiplying so even unchecked external dimensions cannot overflow.
    if (y_size > MAX_MESH_SIZE / x_size) {
        return false;
    }

    return routing_2d_table_builder_detail::action_vector_bytes(y_size, x_size) <=
               Routing2DCodec::ACTION_VECTOR_CAPACITY_BYTES &&
           routing_2d_table_builder_detail::mcast_tree_bytes(y_size, x_size) <=
               Routing2DCodec::MCAST_TREE_CAPACITY_BYTES;
}

namespace routing_2d_table_builder_detail {

inline void set_action_2bit(std::uint8_t* packed_row, std::uint32_t index, std::uint8_t action_2bit) {
    const std::uint32_t byte_index = index / Routing2DCodec::ACTIONS_PER_BYTE;
    const std::uint32_t shift = (index % Routing2DCodec::ACTIONS_PER_BYTE) * Routing2DCodec::BITS_PER_ACTION;
    packed_row[byte_index] =
        static_cast<std::uint8_t>((packed_row[byte_index] & ~(0b11u << shift)) | ((action_2bit & 0b11u) << shift));
}

inline std::uint8_t* y_row(std::uint8_t* table, std::uint32_t y_size, std::uint32_t dst_y) {
    return table + dst_y * Routing2DCodec::row_bytes(y_size);
}

inline std::uint8_t* x_row(std::uint8_t* table, std::uint32_t y_size, std::uint32_t x_size, std::uint32_t dst_x) {
    return table + Routing2DCodec::table_bytes(y_size) + dst_x * Routing2DCodec::row_bytes(x_size);
}

}  // namespace routing_2d_table_builder_detail

// Packs destination-major first-hop vectors. Y actions may be N/S/Z; X actions may be E/W/Z.
// Returns false without writing when the shape or output span is invalid. An off-axis action is a
// topology error and may leave the live span partially populated; ControlPlane treats it as fatal.
template <typename YActionSource, typename XActionSource>
inline bool pack_2d_route_vectors(
    std::uint8_t* out,
    std::size_t out_size,
    std::uint32_t y_size,
    std::uint32_t x_size,
    YActionSource y_action,
    XActionSource x_action) {
    if (!is_valid_2d_route_table_shape(y_size, x_size)) {
        return false;
    }

    const std::uint32_t action_vector_bytes = routing_2d_table_builder_detail::action_vector_bytes(y_size, x_size);
    if (out == nullptr || out_size < action_vector_bytes) {
        return false;
    }
    for (std::uint32_t i = 0; i < action_vector_bytes; ++i) {
        out[i] = 0;
    }

    for (std::uint32_t dst = 0; dst < y_size; ++dst) {
        std::uint8_t* row = routing_2d_table_builder_detail::y_row(out, y_size, dst);
        for (std::uint32_t cur = 0; cur < y_size; ++cur) {
            std::uint8_t action = Routing2DCodec::CODE_STOP;
            if (cur != dst) {
                action = Routing2DCodec::encode(Fabric2DAxis::Y, y_action(cur, dst));
                if (action == Routing2DCodec::INVALID_CODE) {
                    return false;
                }
            }
            routing_2d_table_builder_detail::set_action_2bit(row, cur, action);
        }
    }

    for (std::uint32_t dst = 0; dst < x_size; ++dst) {
        std::uint8_t* row = routing_2d_table_builder_detail::x_row(out, y_size, x_size, dst);
        for (std::uint32_t cur = 0; cur < x_size; ++cur) {
            std::uint8_t action = Routing2DCodec::CODE_STOP;
            if (cur != dst) {
                action = Routing2DCodec::encode(Fabric2DAxis::X, x_action(cur, dst));
                if (action == Routing2DCodec::INVALID_CODE) {
                    return false;
                }
            }
            routing_2d_table_builder_detail::set_action_2bit(row, cur, action);
        }
    }
    return true;
}

}  // namespace tt::tt_fabric
