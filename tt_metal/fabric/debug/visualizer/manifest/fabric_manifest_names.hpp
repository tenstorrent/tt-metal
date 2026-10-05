// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>
#include <variant>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"

// The names and keys the fabric manifest writes. Changing one changes the manifest format.
namespace tt::tt_fabric::manifest {

// The manifest's spelling of an enum value: its enumerator name in lower case, e.g.
// FabricConfig::FABRIC_2D is "fabric_2d". A value with no enumerator name, such as a bitmask combination, is
// written as its number.
template <typename E>
std::string lower_enum_name(E value) {
    const auto enumerator = enchantum::to_string(value);
    if (enumerator.empty()) {
        return std::to_string(static_cast<std::underlying_type_t<E>>(value));
    }
    std::string name(enumerator);
    std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name;
}

// Meshes, chips and routers are maps keyed M<mesh_id>, C<fabric_chip_id> and <direction><routing_plane>.

inline std::string mesh_key(MeshId mesh_id) { return fmt::format("M{}", *mesh_id); }

inline std::string chip_key(LogicalChipId fabric_chip_id) { return fmt::format("C{}", fabric_chip_id); }

inline char direction_letter(eth_chan_directions direction) {
    switch (direction) {
        case eth_chan_directions::EAST: return 'E';
        case eth_chan_directions::WEST: return 'W';
        case eth_chan_directions::NORTH: return 'N';
        case eth_chan_directions::SOUTH: return 'S';
        case eth_chan_directions::Z: return 'Z';
        default: break;
    }
    TT_THROW("Fabric manifest: {} is not a router direction", static_cast<int>(direction));
}

inline std::string router_key(eth_chan_directions direction, routing_plane_id_t routing_plane) {
    return fmt::format("{}{}", direction_letter(direction), routing_plane);
}

// The manifest's spelling of a FieldType, which regions and type layouts both use. FieldType carries no width,
// so integers take theirs from `element_size`, the size in bytes of one element.
inline std::string schema_name(const FieldType& type, uint32_t element_size) {
    return std::visit(
        [element_size](const auto& t) -> std::string {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, field::Uint>) {
                return fmt::format("u{}", element_size * 8);
            } else if constexpr (std::is_same_v<T, field::Int>) {
                return fmt::format("i{}", element_size * 8);
            } else if constexpr (std::is_same_v<T, field::Enum>) {
                return fmt::format("enum:{}", t.name);
            } else if constexpr (std::is_same_v<T, field::Struct>) {
                return fmt::format("struct:{}", t.name);
            } else if constexpr (std::is_same_v<T, field::Packed>) {
                return fmt::format("packed:{}", t.table);
            } else if constexpr (std::is_same_v<T, field::Bytes>) {
                return "bytes";
            } else {
                static_assert(std::is_same_v<T, field::Pad>);
                return "pad";
            }
        },
        type);
}

// The schema of an L1 element of type T.
template <typename T>
std::string schema_name_of() {
    return schema_name(field_type<T>(), sizeof(T));
}

}  // namespace tt::tt_fabric::manifest
