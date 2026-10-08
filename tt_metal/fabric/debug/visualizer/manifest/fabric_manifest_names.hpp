// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
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

// The manifest's spelling of an enumerator in lower case, e.g. FABRIC_2D is "fabric_2d".
inline std::string lower_name(std::string_view enumerator) {
    std::string name(enumerator);
    std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name;
}

// The manifest's spelling of an enum value: its enumerator's lower_name, e.g. FabricConfig::FABRIC_2D is
// "fabric_2d". A value with no enumerator name, such as a bitmask combination, is written as its number.
template <typename E>
std::string lower_enum_name(E value) {
    const auto enumerator = enchantum::to_string(value);
    if (enumerator.empty()) {
        return std::to_string(static_cast<std::underlying_type_t<E>>(value));
    }
    return lower_name(enumerator);
}

// The manifest's spelling of a field kind: its content type's name in snake case, e.g. content::L1 is "l1".
template <typename K>
std::string kind_name() {
    std::string name;
    for (const char c : enchantum::type_name<K>) {
        if (std::isupper(static_cast<unsigned char>(c)) && !name.empty()) {
            name += '_';
        }
        name += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return name;
}

template <typename... Kinds>
std::string kind_name(const std::variant<Kinds...>& kind) {
    return std::visit([]<typename K>(const K&) { return kind_name<K>(); }, kind);
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

// The size in bytes of one of a type's elements.
inline uint32_t element_size(const layout::Type& type) { return type.count == 0 ? type.size : type.size / type.count; }

// The prefixes of a schema that names a struct in the arch's types or an enum in enums, before the type's name.
inline constexpr std::string_view k_struct_schema = "struct:";
inline constexpr std::string_view k_enum_schema = "enum:";

// The manifest's spelling of a type's element, which L1 contents and type layouts both use. An element carries no
// width, so integers take theirs from the size of one element.
inline std::string schema_name(const layout::Type& type) {
    namespace element = layout::element;
    return std::visit(
        [&type](const auto& e) -> std::string {
            using E = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<E, element::Uint>) {
                return fmt::format("u{}", element_size(type) * 8);
            } else if constexpr (std::is_same_v<E, element::Int>) {
                return fmt::format("i{}", element_size(type) * 8);
            } else if constexpr (std::is_same_v<E, element::Enum>) {
                return fmt::format("{}{}", k_enum_schema, e.name);
            } else if constexpr (std::is_same_v<E, element::Struct>) {
                return fmt::format("{}{}", k_struct_schema, e.name);
            } else if constexpr (std::is_same_v<E, element::Packed>) {
                return fmt::format("packed:{}", e.table);
            } else if constexpr (std::is_same_v<E, element::Bytes>) {
                return "bytes";
            } else if constexpr (std::is_same_v<E, element::Pad>) {
                return "pad";
            } else { /* always fails if we get here */
                static_assert(!sizeof(E*), "schema_name has no spelling for this element");
            }
        },
        type.element);
}

}  // namespace tt::tt_fabric::manifest
