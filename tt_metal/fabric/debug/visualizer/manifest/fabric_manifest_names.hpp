// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

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

inline std::string router_key(const RouterKey& key) {
    return fmt::format("{}{}", direction_letter(key.direction), key.routing_plane);
}

// The number in a key made of `prefix` and a number written as the writer writes it, such as 7 in "C7". Fails on any
// other key, including one whose number has a leading zero, since the writer never writes one.
inline uint32_t parse_key_number(std::string_view key, std::string_view prefix) {
    const std::string_view digits = key.substr(std::min(prefix.size(), key.size()));
    uint32_t number = 0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), number);
    TT_FATAL(
        key.starts_with(prefix) && !digits.empty() && error == std::errc{} && end == digits.data() + digits.size() &&
            (digits.size() == 1 || digits.front() != '0'),
        "Fabric manifest: {} is not a key {}<number>",
        key,
        prefix);
    return number;
}

inline MeshId parse_mesh_key(std::string_view key) { return MeshId{parse_key_number(key, "M")}; }

inline LogicalChipId parse_chip_key(std::string_view key) { return parse_key_number(key, "C"); }

inline eth_chan_directions parse_direction_letter(char letter) {
    switch (letter) {
        case 'E': return eth_chan_directions::EAST;
        case 'W': return eth_chan_directions::WEST;
        case 'N': return eth_chan_directions::NORTH;
        case 'S': return eth_chan_directions::SOUTH;
        case 'Z': return eth_chan_directions::Z;
        default: break;
    }
    TT_THROW("Fabric manifest: {} is not a router direction letter", letter);
}

inline RouterKey parse_router_key(std::string_view key) {
    TT_FATAL(!key.empty(), "Fabric manifest: a router key is empty");
    const uint32_t routing_plane = parse_key_number(key, key.substr(0, 1));
    TT_FATAL(
        std::in_range<routing_plane_id_t>(routing_plane),
        "Fabric manifest: router key {}'s routing plane is out of range",
        key);
    return {
        .direction = parse_direction_letter(key.front()),
        .routing_plane = static_cast<routing_plane_id_t>(routing_plane),
    };
}

}  // namespace tt::tt_fabric::manifest
