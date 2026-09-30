// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/assert.hpp>

// The names and keys the fabric manifest writes. Changing one changes the manifest format.
namespace tt::tt_fabric::manifest {

// The manifest's spelling of an enum value: its enumerator name in lower case, e.g.
// EdgeCapability::INTRAMESH_EXPRESS is "intramesh_express". A value with no enumerator name, such as a bitmask
// combination, is written as its number.
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

inline std::string chip_key(ChipId fabric_chip_id) { return fmt::format("C{}", fabric_chip_id); }

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

}  // namespace tt::tt_fabric::manifest
