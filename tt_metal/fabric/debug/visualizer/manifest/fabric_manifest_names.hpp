// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>

#include <enchantum/enchantum.hpp>

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

}  // namespace tt::tt_fabric::manifest
