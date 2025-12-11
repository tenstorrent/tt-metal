// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

struct RelativeCoreCoord {
    long x = 0;
    long y = 0;

    std::string str() const;
};

constexpr bool operator==(const RelativeCoreCoord& a, const RelativeCoreCoord& b) { return a.x == b.x && a.y == b.y; }

constexpr bool operator!=(const RelativeCoreCoord& a, const RelativeCoreCoord& b) { return !(a == b); }

CoreCoord get_core_coord_from_relative(const RelativeCoreCoord& in, const CoreCoord& grid_size);

}  // namespace tt::tt_metal

template <>
struct std::hash<tt::tt_metal::RelativeCoreCoord> {
    std::size_t operator()(const tt::tt_metal::RelativeCoreCoord& o) const;
};

template <>
struct ttsl::json::to_json_t<tt::tt_metal::RelativeCoreCoord> {
    nlohmann::json operator()(const tt::tt_metal::RelativeCoreCoord& relative_core_coord) noexcept;
};

template <>
struct ttsl::json::from_json_t<tt::tt_metal::RelativeCoreCoord> {
    tt::tt_metal::RelativeCoreCoord operator()(const nlohmann::json& json) noexcept;
};
