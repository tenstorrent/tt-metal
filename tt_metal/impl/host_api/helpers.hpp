// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt_stl/overloaded.hpp>
#include "tracy/Tracy.hpp"

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal::detail {

inline CoreRangeSet GetCoreRangeSet(const std::variant<CoreCoord, CoreRange, CoreRangeSet>& specified_core_spec) {
    ZoneScoped;
    return std::visit(
        ttsl::overloaded{
            [](const CoreCoord& core_spec) { return CoreRangeSet(CoreRange(core_spec, core_spec)); },
            [](const CoreRange& core_spec) { return CoreRangeSet(core_spec); },
            [](const CoreRangeSet& core_spec) { return core_spec; },
        },
        specified_core_spec);
}

}  // namespace tt::tt_metal::detail
