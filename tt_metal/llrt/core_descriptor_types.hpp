// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "common/core_coord.hpp"

namespace tt {

struct core_descriptor_t {
    CoreCoord compute_grid_size;
    std::vector<tt_metal::RelativeCoreCoord> relative_compute_cores;
    std::vector<tt_metal::RelativeCoreCoord> relative_dispatch_cores;
    std::vector<tt_metal::RelativeCoreCoord> relative_fabric_mux_cores;

    std::vector<CoreCoord> logical_compute_cores;
    std::vector<CoreCoord> logical_dispatch_cores;
    std::vector<CoreCoord> logical_fabric_mux_cores;
};

}  // namespace tt
