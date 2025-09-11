// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include <cstddef>

namespace tt::tt_fabric::core_placement {

struct CorePlacementContext {
    Topology topology{};
    bool is_galaxy{false};
    size_t num_links{0};
};

void apply_core_placement_optimizations(
    const CorePlacementContext& ctx,
    FabricEriscDatamoverBuilder& edm_fwd,
    FabricEriscDatamoverBuilder& edm_bwd,
    size_t link_index);

}  // namespace tt::tt_fabric::core_placement
