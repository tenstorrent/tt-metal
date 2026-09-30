// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tt_metal/fabric/builder/fabric_manifest_model.hpp"

namespace tt::tt_fabric {

class FabricEriscDatamoverBuilder;
struct ChipRoutingFacts;
struct RouterLocation;
struct RouterVcShape;

// Collects a built router's manifest facts from its builders, checked against the compile-time
// arguments each of its RISCs receives.
manifest::Router collect_manifest_router(
    const FabricEriscDatamoverBuilder& erisc_builder,
    const RouterVcShape& vc_shape,
    const std::vector<std::unordered_map<std::string, uint32_t>>& named_ct_args_per_risc,
    const RouterLocation& location,
    const ChipRoutingFacts& chip_facts);

}  // namespace tt::tt_fabric
