// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric {

class ControlPlane;
class FabricEriscDatamoverBuilder;
struct ChipRoutingFacts;
struct RouterLocation;
struct RouterVcShape;

// Everything the collector reads about one router.
struct ManifestRouterInputs {
    const FabricEriscDatamoverBuilder& erisc_builder;
    const RouterVcShape& vc_shape;
    const RouterLocation& location;
    const ChipRoutingFacts& chip_facts;
    const ControlPlane& control_plane;
    // Indexed by RISC id, one per RISC the router runs.
    std::vector<std::unordered_map<std::string, uint32_t>> named_ct_args_per_risc;
};

// Collects a built router's manifest facts from its builders, checked against the compile-time arguments each of
// its RISCs receives.
manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs);

}  // namespace tt::tt_fabric
