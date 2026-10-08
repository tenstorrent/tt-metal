// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric {

class FabricEriscDatamoverBuilder;
struct ChipRoutingFacts;
struct RouterLocation;
struct RouterVcShape;

// Information used in building router kernels.
struct RouterKernelInputs {
    // RISC-wide
    std::map<std::string, std::string> defines;
    // Per-RISC
    std::vector<tt::tt_metal::DataMovementProcessor> processors;
    std::vector<std::unordered_map<std::string, uint32_t>> named_ct_args;
};

// Everything the collector reads about one router.
struct ManifestRouterInputs {
    const FabricEriscDatamoverBuilder& erisc_builder;
    const RouterVcShape& vc_shape;
    const RouterLocation& location;
    const ChipRoutingFacts& chip_facts;
    // get_fabric_router_addresses_to_clear()
    const std::vector<size_t>& addresses_to_clear;
    const RouterKernelInputs& kernel;
};

// Collects a built router's manifest facts from what its kernels were fed and from its builders.
manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs);

}  // namespace tt::tt_fabric
