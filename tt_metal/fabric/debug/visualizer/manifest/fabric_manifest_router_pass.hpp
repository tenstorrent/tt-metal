// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

class FabricEriscDatamoverBuilder;
struct RouterVcShape;

// The router pass's view of a router's kernel inputs.
struct RouterKernelInputs {
    // Shared by eriscs.
    std::map<std::string, std::string> defines;
    // Per-erisc.
    std::vector<tt::tt_metal::DataMovementProcessor> processors;
    // Per-erisc.
    std::vector<std::unordered_map<std::string, uint32_t>> named_ct_args;
};

// Builds RouterKernelInputs from the KernelInputs for each erisc in the router.
RouterKernelInputs make_router_kernel_inputs(const std::vector<KernelInputs>& kernel_inputs_by_erisc);

// Everything the router pass reads about one router.
struct ManifestRouterInputs {
    const FabricEriscDatamoverBuilder& erisc_builder;
    const RouterVcShape& vc_shape;
    const RouterLocation& location;
    const ChipRoutingFacts& chip_facts;
    // get_fabric_router_addresses_to_clear()
    const std::vector<size_t>& addresses_to_clear;
    const RouterKernelInputs& kernel;
    const tt::tt_metal::Hal& hal;
};

// Throws unless every named argument and define in `kernel` is read by the manifest or listed as unrecorded
// (fabric_manifest_fields.hpp).
void check_kernel_inputs_accounted(const RouterKernelInputs& kernel);

// Collects a built router's manifest facts from what its kernels were fed and from its builders.
manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs);

namespace manifest {

// Each chip's routers and chip-wide facts, published by the chip's FabricBuilder once its kernels are created and read
// by the topology pass when the manifest is written. The builder context owns it, since it outlives each FabricBuilder.
class PublishedChips {
public:
    explicit PublishedChips(size_t num_chips) : chips_by_physical_id_(num_chips) {}

    // Publishes `chip`, whose routers must be every router its builder initialized.
    void publish(ChipId physical_chip_id, Chip chip, uint32_t num_initialized_routers);
    bool has(ChipId physical_chip_id) const;
    const Chip& get(ChipId physical_chip_id) const;

private:
    std::vector<std::optional<Chip>> chips_by_physical_id_;
};

}  // namespace manifest

}  // namespace tt::tt_fabric
