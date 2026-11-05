// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_kernel_inputs.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

class FabricEriscDatamoverBuilder;
struct PacketHeaderConfig;
struct RouterVcShape;

// Everything the router pass reads about one router.
struct ManifestRouterInputs {
    const FabricEriscDatamoverBuilder& erisc_builder;
    const RouterVcShape& vc_shape;
    const RouterLocation& location;
    const ChipRoutingFacts& chip_facts;
    // The connections this router made to its siblings, as establish_connections_to_router recorded them.
    std::span<const RouterConnectionRecord> connections;
    // What create_kernel built each RISC's kernel with. The router pass reads what an argument carries from here, and
    // only uses the builders for info not carried in arguments.
    const manifest::RouterArgs& args;
    // The run's packet header (FabricContext::get_packet_header_config()), which starts every ring buffer slot.
    const PacketHeaderConfig& packet_header;
    const tt::tt_metal::Hal& hal;
};

// Throws unless every named argument and define in `kernel_inputs_by_erisc` is read by the manifest or listed as
// unrecorded (fabric_manifest_fields.hpp).
void check_kernel_inputs_accounted(std::span<const KernelInputs> kernel_inputs_by_erisc);

// Collects a built router's manifest facts from what its kernels were fed and from its builders.
manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs);

}  // namespace tt::tt_fabric
