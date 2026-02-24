// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_fabric {

std::unique_ptr<tt::tt_fabric::ControlPlane> construct_control_plane(
    const std::filesystem::path& mesh_graph_desc_path,
    tt::Cluster& cluster,
    const ::tt::llrt::RunTimeOptions& rtoptions,
    const ::tt::tt_metal::Hal& hal,
    const tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const tt_fabric::FabricConfig& fabric_config,
    const tt_fabric::FabricReliabilityMode& fabric_reliability_mode,
    const tt_fabric::FabricTensixConfig& fabric_tensix_config,
    const tt_fabric::FabricUDMMode& fabric_udm_mode,
    const tt_fabric::FabricRouterConfig& fabric_router_config,
    const tt_fabric::FabricManagerMode& fabric_manager,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping.empty()) {
        log_info(tt::LogDistributed, "Using custom Fabric Node Id to physical chip mapping.");
        return std::make_unique<tt::tt_fabric::ControlPlane>(
            cluster,
            rtoptions,
            hal,
            distributed_context,
            mesh_graph_desc_path.string(),
            logical_mesh_chip_id_to_physical_chip_id_mapping,
            fabric_config,
            fabric_reliability_mode,
            fabric_tensix_config,
            fabric_udm_mode,
            fabric_router_config,
            fabric_manager);
    }

    return std::make_unique<tt::tt_fabric::ControlPlane>(
        cluster,
        rtoptions,
        hal,
        distributed_context,
        mesh_graph_desc_path.string(),
        fabric_config,
        fabric_reliability_mode,
        fabric_tensix_config,
        fabric_udm_mode,
        fabric_router_config,
        fabric_manager);
}

std::unique_ptr<tt::tt_fabric::ControlPlane> construct_control_plane(
    tt::Cluster& cluster,
    const ::tt::llrt::RunTimeOptions& rtoptions,
    const ::tt::tt_metal::Hal& hal,
    const tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const tt_fabric::FabricConfig& fabric_config,
    const tt_fabric::FabricReliabilityMode& fabric_reliability_mode,
    const tt_fabric::FabricTensixConfig& fabric_tensix_config,
    const tt_fabric::FabricUDMMode& fabric_udm_mode,
    const tt_fabric::FabricRouterConfig& fabric_router_config,
    const tt_fabric::FabricManagerMode& fabric_manager,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    // Use auto-discovery to generate mesh graph from physical system descriptor
    // This uses MeshGraph::generate_from_physical_system_descriptor which internally
    // uses map_mesh_to_physical to find a valid mapping
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping.empty()) {
        log_warning(
            tt::LogDistributed,
            "Custom Fabric Node Id to physical chip mapping provided but no mesh graph descriptor path. "
            "Mapping will be ignored. Please provide a custom mesh graph descriptor path for custom logical to "
            "physical mapping.");
    }
    log_info(tt::LogDistributed, "Constructing control plane using auto-discovery (no mesh graph descriptor).");
    return std::make_unique<tt::tt_fabric::ControlPlane>(
        cluster,
        rtoptions,
        hal,
        distributed_context,
        fabric_config,
        fabric_reliability_mode,
        fabric_tensix_config,
        fabric_udm_mode,
        fabric_router_config,
        fabric_manager);
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program(tt::tt_metal::IDevice* device) {
    auto fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();

    // Use FabricBuilder to coordinate the build phases
    FabricBuilder builder(device, *fabric_program_ptr, fabric_context);

    // Execute build phases
    builder.discover_channels();
    builder.create_routers();
    if (!builder.has_routers()) {
        return nullptr;
    }

    builder.connect_routers();
    builder.compile_ancillary_kernels();
    builder.create_kernels();

    // Compile the program
    tt::tt_metal::detail::CompileProgram(
        device, *fabric_program_ptr, tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch());

    return fabric_program_ptr;
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device) {
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        return create_and_compile_tt_fabric_program(device);
    }
    return nullptr;
}

void configure_fabric_cores(tt::tt_metal::IDevice* device) {
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto addresses_to_clear = builder_context.get_fabric_router_addresses_to_clear();
    const auto& router_config = builder_context.get_fabric_router_config();
    std::vector<uint32_t> router_zero_buf(router_config.router_buffer_clear_size_words, 0);
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }
}

}  // namespace tt::tt_fabric
