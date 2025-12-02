// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/builder/fabric_core_placement.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "metal_soc_descriptor.h"
#include "tt_metal.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include <set>

namespace tt::tt_fabric {

// ============ Helper Functions ============

std::vector<RouterConnectionPair> get_router_connection_pairs(
    const std::unordered_map<RoutingDirection, std::vector<chan_id_t>>& channels_by_direction,
    const FabricContext& fabric_context,
    const std::unordered_map<RoutingDirection, FabricNodeId>& chip_neighbors,
    bool wrap_around_mesh) {
    std::vector<RouterConnectionPair> pairs;

    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();
    const size_t num_intra_chip_neighbors = chip_neighbors.size();

    // Helper to check if we can connect two directions
    auto can_connect = [&](RoutingDirection dir1, RoutingDirection dir2) {
        return chip_neighbors.count(dir1) > 0 && chip_neighbors.count(dir2) > 0 &&
               channels_by_direction.count(dir1) > 0 && channels_by_direction.count(dir2) > 0;
    };

    // Helper to add connection pairs for two directions
    auto add_direction_pairs = [&](RoutingDirection dir1, RoutingDirection dir2) {
        if (!can_connect(dir1, dir2)) {
            return;
        }

        const auto& chans_dir1 = channels_by_direction.at(dir1);
        const auto& chans_dir2 = channels_by_direction.at(dir2);
        uint32_t num_links = std::min(chans_dir1.size(), chans_dir2.size());

        for (uint32_t link = 0; link < num_links; link++) {
            pairs.push_back(RouterConnectionPair{
                .chan1 = chans_dir1[link],
                .chan2 = chans_dir2[link],
                .link_idx = link,
                .num_links = num_links,
            });
        }
    };

    if (is_2D_routing) {
        // 2D Routing - connect all orthogonal direction pairs
        add_direction_pairs(RoutingDirection::N, RoutingDirection::S);
        add_direction_pairs(RoutingDirection::E, RoutingDirection::W);
        add_direction_pairs(RoutingDirection::N, RoutingDirection::E);
        add_direction_pairs(RoutingDirection::N, RoutingDirection::W);
        add_direction_pairs(RoutingDirection::S, RoutingDirection::E);
        add_direction_pairs(RoutingDirection::S, RoutingDirection::W);
    } else if (wrap_around_mesh && num_intra_chip_neighbors == 2) {
        // 1D Routing wrap the corner chips, fold the internal connections
        auto it = chip_neighbors.begin();
        auto dir1 = it->first;
        it++;
        auto dir2 = it->first;
        add_direction_pairs(dir1, dir2);
    } else {
        // 1D Routing - connect opposite directions
        add_direction_pairs(RoutingDirection::N, RoutingDirection::S);
        add_direction_pairs(RoutingDirection::E, RoutingDirection::W);
    }

    return pairs;
}

// ============ FabricBuilder Implementation ============

FabricBuilder::FabricBuilder(
    tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, FabricContext& fabric_context) :
    device_(device),
    program_(program),
    fabric_context_(fabric_context),
    builder_context_(fabric_context.get_builder_context()),
    local_node_(tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_node_id_from_physical_chip_id(
        device->id())) {}

void FabricBuilder::create_routers() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const bool is_2D_routing = fabric_context_.is_2D_routing_enabled();

    // Determine if this device has tunneling dispatch (affects dispatch link selection)
    const auto device_has_dispatch_tunnel = [&]() -> bool {
        auto mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_->id());
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
        TT_ASSERT(!tunnels_from_mmio.empty());
        return (tunnels_from_mmio.size() - 1) > 0;
    }();

    auto is_dispatch_link = [&](chan_id_t eth_chan, uint32_t dispatch_link_idx) {
        auto link_idx = control_plane.get_routing_plane_id(local_node_, eth_chan);
        return device_has_dispatch_tunnel && link_idx == dispatch_link_idx;
    };

    // Build neighbors and active channels by iterating over directions
    for (const auto& direction : FabricContext::routing_directions) {
        auto active_eth_chans = control_plane.get_active_fabric_eth_routing_planes_in_direction(local_node_, direction);
        if (active_eth_chans.empty()) {
            continue;
        }

        auto neighbors = control_plane.get_chip_neighbors(local_node_, direction);
        auto intra_chip_neighbors = neighbors.find(local_node_.mesh_id);

        // Validation
        TT_FATAL(neighbors.size() == 1, "Multiple neighbor meshes per direction is unsupported");
        TT_FATAL(
            std::set<ChipId>(neighbors.begin()->second.begin(), neighbors.begin()->second.end()).size() == 1,
            "Multiple neighbors per direction is currently unsupported");

        // 1D fabric only supports intramesh connections
        if (!is_2D_routing) {
            bool has_inter_mesh_connections = (intra_chip_neighbors == neighbors.end());
            TT_FATAL(!has_inter_mesh_connections, "1D routing does not support intermesh connections");
        }

        FabricNodeId neighbor_fabric_node_id = FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
        chip_neighbors_.emplace(direction, neighbor_fabric_node_id);
        channels_by_direction_[direction] = active_eth_chans;

        // Get dispatch link index for this direction
        uint32_t dispatch_link_idx =
            tt::tt_metal::RelayMux::get_dispatch_link_index(local_node_, neighbor_fabric_node_id, device_);

        for (const auto& eth_chan : active_eth_chans) {
            bool dispatch_link = is_dispatch_link(eth_chan, dispatch_link_idx);

            // Use RouterLocation + RouterBuildSpec abstractions
            auto location = RouterLocation::create(eth_chan, neighbor_fabric_node_id, direction, dispatch_link);
            auto spec = builder_context_.get_router_build_spec(location, local_node_);

            // Use factory method - will route to ComputeMeshRouterBuilder or SwitchMeshRouterBuilder
            auto router_builder = FabricRouterBuilder::create(device_, program_, local_node_, location, spec);
            routers_.insert({eth_chan, std::move(router_builder)});
        }

        // Configure dispatch link if present
        // Dispatch requires higher context switching frequency to service slow dispatch / UMD / debug tools
        if (!active_eth_chans.empty() && device_has_dispatch_tunnel) {
            constexpr uint32_t k_DispatchFabricRouterContextSwitchInterval = 16;
            const auto dispatch_eth_chan = active_eth_chans.back();
            auto& edm_builder = routers_.at(dispatch_eth_chan)->get_erisc_builder();
            edm_builder.set_firmware_context_switch_interval(k_DispatchFabricRouterContextSwitchInterval);
            edm_builder.set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::INTERVAL);
        }
    }

    wrap_around_mesh_ = fabric_context_.is_wrap_around_mesh(local_node_.mesh_id);

    // Record master router channel
    if (!routers_.empty()) {
        master_router_chan_ = routers_.begin()->first;
    }
}

void FabricBuilder::connect_routers() {
    const auto topology = fabric_context_.get_fabric_topology();
    const bool is_galaxy = tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy();

    // Get connection pairs based on topology
    auto connection_pairs =
        get_router_connection_pairs(channels_by_direction_, fabric_context_, chip_neighbors_, wrap_around_mesh_);

    // Connect each pair
    for (const auto& pair : connection_pairs) {
        auto& router1 = routers_.at(pair.chan1);
        auto& router2 = routers_.at(pair.chan2);

        // Bidirectional VC0 connections
        router1->connect_to_downstream_router_over_noc(*router2, 0);
        router2->connect_to_downstream_router_over_noc(*router1, 0);

        // Configure NOC VC based on link index
        auto& edm_builder1 = router1->get_erisc_builder();
        auto& edm_builder2 = router2->get_erisc_builder();
        auto edm_noc_vc = edm_builder1.config.DEFAULT_NOC_VC + (pair.link_idx % edm_builder1.config.NUM_EDM_NOC_VCS);
        edm_builder1.config.edm_noc_vc = edm_noc_vc;
        edm_builder2.config.edm_noc_vc = edm_noc_vc;

        // Apply core placement optimizations
        tt::tt_fabric::core_placement::CorePlacementContext cctx{
            .topology = topology,
            .is_galaxy = is_galaxy,
            .num_links = pair.num_links,
        };
        tt::tt_fabric::core_placement::apply_core_placement_optimizations(
            cctx, edm_builder1, edm_builder2, pair.link_idx);
    }
}

void FabricBuilder::compile_ancillary_kernels() {
    if (tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() ==
        tt::tt_fabric::FabricTensixConfig::DISABLED) {
        return;
    }

    for (auto& [eth_chan, router_builder] : routers_) {
        if (router_builder->has_tensix_builder()) {
            router_builder->get_tensix_builder().create_and_compile(program_);
        }
    }
}

void FabricBuilder::create_kernels() {
    std::map<std::string, std::string> defines = {};
    if (fabric_context_.is_2D_routing_enabled()) {
        defines["FABRIC_2D"] = "";
    }

    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_->id());
    const auto num_enabled_eth_cores = routers_.size();
    const auto num_enabled_risc_cores = routers_.begin()->second->get_configured_risc_count();

    uint32_t router_channels_mask = 0;
    for (const auto& [router_chan, _] : routers_) {
        router_channels_mask += 0x1 << static_cast<uint32_t>(router_chan);
    }

    size_t num_local_fabric_routers = num_enabled_eth_cores;

    for (auto& [eth_chan, router_builder] : routers_) {
        auto& edm_builder = router_builder->get_erisc_builder();
        edm_builder.set_wait_for_host_signal(true);
        const std::vector<uint32_t> rt_args = edm_builder.get_runtime_args();

        for (uint32_t risc_id = 0; risc_id < num_enabled_risc_cores; risc_id++) {
            std::vector<uint32_t> ct_args = edm_builder.get_compile_time_args(risc_id);

            const auto is_master_risc_core = eth_chan == master_router_chan_ && (risc_id == 0);
            ct_args.push_back(is_master_risc_core);
            ct_args.push_back(master_router_chan_);
            ct_args.push_back(num_local_fabric_routers);
            ct_args.push_back(router_channels_mask);

            auto proc = static_cast<tt::tt_metal::DataMovementProcessor>(risc_id);
            if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
                tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode() &&
                num_enabled_risc_cores == 1) {
                // Force fabric to run on erisc1 due to stack usage exceeded with MUX on erisc0
                proc = tt::tt_metal::DataMovementProcessor::RISCV_1;
            }

            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
            auto kernel = tt::tt_metal::CreateKernel(
                program_,
                "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
                eth_logical_core,
                tt::tt_metal::EthernetConfig{
                    .noc = edm_builder.config.risc_configs[risc_id].get_configured_noc(),
                    .processor = proc,
                    .compile_args = ct_args,
                    .defines = defines,
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

            tt::tt_metal::SetRuntimeArgs(program_, kernel, eth_logical_core, rt_args);
        }

        log_debug(
            tt::LogMetal,
            "Fabric router created for device {}: local_node={}, eth_chan={}, direction={}",
            device_->id(),
            local_node_.to_string(),
            eth_chan,
            edm_builder.get_direction());
    }
}

void FabricBuilder::finalize_build_state() {
    builder_context_.set_num_fabric_initialized_routers(device_->id(), routers_.size());
    if (!routers_.empty()) {
        builder_context_.set_fabric_master_router_chan(device_->id(), master_router_chan_);
    }
}

}  // namespace tt::tt_fabric
