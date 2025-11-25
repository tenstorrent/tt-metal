// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <umd/device/types/arch.hpp>
#include <set>
#include <variant>

#include "erisc_datamover_builder.hpp"
#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_core_placement.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "device.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "metal_soc_descriptor.h"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include <fmt/ranges.h>

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_fabric {

void build_tt_fabric_program(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program* fabric_program_ptr,
    std::unordered_map<tt::tt_fabric::chan_id_t, std::unique_ptr<tt::tt_fabric::FabricRouterBuilder>>& router_builders) {
    using namespace tt_fabric;
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const bool is_TG =
        (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG);
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());

    const auto& fabric_context = control_plane.get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();

    const auto configure_edm_builder_for_dispatch = [&](tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder) {
        constexpr uint32_t k_DispatchFabricRouterContextSwitchInterval = 16;
        // Dispatch requires a higher context switching freq to service slow dispatch / UMD / debug tools
        edm_builder.set_firmware_context_switch_interval(k_DispatchFabricRouterContextSwitchInterval);
        edm_builder.set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::INTERVAL);
    };

    std::unordered_map<RoutingDirection, std::vector<chan_id_t>> active_fabric_eth_channels;
    std::unordered_map<RoutingDirection, FabricNodeId> chip_neighbors;
    uint32_t num_intra_chip_neighbors = 0;

    const auto device_has_dispatch_tunnel = [&]() -> bool {
        auto mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
        // results are inclusive of the mmio_device_id so they will never be zero
        TT_ASSERT(!tunnels_from_mmio.empty());
        return (tunnels_from_mmio.size() - 1) > 0;
    }();

    auto is_dispatch_link = [&](auto eth_chan, uint32_t dispatch_link_idx) {
        auto link_idx = control_plane.get_routing_plane_id(fabric_node_id, eth_chan);
        return device_has_dispatch_tunnel && link_idx == dispatch_link_idx;
    };

    for (const auto& direction : tt::tt_fabric::FabricContext::routing_directions) {
        auto active_eth_chans =
            control_plane.get_active_fabric_eth_routing_planes_in_direction(fabric_node_id, direction);
        if (active_eth_chans.empty()) {
            continue;
        }
        auto neighbors = control_plane.get_chip_neighbors(fabric_node_id, direction);
        auto intra_chip_neighbors = neighbors.find(fabric_node_id.mesh_id);
        if (intra_chip_neighbors != neighbors.end()) {
            // only count the number of unique intra chip neighbors
            // we assume that all neighbors in a direction are the same
            num_intra_chip_neighbors++;
        }
        // assume same neighbor per direction
        TT_FATAL(neighbors.size() == 1, "Multiple neighbor meshes per direction is unsupported");
        TT_FATAL(
            std::set<ChipId>(neighbors.begin()->second.begin(), neighbors.begin()->second.end()).size() == 1,
            "Multiple neighbors per direction is currently unsupported");

        // 1D fabric only supports intramesh connections apart from TG gateways
        if (!is_2D_routing) {
            uint32_t has_inter_mesh_connections = intra_chip_neighbors == neighbors.end();
            if (is_TG && has_inter_mesh_connections) {
                // if active eth channels are found but no neighbor on the same mesh, then the neighbor should be the
                // gateway
                TT_FATAL(
                    active_eth_chans.size() == 1, "Found more than one active eth link b/w mmio and remote chip on TG");
            } else {
                TT_FATAL(!has_inter_mesh_connections, "1D routing does not support intermesh connections");
            }
        }

        FabricNodeId neighbor_fabric_node_id = FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
        chip_neighbors.emplace(direction, neighbor_fabric_node_id);

        active_fabric_eth_channels.insert({direction, active_eth_chans});
        log_debug(
            tt::LogMetal,
            "Building fabric router -> device (phys): {}, (logical): {}, direction: {}, active_eth_chans.size(): {}, "
            "active_eth_chans: [{}]",
            device->id(),
            control_plane.get_fabric_node_id_from_physical_chip_id(device->id()).chip_id,
            direction,
            active_eth_chans.size(),
            fmt::join(active_eth_chans, ", "));
    }

    if (active_fabric_eth_channels.empty()) {
        // Need at least 1 active fabric eth channel in at least 1 direction with a neighbor
        return;
    }

    const bool wrap_around_mesh = fabric_context.is_wrap_around_mesh(fabric_node_id.mesh_id);

    for (const auto& [direction, remote_fabric_node_id] : chip_neighbors) {
        // Create fabric tensix builder for this ethernet channel
        // Skip the link used by dispatch using relay mux API
        uint32_t dispatch_link_idx =
            tt::tt_metal::RelayMux::get_dispatch_link_index(fabric_node_id, remote_fabric_node_id, device);

        auto get_fabric_router_config = [&](bool is_dispatch_link, auto eth_direction) {
            auto fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::DISABLED;
            // if not the link used by dispatch, get the fabric router config with tensix extension.
            if (!is_dispatch_link) {
                fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
            }
            return fabric_context.get_fabric_router_config(
                fabric_tensix_config, eth_direction);
        };

        for (const auto& eth_chan : active_fabric_eth_channels[direction]) {
            auto eth_direction = control_plane.routing_direction_to_eth_direction(direction);
            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);

            bool dispatch_link = is_dispatch_link(eth_chan, dispatch_link_idx);
            const auto& curr_edm_config = get_fabric_router_config(dispatch_link, eth_direction);

            const auto topology = fabric_context.get_fabric_topology();
            auto router_builder = tt::tt_fabric::FabricRouterBuilder::build(
                device,
                *fabric_program_ptr,
                eth_logical_core,
                fabric_node_id,
                remote_fabric_node_id,
                curr_edm_config,
                eth_direction,
                dispatch_link,
                eth_chan,
                topology);
            router_builders.insert({eth_chan, std::move(router_builder)});
        }

        // Last link may be used by dispatch if there is tunneling
        // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
        if (!active_fabric_eth_channels[direction].empty() && device_has_dispatch_tunnel) {
            const auto dispatch_eth_chan = active_fabric_eth_channels[direction].back();
            configure_edm_builder_for_dispatch(router_builders.at(dispatch_eth_chan)->get_erisc_builder());
        }
    }

    const auto topology = fabric_context.get_fabric_topology();
    const bool is_galaxy = tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy();

    auto build_downstream_connections = [&](tt::tt_fabric::chan_id_t eth_chan_dir1,
                                            tt::tt_fabric::chan_id_t eth_chan_dir2) {
        auto& router_builder1 = router_builders.at(eth_chan_dir1);
        auto& router_builder2 = router_builders.at(eth_chan_dir2);

        router_builder1->connect_to_downstream_router_over_noc(*router_builder2, 0);
        router_builder2->connect_to_downstream_router_over_noc(*router_builder1, 0);
    };

    auto connect_downstream_builders = [&](RoutingDirection dir1, RoutingDirection dir2) {
        bool can_connect =
            (chip_neighbors.find(dir1) != chip_neighbors.end()) && (chip_neighbors.find(dir2) != chip_neighbors.end());
        if (can_connect) {
            auto eth_chans_dir1 = active_fabric_eth_channels.at(dir1);
            auto eth_chans_dir2 = active_fabric_eth_channels.at(dir2);

            // Hack for TG to connect the last routing plane correctly for dispatch
            // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
            if (is_TG && (eth_chans_dir1.size() != eth_chans_dir2.size())) {
                log_trace(tt::LogMetal, "applying hack for chip: {}", device->id());
                std::reverse(eth_chans_dir1.begin(), eth_chans_dir1.end());
                std::reverse(eth_chans_dir2.begin(), eth_chans_dir2.end());
            }

            // since tunneling cores are not guaraneteed to be reserved on the same routing plane, iterate through
            // the ordered eth channels in both directions
            uint32_t num_links = std::min(eth_chans_dir1.size(), eth_chans_dir2.size());
            for (uint32_t link = 0; link < num_links; link++) {
                auto eth_chan_dir1 = eth_chans_dir1[link];
                auto eth_chan_dir2 = eth_chans_dir2[link];

                auto& router_builder1 = router_builders.at(eth_chan_dir1);
                auto& router_builder2 = router_builders.at(eth_chan_dir2);

                build_downstream_connections(eth_chan_dir1, eth_chan_dir2);

                // select VC based on the current link
                auto& edm_builder1 = router_builder1->get_erisc_builder();
                auto& edm_builder2 = router_builder2->get_erisc_builder();
                auto edm_noc_vc = edm_builder1.config.DEFAULT_NOC_VC + (link % edm_builder1.config.NUM_EDM_NOC_VCS);
                edm_builder1.config.edm_noc_vc = edm_noc_vc;
                edm_builder2.config.edm_noc_vc = edm_noc_vc;

                tt::tt_fabric::core_placement::CorePlacementContext cctx{
                    .topology = topology,
                    .is_galaxy = is_galaxy,
                    .num_links = num_links,
                };
                tt::tt_fabric::core_placement::apply_core_placement_optimizations(
                    cctx, edm_builder1, edm_builder2, link);
            }
        }
    };

    if (is_2D_routing) {
        // 2D Routing
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::S);
        connect_downstream_builders(RoutingDirection::E, RoutingDirection::W);
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::E);
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::W);
        connect_downstream_builders(RoutingDirection::S, RoutingDirection::E);
        connect_downstream_builders(RoutingDirection::S, RoutingDirection::W);
    } else if (wrap_around_mesh && num_intra_chip_neighbors == 2) {
        // 1D Routing wrap the corner chips, fold the internal connections
        auto it = chip_neighbors.begin();
        auto dir1 = it->first;
        it++;
        auto dir2 = it->first;
        connect_downstream_builders(dir1, dir2);
    } else {
        // 1D Routing
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::S);
        connect_downstream_builders(RoutingDirection::E, RoutingDirection::W);
    }
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program(tt::tt_metal::IDevice* device) {
    std::unique_ptr<tt::tt_metal::Program> fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();
    std::unordered_map<tt::tt_fabric::chan_id_t, std::unique_ptr<tt::tt_fabric::FabricRouterBuilder>> router_builders;

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();

    build_tt_fabric_program(device, fabric_program_ptr.get(), router_builders);
    fabric_context.set_num_fabric_initialized_routers(device->id(), router_builders.size());
    if (router_builders.empty()) {
        return nullptr;
    }

    // Compile all fabric tensix builders through router builders
    if (tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
        tt::tt_fabric::FabricTensixConfig::DISABLED) {
        // Track which directions have been built (for UDM mode finalization)
        std::set<tt::tt_fabric::eth_chan_directions> built_directions;

        // First pass: compile all existing tensix builders (from active eth channels)
        for (auto& [eth_chan, router_builder] : router_builders) {
            if (router_builder->has_tensix_builder()) {
                router_builder->get_tensix_builder().create_and_compile(*fabric_program_ptr);

                // Register the direction of this tensix builder
                auto direction = router_builder->get_tensix_builder().get_direction();
                built_directions.insert(direction);
            }
        }

        // Second pass (UDM mode only): build and compile tensix builders for missing directions
        // Edge devices (e.g., top-left corner of a 4x2 mesh) only have east/south builders,
        // leaving north/west empty. We need to build and compile them for inter-mux communication.
        if (tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() ==
            tt::tt_fabric::FabricTensixConfig::UDM) {
            const auto& tensix_config = fabric_context.get_tensix_config();
            if (tensix_config.has_missing_directions(device->id())) {
                const auto& missing_directions = tensix_config.get_missing_directions(device->id());
                auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());

                for (const auto& [routing_plane_id, missing_dir] : missing_directions) {
                    log_warning(
                        tt::LogMetal,
                        "Building missing direction tensix builder for fabric_node {}, routing_plane {}, direction {}",
                        fabric_node_id,
                        routing_plane_id,
                        static_cast<uint32_t>(missing_dir));
                    // Build and compile tensix builder for this missing (routing_plane_id, direction) pair
                    auto tensix_builder = tt::tt_fabric::FabricTensixDatamoverBuilder::build_for_missing_direction(
                        device, *fabric_program_ptr, fabric_node_id, routing_plane_id, missing_dir);
                    tensix_builder.create_and_compile(*fabric_program_ptr);
                }
            }
        }
    }

    // for now it doesnt matter which channel is the master, so just pick the 1st in the map
    auto master_router_chan = router_builders.begin()->first;
    fabric_context.set_fabric_master_router_chan(device->id(), master_router_chan);

    uint32_t router_channels_mask = 0;
    for (const auto& [router_chan, _] : router_builders) {
        router_channels_mask += 0x1 << (uint32_t)router_chan;
    }

    std::map<std::string, std::string> defines = {};
    if (fabric_context.is_2D_routing_enabled()) {
        defines["FABRIC_2D"] = "";
    }

    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto num_enabled_eth_cores = router_builders.size();
    const auto num_enabled_risc_cores =
        router_builders.begin()->second->get_configured_risc_count();  // same across all eth cores
    size_t num_local_fabric_routers = num_enabled_eth_cores;
    for (auto& [eth_chan, router_builder] : router_builders) {
        auto& edm_builder = router_builder->get_erisc_builder();
        edm_builder.set_wait_for_host_signal(true);
        const std::vector<uint32_t> rt_args = edm_builder.get_runtime_args();
        for (uint32_t risc_id = 0; risc_id < num_enabled_risc_cores; risc_id++) {
            std::vector<uint32_t> ct_args = edm_builder.get_compile_time_args(risc_id);

            const auto is_master_risc_core = eth_chan == master_router_chan && (risc_id == 0);
            ct_args.push_back(is_master_risc_core);
            ct_args.push_back(master_router_chan);
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
                *fabric_program_ptr,
                "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
                eth_logical_core,
                tt::tt_metal::EthernetConfig{
                    .noc = edm_builder.config.risc_configs[risc_id].get_configured_noc(),
                    .processor = proc,
                    .compile_args = ct_args,
                    .defines = defines,
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

            tt::tt_metal::SetRuntimeArgs(*fabric_program_ptr, kernel, eth_logical_core, rt_args);
        }

        log_debug(
            tt::LogMetal,
            "Building fabric router -> device (phys): {}, (logical): {}, channel: {}, num_local_fabric_routers: {}",
            device->id(),
            control_plane.get_fabric_node_id_from_physical_chip_id(device->id()).chip_id,
            eth_chan,
            num_local_fabric_routers);
    }

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
    const auto addresses_to_clear = fabric_context.get_fabric_router_addresses_to_clear();
    const auto& router_config = fabric_context.get_fabric_router_config();
    std::vector<uint32_t> router_zero_buf(router_config.router_buffer_clear_size_words, 0);
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }
}

}  // namespace tt::tt_fabric
