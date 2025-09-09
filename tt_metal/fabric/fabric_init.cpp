// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric.hpp"

#include <variant>

#include "erisc_datamover_builder.hpp"
#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/builder/fabric_core_placement.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "device.hpp"
#include "control_plane.hpp"
#include "metal_soc_descriptor.h"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_fabric {

std::pair<tt::tt_fabric::FabricEriscDatamoverType, tt::tt_fabric::FabricEriscDatamoverAxis> get_fabric_edm_type(
    const tt::tt_fabric::ControlPlane& control_plane,
    const tt::tt_fabric::RoutingDirection direction,
    tt::tt_fabric::MeshId mesh_id0,
    tt::tt_fabric::MeshId mesh_id1,
    chip_id_t chip0,
    chip_id_t chip1,
    bool wrap_around_mesh) {
    auto fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Default;
    auto fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Short;

    const auto& fabric_context = control_plane.get_fabric_context();

    const auto eth_chan_direction = control_plane.routing_direction_to_eth_direction(direction);
    if (mesh_id0 != mesh_id1 || !fabric_context.need_deadlock_avoidance_support(eth_chan_direction)) {
        return {fabric_edm_type, fabric_edm_axis};
    }

    auto physical_mesh_shape = control_plane.get_physical_mesh_shape(mesh_id0);
    TT_FATAL(physical_mesh_shape.dims() == 2, "Dateline routing only supported for 2D mesh");

    auto mesh_num_rows = physical_mesh_shape[0];
    auto mesh_num_columns = physical_mesh_shape[1];

    auto smaller_chip_id = std::min(chip0, chip1);
    auto larger_chip_id = std::max(chip0, chip1);

    // Refactor this once mesh_id0 has row/col control
    // wrap_around_mesh is used to fold the edm connections on the corner chips of a 2D mesh to form an outer ring of
    // devices on the mesh.
    if (wrap_around_mesh) {
        // Wrap around dateline
        if (smaller_chip_id == 0 && larger_chip_id == mesh_num_columns) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
        } else if ((chip0 == 0 || chip0 == mesh_num_columns) && chip1 == chip0 + 1) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
        } else if ((chip1 == 0 || chip1 == mesh_num_columns) && chip0 == chip1 + 1) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
        }
        // check if edm is on the longer axis
        if ((mesh_num_rows * mesh_num_columns) >=
            tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD) {
            fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Long;
        }
    } else {
        bool is_dateline_edm_along_column =
            smaller_chip_id % mesh_num_columns == 0 && larger_chip_id == (smaller_chip_id + mesh_num_columns - 1);
        bool is_dateline_edm_along_row = smaller_chip_id < mesh_num_columns &&
                                         larger_chip_id >= (mesh_num_columns * (mesh_num_rows - 1)) &&
                                         smaller_chip_id == larger_chip_id % mesh_num_columns;
        bool is_dateline_upstream_edm_along_column =
            (chip0 % mesh_num_columns == 0 && chip1 == chip0 + 1) ||
            (chip0 % mesh_num_columns == mesh_num_columns - 1 && chip1 == chip0 - 1);
        bool is_dateline_upstream_edm_along_row =
            (chip0 < mesh_num_columns && chip1 == chip0 + mesh_num_columns) ||
            (chip0 >= (mesh_num_columns * (mesh_num_rows - 1)) && chip1 == chip0 - mesh_num_columns);
        bool is_dateline_upstream_adjacent_edm_along_column =
            (chip1 % mesh_num_columns == 0 && chip0 == chip1 + 1) ||
            (chip1 % mesh_num_columns == mesh_num_columns - 1 && chip0 == chip1 - 1);
        bool is_dateline_upstream_adjacent_edm_along_row =
            (chip1 < mesh_num_columns && chip0 == chip1 + mesh_num_columns) ||
            (chip1 >= (mesh_num_columns * (mesh_num_rows - 1)) && chip0 == chip1 - mesh_num_columns);
        bool is_edm_along_row = ((larger_chip_id - smaller_chip_id) == mesh_num_columns) ||
                                (smaller_chip_id == larger_chip_id % mesh_num_columns);

        // Column dateline
        if (is_dateline_edm_along_column) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
        }
        // Row dateline
        else if (is_dateline_edm_along_row) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
        }
        // Column dateline upstream
        else if (is_dateline_upstream_edm_along_column) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
        }
        // Row dateline upstream
        else if (is_dateline_upstream_edm_along_row) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
        }
        // Column dateline upstream adjacent
        else if (is_dateline_upstream_adjacent_edm_along_column) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
        }
        // Row dateline upstream adjacent
        else if (is_dateline_upstream_adjacent_edm_along_row) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
        }

        // check if edm is on the longer axis
        if ((mesh_num_columns >= tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD &&
             !is_edm_along_row) ||
            (mesh_num_rows >= tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD &&
             is_edm_along_row)) {
            fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Long;
        }
    }

    if (fabric_context.is_2D_routing_enabled()) {
        // for 2D fabric, we need to re-work the buffer space optimization, cannot use 1D optimizations because
        // of more number of sender channels in 2D
        // only handling default and dateline edm types for now
        if (fabric_edm_type != tt::tt_fabric::FabricEriscDatamoverType::Default &&
            fabric_edm_type != tt::tt_fabric::FabricEriscDatamoverType::Dateline) {
            // reset to default if set to a non-dateline config
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Default;
            fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Short;
        }
    }

    return {fabric_edm_type, fabric_edm_axis};
}

void build_tt_fabric_program(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program* fabric_program_ptr,
    std::unordered_map<tt::tt_fabric::chan_id_t, tt::tt_fabric::FabricEriscDatamoverBuilder>& edm_builders,
    std::unordered_map<tt::tt_fabric::chan_id_t, tt::tt_fabric::FabricTensixDatamoverBuilder>& tensix_builders) {
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

    if (is_TG && device->is_mmio_capable()) {
        const auto& edm_config = fabric_context.get_fabric_router_config();
        auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
        for (const auto& [eth_chan, eth_direction] : router_chans_and_direction) {
            // remote_fabric_node_id is only used to determine the handshake master, no functional impact
            // for now treat the mmio chips as the handshake master
            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
            auto edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
                device,
                *fabric_program_ptr,
                eth_logical_core,
                fabric_node_id,
                FabricNodeId{fabric_node_id.mesh_id, fabric_node_id.chip_id + 1},
                edm_config,
                false,                                            /* build_in_worker_connection_mode */
                tt::tt_fabric::FabricEriscDatamoverType::Default, /* fabric_edm_type */
                eth_direction);
            // Both links used by dispatch on TG Gateway (mmio device)
            // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
            configure_edm_builder_for_dispatch(edm_builder);
            edm_builders.insert({eth_chan, edm_builder});
        }

        return;
    }

    std::unordered_map<RoutingDirection, std::vector<chan_id_t>> active_fabric_eth_channels;
    std::unordered_map<RoutingDirection, FabricNodeId> chip_neighbors;
    uint32_t num_intra_chip_neighbors = 0;

    const auto device_has_dispatch_tunnel = [&]() -> bool {
        auto mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
        // results are inclusive of the mmio_device_id so they will never be zero
        TT_ASSERT(tunnels_from_mmio.size() > 0);
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
            std::set<chip_id_t>(neighbors.begin()->second.begin(), neighbors.begin()->second.end()).size() == 1,
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
            "Building fabric router -> device (phys): {}, (logical): {}, direction: {}, active_eth_chans: {}",
            device->id(),
            control_plane.get_fabric_node_id_from_physical_chip_id(device->id()).chip_id,
            direction,
            active_eth_chans.size());
    }

    if (active_fabric_eth_channels.empty()) {
        // Need at least 1 active fabric eth channel in at least 1 direction with a neighbor
        return;
    }

    const bool wrap_around_mesh = fabric_context.is_wrap_around_mesh(fabric_node_id.mesh_id);

    // check whether using tensix extension for connection between worker and fabric routers.
    bool fabric_tensix_extension_enabled = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
                                           tt::tt_fabric::FabricTensixConfig::DISABLED;

    for (const auto& [direction, remote_fabric_node_id] : chip_neighbors) {
        const auto& [fabric_edm_type, fabric_edm_axis] = get_fabric_edm_type(
            control_plane,
            direction,
            fabric_node_id.mesh_id,
            remote_fabric_node_id.mesh_id,
            fabric_node_id.chip_id,
            remote_fabric_node_id.chip_id,
            wrap_around_mesh);

        // Create fabric tensix builder for this ethernet channel
        // Skip the link used by dispatch using relay mux API
        uint32_t dispatch_link_idx =
            tt::tt_metal::RelayMux::get_dispatch_link_index(fabric_node_id, remote_fabric_node_id, device);

        auto get_fabric_router_config =
            [&](bool fabric_tensix_extension_enabled, bool is_dispatch_link, auto eth_direction) {
                auto fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::DISABLED;
                // if not the link used by dispatch, get the fabric router config with tensix extension.
                if (fabric_tensix_extension_enabled && !is_dispatch_link) {
                    fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::MUX;
                }
                return fabric_context.get_fabric_router_config(
                    fabric_edm_type, fabric_edm_axis, fabric_tensix_config, eth_direction);
            };

        for (const auto& eth_chan : active_fabric_eth_channels[direction]) {
            auto eth_direction = control_plane.routing_direction_to_eth_direction(direction);
            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);

            bool dispatch_link = is_dispatch_link(eth_chan, dispatch_link_idx);
            const auto& curr_edm_config =
                get_fabric_router_config(fabric_tensix_extension_enabled, dispatch_link, eth_direction);

            bool has_tensix_extension = false;
            if (fabric_tensix_extension_enabled && !dispatch_link) {
                has_tensix_extension = true;
            }

            auto edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
                device,
                *fabric_program_ptr,
                eth_logical_core,
                fabric_node_id,
                remote_fabric_node_id,
                curr_edm_config,
                false, /* build_in_worker_connection_mode */
                fabric_edm_type,
                eth_direction,
                has_tensix_extension);
            edm_builders.insert({eth_chan, edm_builder});

            if (fabric_tensix_extension_enabled) {
                // Only create tensix builder if this channel is not used by dispatch
                if (!dispatch_link) {
                    auto tensix_builder = tt::tt_fabric::FabricTensixDatamoverBuilder::build(
                        device, *fabric_program_ptr, fabric_node_id, remote_fabric_node_id, eth_chan, eth_direction);
                    tensix_builders.insert({eth_chan, tensix_builder});
                }
            }
        }

        // Last link may be used by dispatch if there is tunneling
        // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
        if (!active_fabric_eth_channels[direction].empty() && device_has_dispatch_tunnel) {
            const auto dispatch_eth_chan = active_fabric_eth_channels[direction].back();
            configure_edm_builder_for_dispatch(edm_builders.at(dispatch_eth_chan));
        }
    }

    const auto topology = fabric_context.get_fabric_topology();
    const bool is_galaxy =
        tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::GALAXY;

    auto build_downstream_connections = [&](tt::tt_fabric::chan_id_t eth_chan_dir1,
                                            tt::tt_fabric::chan_id_t eth_chan_dir2) {
        auto& edm_builder1 = edm_builders.at(eth_chan_dir1);
        auto& edm_builder2 = edm_builders.at(eth_chan_dir2);

        if (fabric_tensix_extension_enabled) {
            if (tensix_builders.find(eth_chan_dir1) != tensix_builders.end() &&
                tensix_builders.find(eth_chan_dir2) != tensix_builders.end()) {
                auto& tensix_builder1 = tensix_builders.at(eth_chan_dir1);
                auto& tensix_builder2 = tensix_builders.at(eth_chan_dir2);

                // need to also pass in edm_builder because it is used to build vc1 connection
                edm_builder1.connect_to_downstream_edm(tensix_builder2, edm_builder2);
                edm_builder2.connect_to_downstream_edm(tensix_builder1, edm_builder1);
            } else {
                // build the downstream connection for the eth channels without tensix extension (dispatch routing
                // plane)
                edm_builder1.connect_to_downstream_edm(edm_builder2);
                edm_builder2.connect_to_downstream_edm(edm_builder1);
            }
        } else {
            edm_builder1.connect_to_downstream_edm(edm_builder2);
            edm_builder2.connect_to_downstream_edm(edm_builder1);
        }
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

                auto& edm_builder1 = edm_builders.at(eth_chan_dir1);
                auto& edm_builder2 = edm_builders.at(eth_chan_dir2);

                build_downstream_connections(eth_chan_dir1, eth_chan_dir2);

                // select VC based on the current link
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

    return;
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program(tt::tt_metal::IDevice* device) {
    std::unique_ptr<tt::tt_metal::Program> fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();
    std::unordered_map<tt::tt_fabric::chan_id_t, tt::tt_fabric::FabricEriscDatamoverBuilder> edm_builders;
    std::unordered_map<tt::tt_fabric::chan_id_t, tt::tt_fabric::FabricTensixDatamoverBuilder> tensix_builders;

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();

    build_tt_fabric_program(device, fabric_program_ptr.get(), edm_builders, tensix_builders);
    fabric_context.set_num_fabric_initialized_routers(device->id(), edm_builders.size());
    if (edm_builders.empty()) {
        return nullptr;
    }

    // Compile all fabric tensix builders
    if (tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
        tt::tt_fabric::FabricTensixConfig::DISABLED) {
        for (auto& [eth_chan, tensix_builder] : tensix_builders) {
            tensix_builder.create_and_compile(device, *fabric_program_ptr);
        }
    }

    // for now it doesnt matter which channel is the master, so just pick the 1st in the map
    auto master_router_chan = edm_builders.begin()->first;
    fabric_context.set_fabric_master_router_chan(device->id(), master_router_chan);

    uint32_t router_channels_mask = 0;
    for (const auto& [router_chan, _] : edm_builders) {
        router_channels_mask += 0x1 << (uint32_t)router_chan;
    }

    std::map<std::string, std::string> defines = {};
    if (fabric_context.is_2D_routing_enabled()) {
        defines["FABRIC_2D"] = "";
    }

    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto num_enabled_eth_cores = edm_builders.size();
    const auto num_enabled_risc_cores =
        edm_builders.begin()->second.get_configured_risc_count();  // same across all eth cores
    size_t num_local_fabric_routers = num_enabled_eth_cores;
    for (auto& [eth_chan, edm_builder] : edm_builders) {
        edm_builder.set_wait_for_host_signal(true);
        const std::vector<uint32_t> rt_args = edm_builder.get_runtime_args();
        for (uint32_t risc_id = 0; risc_id < num_enabled_risc_cores; risc_id++) {
            std::vector<uint32_t> ct_args = edm_builder.get_compile_time_args(risc_id);

            const auto is_master_risc_core = eth_chan == master_router_chan && (risc_id == 0);
            ct_args.push_back(is_master_risc_core);
            ct_args.push_back(master_router_chan);
            ct_args.push_back(num_local_fabric_routers);
            ct_args.push_back(router_channels_mask);

            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
            auto kernel = tt::tt_metal::CreateKernel(
                *fabric_program_ptr,
                "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
                eth_logical_core,
                tt::tt_metal::EthernetConfig{
                    .noc = edm_builder.config.risc_configs[risc_id].get_configured_noc(),
                    .processor = static_cast<tt::tt_metal::DataMovementProcessor>(risc_id),
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
    std::vector<uint32_t> router_zero_buf(1, 0);
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto addresses_to_clear = control_plane.get_fabric_context().get_fabric_router_addresses_to_clear();
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }
}

}  // namespace tt::tt_fabric
