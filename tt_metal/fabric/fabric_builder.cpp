// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/compute_mesh_router_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "dispatch/kernel_config/relay_mux.hpp"
#include <set>

namespace tt::tt_fabric {

FabricBuilder::FabricBuilder(
    tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, FabricContext& fabric_context) :
    device_(device),
    program_(program),
    fabric_context_(fabric_context),
    builder_context_(fabric_context.get_builder_context()),
    local_node_(tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_node_id_from_physical_chip_id(
        device->id())),
    wrap_around_mesh_(fabric_context_.is_wrap_around_mesh(local_node_.mesh_id)) {
    // Determine if this device has tunneling dispatch
    auto mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_->id());
    auto tunnels_from_mmio =
        tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
    TT_ASSERT(!tunnels_from_mmio.empty());
    device_has_dispatch_tunnel_ = (tunnels_from_mmio.size() - 1) > 0;
}

void FabricBuilder::discover_channels() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const bool is_2D_routing = fabric_context_.is_2D_routing_enabled();

    auto is_dispatch_link = [&](chan_id_t eth_chan, uint32_t dispatch_link_idx) {
        auto link_idx = control_plane.get_routing_plane_id(local_node_, eth_chan);
        return device_has_dispatch_tunnel_ && link_idx == dispatch_link_idx;
    };

    // Discover active channels and neighbors
    for (const auto& direction : FabricContext::routing_directions) {
        auto active_eth_chans = control_plane.get_active_fabric_eth_routing_planes_in_direction(local_node_, direction);
        if (active_eth_chans.empty()) {
            continue;
        }

        auto neighbors = control_plane.get_chip_neighbors(local_node_, direction);
        auto intra_chip_neighbors = neighbors.find(local_node_.mesh_id);

        TT_FATAL(neighbors.size() == 1, "Multiple neighbor meshes per direction is unsupported");
        TT_FATAL(
            std::set<ChipId>(neighbors.begin()->second.begin(), neighbors.begin()->second.end()).size() == 1,
            "Multiple neighbors per direction is currently unsupported");

        // 1D fabric only supports intramesh connections
        if (!is_2D_routing) {
            bool has_inter_mesh_connections = (intra_chip_neighbors == neighbors.end());
            TT_FATAL(!has_inter_mesh_connections, "1D routing does not support intermesh connections");
        }

        // Cache neighbor and channel info
        FabricNodeId neighbor_fabric_node_id = FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
        chip_neighbors_.emplace(direction, neighbor_fabric_node_id);
        channels_by_direction_[direction] = active_eth_chans;

        // Identify and cache dispatch links
        uint32_t dispatch_link_idx =
            tt::tt_metal::RelayMux::get_dispatch_link_index(local_node_, neighbor_fabric_node_id, device_);
        for (const auto& eth_chan : active_eth_chans) {
            if (is_dispatch_link(eth_chan, dispatch_link_idx)) {
                dispatch_links_.insert(eth_chan);
            }
        }
    }
}

void FabricBuilder::create_routers() {
    // Create router builders
    for (const auto& [direction, eth_channels] : channels_by_direction_) {
        const auto& neighbor_node = chip_neighbors_.at(direction);

        for (const auto& eth_chan : eth_channels) {
            bool is_dispatch = dispatch_links_.contains(eth_chan);

            RouterLocation location{
                .eth_chan = eth_chan,
                .remote_node = neighbor_node,
                .direction = direction,
                .is_dispatch_link = is_dispatch,
            };

            auto router_builder = FabricRouterBuilder::create(device_, program_, local_node_, location);
            routers_.insert({eth_chan, std::move(router_builder)});
        }
    }

    // Configure dispatch links
    if (device_has_dispatch_tunnel_) {
        for (const auto& dispatch_chan : dispatch_links_) {
            routers_.at(dispatch_chan)->configure_for_dispatch();
        }
    }

    // Record build state
    builder_context_.set_num_fabric_initialized_routers(device_->id(), routers_.size());
    if (!routers_.empty()) {
        master_router_chan_ = routers_.begin()->first;
        builder_context_.set_fabric_master_router_chan(device_->id(), master_router_chan_);
    }
}

std::vector<FabricBuilder::RouterConnectionPair> FabricBuilder::get_router_connection_pairs() const {
    std::vector<RouterConnectionPair> pairs;

    const bool is_2D_routing = fabric_context_.is_2D_routing_enabled();
    const size_t num_intra_chip_neighbors = chip_neighbors_.size();

    // Check if we can connect two directions
    auto can_connect = [&](RoutingDirection dir1, RoutingDirection dir2) {
        return chip_neighbors_.contains(dir1) && chip_neighbors_.contains(dir2) &&
               channels_by_direction_.contains(dir1) && channels_by_direction_.contains(dir2);
    };

    // Add connection pairs for two directions
    auto add_direction_pairs = [&](RoutingDirection dir1, RoutingDirection dir2) {
        if (!can_connect(dir1, dir2)) {
            return;
        }

        const auto& chans_dir1 = channels_by_direction_.at(dir1);
        const auto& chans_dir2 = channels_by_direction_.at(dir2);
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

        // Z router connections - connect Z routers to all 4 mesh directions
        add_direction_pairs(RoutingDirection::Z, RoutingDirection::N);
        add_direction_pairs(RoutingDirection::Z, RoutingDirection::S);
        add_direction_pairs(RoutingDirection::Z, RoutingDirection::E);
        add_direction_pairs(RoutingDirection::Z, RoutingDirection::W);
    } else if (wrap_around_mesh_ && num_intra_chip_neighbors == 2) {
        // 1D Routing wrap the corner chips, fold the internal connections
        auto it = chip_neighbors_.begin();
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

void FabricBuilder::connect_routers() {
    const auto topology = fabric_context_.get_fabric_topology();
    const bool is_galaxy = tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy();

    // If NeighborExchange topology is used, message forwarding is not supported, and thus there is no need to connect
    // routers on the same device together
    if (topology == Topology::NeighborExchange) {
        return;
    }

    // Get connection pairs based on topology
    auto connection_pairs = get_router_connection_pairs();

    std::map<FabricRouterBuilder*, std::map<RoutingDirection, FabricRouterBuilder*>> routers_by_direction_map{};
    // Connect each pair (inter-device INTRA_MESH connections)
    for (const auto& pair : connection_pairs) {
        auto& router1 = routers_.at(pair.chan1);
        auto& router2 = routers_.at(pair.chan2);

        router1->configure_connection(*router2, pair.link_idx, pair.num_links, topology, is_galaxy);

        routers_by_direction_map[router1.get()].insert({router2->get_location().direction, router2.get()});
        routers_by_direction_map[router2.get()].insert({router1->get_location().direction, router1.get()});
    }

    // Configure local connections between routers on this device
    configure_local_connections(routers_by_direction_map);
}

void FabricBuilder::configure_local_connections(
    const std::map<FabricRouterBuilder*, std::map<RoutingDirection, FabricRouterBuilder*>>& routers_by_direction_map) {
    // Generic local connection establishment: iterate through all routers and
    // establish connections to local targets based on their connection mappings

    // For each router, establish its local connections
    for (const auto& [source_router, target_routers_by_direction] : routers_by_direction_map) {
        // Build map of potential local targets (all other routers on this device)
        std::map<RoutingDirection, FabricRouterBuilder*> local_targets;
        for (const auto& [target_dir, target_router] : target_routers_by_direction) {
            local_targets[target_dir] = target_router;
        }

        source_router->configure_local_connections(local_targets);
    }
}

void FabricBuilder::compile_ancillary_kernels() {
    // Router-associated ancillary kernels
    for (auto& [eth_chan, router_builder] : routers_) {
        router_builder->compile_ancillary_kernels(program_);
    }

    // Device-level kernels for missing directions (e.g., UDM mode)
    compile_kernels_for_missing_directions();
}

void FabricBuilder::compile_kernels_for_missing_directions() {
    // Only applicable in UDM mode
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    if (fabric_tensix_config != FabricTensixConfig::UDM) {
        return;
    }

    const auto& tensix_config = builder_context_.get_tensix_config();
    if (!tensix_config.has_missing_directions(device_->id())) {
        return;
    }

    const auto& missing_directions = tensix_config.get_missing_directions(device_->id());

    for (const auto& [routing_plane_id, missing_dir] : missing_directions) {
        log_warning(
            tt::LogMetal,
            "Building missing direction tensix builder for fabric_node {}, routing_plane {}, direction {}",
            local_node_,
            routing_plane_id,
            static_cast<uint32_t>(missing_dir));

        // Build and compile tensix builder for this missing (routing_plane_id, direction) pair
        auto tensix_builder = FabricTensixDatamoverBuilder::build_for_missing_direction(
            device_, program_, local_node_, routing_plane_id, missing_dir);
        tensix_builder.create_and_compile(program_);
    }
}

void FabricBuilder::create_kernels() {
    uint32_t router_channels_mask = 0;
    for (const auto& [router_chan, _] : routers_) {
        router_channels_mask |= (1 << static_cast<uint32_t>(router_chan));
    }

    KernelCreationContext ctx{
        .is_2D_routing = fabric_context_.is_2D_routing_enabled(),
        .master_router_chan = master_router_chan_,
        .num_local_fabric_routers = routers_.size(),
        .router_channels_mask = router_channels_mask,
    };

    for (auto& [eth_chan, router_builder] : routers_) {
        router_builder->create_kernel(program_, ctx);
    }
}

}  // namespace tt::tt_fabric
