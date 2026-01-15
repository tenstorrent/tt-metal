// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_mesh_router_builder.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_core_placement.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/third_party/umd/device/api/umd/device/types/core_coordinates.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "tt_metal.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

ComputeMeshRouterBuilder::ComputeMeshRouterBuilder(
    FabricNodeId local_node,
    const RouterLocation& location,
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder,
    std::optional<FabricTensixDatamoverBuilder> tensix_builder,
    FabricRouterChannelMapping channel_mapping,
    RouterConnectionMapping connection_mapping,
    std::shared_ptr<ConnectionRegistry> connection_registry) :
    FabricRouterBuilder(local_node, location),
    erisc_builder_(std::move(erisc_builder)),
    tensix_builder_(std::move(tensix_builder)),
    channel_mapping_(std::move(channel_mapping)),
    connection_mapping_(std::move(connection_mapping)),
    connection_registry_(std::move(connection_registry)),
    is_inter_mesh_(local_node.mesh_id != location.remote_node.mesh_id) {
    TT_FATAL(erisc_builder_ != nullptr, "Erisc builder cannot be null");
}

std::unique_ptr<ComputeMeshRouterBuilder> ComputeMeshRouterBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    FabricNodeId local_node,
    const RouterLocation& location,
    std::shared_ptr<ConnectionRegistry> connection_registry) {
    // Get fabric context and config
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto topology = fabric_context.get_fabric_topology();

    // Convert RoutingDirection to eth_chan_directions
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto eth_direction = control_plane.routing_direction_to_eth_direction(location.direction);

    // Get SOC descriptor for eth core lookup
    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    auto eth_logical_core = soc_desc.get_eth_core_for_channel(location.eth_chan, CoordSystem::LOGICAL);

    // Determine tensix config
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    bool fabric_tensix_extension_enabled = fabric_tensix_config != FabricTensixConfig::DISABLED;
    bool fabric_tensix_extension_mux_mode = fabric_tensix_config == FabricTensixConfig::MUX;
    bool fabric_tensix_extension_udm_mode = fabric_tensix_config == FabricTensixConfig::UDM;

    // Determine if tensix builder will be created (reusable condition)
    bool will_create_tensix_builder = fabric_tensix_extension_enabled && !location.is_dispatch_link;
    bool downstream_is_tensix_builder = will_create_tensix_builder && fabric_tensix_extension_mux_mode;

    // Get the appropriate EDM config from builder context
    auto tensix_config_for_lookup = will_create_tensix_builder ? fabric_tensix_config : FabricTensixConfig::DISABLED;
    const auto& edm_config = builder_context.get_fabric_router_config(tensix_config_for_lookup, eth_direction);

    // Create channel mapping EARLY (needed for computing injection flags)
    RouterVariant variant = (location.direction == RoutingDirection::Z) ? RouterVariant::Z_ROUTER : RouterVariant::MESH;
    const auto& intermesh_config = fabric_context.get_builder_context().get_intermesh_vc_config();
    auto channel_mapping =
        FabricRouterChannelMapping(topology, downstream_is_tensix_builder, variant, &intermesh_config);

    // Create connection mapping (Phase 3)
    RouterConnectionMapping connection_mapping;
    if (variant == RouterVariant::Z_ROUTER) {
        connection_mapping = RouterConnectionMapping::for_z_router();
    } else {
        // Check if this device has a Z router
        bool has_z = fabric_context.has_z_router_on_device(device->id());
        // Enable VC1 for all routers when intermesh VC is configured
        bool enable_vc1 = intermesh_config.requires_vc1;
        connection_mapping = RouterConnectionMapping::for_mesh_router(topology, location.direction, has_z, enable_vc1);
    }

    // Compute injection channel flags at router level BEFORE creating builders
    // Injection semantics are per-VC, so compute for each VC and flatten into router-level vector
    // Injection channel status flags are used by sender channels to understand if that channel must
    // implement bubble flow-control behaviour.

    // First, compute the total number of channels across all VCs
    size_t total_router_channels = 0;
    uint32_t num_vcs = channel_mapping.get_num_virtual_channels();
    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        total_router_channels += channel_mapping.get_num_sender_channels_for_vc(vc);
    }

    std::vector<bool> router_injection_flags;
    router_injection_flags.reserve(total_router_channels);

    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        uint32_t num_channels_in_vc = channel_mapping.get_num_sender_channels_for_vc(vc);
        auto vc_injection_flags =
            compute_sender_channel_injection_flags_for_vc(topology, eth_direction, vc, num_channels_in_vc);

        // Flatten into router-level vector
        for (uint32_t ch_idx = 0; ch_idx < num_channels_in_vc; ++ch_idx) {
            router_injection_flags.push_back(vc_injection_flags.at(ch_idx));
        }
    }

    // Build reverse channel maps and compute injection flags for each builder variant
    // Get ERISC's channel count from config
    size_t erisc_num_channels = edm_config.num_used_sender_channels;
    auto erisc_to_router_channel_map =
        get_variant_to_router_channel_map(channel_mapping, BuilderType::ERISC, erisc_num_channels);
    auto erisc_injection_flags =
        get_child_builder_variant_sender_channel_injection_flags(router_injection_flags, erisc_to_router_channel_map);

    std::vector<bool> tensix_injection_flags;
    if (downstream_is_tensix_builder) {
        size_t tensix_num_channels = builder_config::get_num_tensix_sender_channels(topology, fabric_tensix_config);
        auto tensix_to_router_channel_map =
            get_variant_to_router_channel_map(channel_mapping, BuilderType::TENSIX, tensix_num_channels);
        tensix_injection_flags = get_child_builder_variant_sender_channel_injection_flags(
            router_injection_flags, tensix_to_router_channel_map);
    }

    // Compute actual per-VC channel counts for this router
    std::array<std::size_t, builder_config::MAX_NUM_VCS> actual_sender_channels_per_vc{};
    std::array<std::size_t, builder_config::MAX_NUM_VCS> actual_receiver_channels_per_vc{};
    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        actual_sender_channels_per_vc[vc] = channel_mapping.get_num_sender_channels_for_vc(vc);
        actual_receiver_channels_per_vc[vc] = 1;  // Always 1 receiver per VC (when VC exists)
    }

    // NOW create erisc builder with computed injection flags and actual channel counts
    auto edm_builder = std::make_unique<FabricEriscDatamoverBuilder>(FabricEriscDatamoverBuilder::build(
        device,
        program,
        eth_logical_core,
        local_node,
        location.remote_node,
        edm_config,
        std::move(erisc_injection_flags),
        false, /* build_in_worker_connection_mode */
        eth_direction,
        downstream_is_tensix_builder,
        actual_sender_channels_per_vc,
        actual_receiver_channels_per_vc));

    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode()) {
        // Enable updates at a fixed interval for link stability and link status updates
        constexpr uint32_t k_BlackholeFabricRouterContextSwitchInterval = 32;
        edm_builder->set_firmware_context_switch_interval(k_BlackholeFabricRouterContextSwitchInterval);
        edm_builder->set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::WAIT_FOR_IDLE);
    }

    // Create tensix builder if needed
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_opt;
    if (will_create_tensix_builder) {
        tensix_builder_opt = FabricTensixDatamoverBuilder::build(
            device,
            program,
            local_node,
            location.remote_node,
            location.eth_chan,
            eth_direction,
            std::move(tensix_injection_flags));
    }

    // Use unique_ptr constructor directly since ComputeMeshRouterBuilder constructor is private
    auto router_builder = std::unique_ptr<ComputeMeshRouterBuilder>(new ComputeMeshRouterBuilder(
        local_node, location, std::move(edm_builder), std::move(tensix_builder_opt), std::move(channel_mapping), std::move(connection_mapping), std::move(connection_registry)));

    // Setup the local relay kernel connection if in UDM mode
    if (fabric_tensix_extension_udm_mode && router_builder->has_tensix_builder()) {
        router_builder->connect_to_local_tensix_builder(router_builder->get_tensix_builder());
    }

    return router_builder;
}

uint32_t ComputeMeshRouterBuilder::get_downstream_sender_channel(
    const bool is_2D_routing, const eth_chan_directions downstream_direction, uint32_t vc) const {
    if (!is_2D_routing) {
        return 1;  // 1D: sender channel 1 for forwarding
    }

    // Sender channel 0 is always reserved for the local worker (VC0 only).
    //
    // VC0: Sender channels 1–3 correspond to the three upstream neighbors
    // VC1: Sender channels 0–2 correspond to the three upstream neighbors
    //
    // The mapping from receiver direction → sender channel depends on the
    // downstream forwarding direction:
    //
    //   • Downstream = EAST:
    //         WEST  → channel 1 (VC0) or 0 (VC1)
    //         NORTH → channel 2 (VC0) or 1 (VC1)
    //         SOUTH → channel 3 (VC0) or 2 (VC1)
    //
    //   • Downstream = WEST:
    //         EAST  → channel 1 (VC0) or 0 (VC1)
    //         NORTH → channel 2 (VC0) or 1 (VC1)
    //         SOUTH → channel 3 (VC0) or 2 (VC1)
    //
    //   • Downstream = NORTH:
    //         EAST  → channel 1 (VC0) or 0 (VC1)
    //         WEST  → channel 2 (VC0) or 1 (VC1)
    //         SOUTH → channel 3 (VC0) or 2 (VC1)
    //
    //   • Downstream = SOUTH:
    //         EAST  → channel 1 (VC0) or 0 (VC1)
    //         WEST  → channel 2 (VC0) or 1 (VC1)
    //         NORTH → channel 3 (VC0) or 2 (VC1)

    size_t downstream_compact_index_for_upstream;
    if (downstream_direction == eth_chan_directions::EAST) {
        // EAST downstream: WEST(1)→0, NORTH(2)→1, SOUTH(3)→2
        downstream_compact_index_for_upstream = this->get_eth_direction() - 1;
    } else {
        // For other downstream directions: if upstream < downstream, use as-is; else subtract 1
        downstream_compact_index_for_upstream = (this->get_eth_direction() < downstream_direction)
                                                    ? this->get_eth_direction()
                                                    : (this->get_eth_direction() - 1);
    }

    // Compute sender channel based on VC
    // VC0: 1 + compact_index (channels 1-3, skip 0 for local worker)
    // VC1: compact_index (channels 0-2, no local worker channel)
    uint32_t sender_channel =
        (vc == 0) ? (1 + downstream_compact_index_for_upstream) : downstream_compact_index_for_upstream;

    return sender_channel;
}

eth_chan_directions ComputeMeshRouterBuilder::get_eth_direction() const { return erisc_builder_->get_direction(); }

size_t ComputeMeshRouterBuilder::get_noc_x() const { return erisc_builder_->get_noc_x(); }

size_t ComputeMeshRouterBuilder::get_noc_y() const { return erisc_builder_->get_noc_y(); }

size_t ComputeMeshRouterBuilder::get_configured_risc_count() const {
    return erisc_builder_->get_configured_risc_count();
}

std::vector<bool> ComputeMeshRouterBuilder::compute_sender_channel_injection_flags_for_vc(
    Topology topology, eth_chan_directions direction, uint32_t vc, uint32_t num_channels) {
    std::vector<bool> injection_flags(num_channels, false);

    // VC1 is for inter-mesh routing and doesn't need bubble flow control
    // All VC1 channels are marked as non-injection (false) regardless of topology
    if (vc == 1) {
        return injection_flags;
    }

    // Early return for Linear/Mesh - no injection channels marked
    if (topology == Topology::Linear || topology == Topology::Mesh) {
        return injection_flags;
    }

    // VC0: Worker channel (idx 0) is always an injection channel
    injection_flags.at(0) = true;

    if (topology != Topology::Torus) {
        return injection_flags;
    }

    // For Torus: Turn channels are injection channels (VC0 only)
    // A turn channel is where my_direction differs from sender_channel_direction

    bool I_am_ew = builder::is_east_or_west(direction);
    bool I_am_ns = builder::is_north_or_south(direction);

    TT_FATAL(
        I_am_ew ^ I_am_ns,
        "Internal error: In compute_sender_channel_injection_flags_for_vc, I_am_ew and I_am_ns cannot both be true");

    for (size_t ch_idx = 1; ch_idx < num_channels; ++ch_idx) {
        // Map to VC0 equivalent channel for direction lookup
        auto sender_channel_direction = builder::get_sender_channel_direction(direction, ch_idx);

        bool sender_channel_is_ew = builder::is_east_or_west(sender_channel_direction);
        bool sender_channel_is_ns = builder::is_north_or_south(sender_channel_direction);
        bool sender_channel_is_turn = (I_am_ew && !sender_channel_is_ew) || (I_am_ns && !sender_channel_is_ns);

        injection_flags.at(ch_idx) = sender_channel_is_turn;
    }

    return injection_flags;
}

std::vector<bool> ComputeMeshRouterBuilder::get_child_builder_variant_sender_channel_injection_flags(
    const std::vector<bool>& router_injection_flags,
    const std::vector<std::optional<size_t>>& variant_to_router_channel_map) {
    std::vector<bool> variant_injection_flags;
    variant_injection_flags.reserve(variant_to_router_channel_map.size());

    // Iterate through variant's internal channels in order (0, 1, 2, ...)
    // For each variant channel, look up its corresponding router channel and get the injection flag
    for (auto router_channel_opt : variant_to_router_channel_map) {
        if (router_channel_opt.has_value()) {
            // Channel is externally-facing, get injection status from router
            size_t router_channel_id = *router_channel_opt;
            TT_FATAL(
                router_channel_id < router_injection_flags.size(),
                "Internal error: Router channel ID {} out of bounds (max {})",
                router_channel_id,
                router_injection_flags.size());
            variant_injection_flags.push_back(router_injection_flags.at(router_channel_id));
        } else {
            // Channel is internal-only (e.g., ERISC in MUX mode fed by TENSIX)
            // Internal channels are never injection channels
            variant_injection_flags.push_back(false);
        }
    }

    return variant_injection_flags;
}

std::vector<std::optional<size_t>> ComputeMeshRouterBuilder::get_variant_to_router_channel_map(
    const FabricRouterChannelMapping& channel_mapping, BuilderType builder_type, size_t variant_num_sender_channels) {
    // Get all mappings from channel_mapping (implicitly handles VC structure)
    auto all_mappings = channel_mapping.get_all_sender_mappings();

    // Default to nullopt (not externally mapped)
    std::vector<std::optional<size_t>> variant_to_router_channel_map(variant_num_sender_channels);

    // Populate the mapping
    for (size_t router_ch_id = 0; router_ch_id < all_mappings.size(); ++router_ch_id) {
        const auto& mapping = all_mappings.at(router_ch_id);
        if (mapping.builder_type == builder_type) {
            // Only set for externally-facing channels
            variant_to_router_channel_map.at(mapping.internal_sender_channel_id) = router_ch_id;
        }
    }

    return variant_to_router_channel_map;
}

void ComputeMeshRouterBuilder::connect_to_local_tensix_builder(FabricTensixDatamoverBuilder& tensix_builder) {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();
    TT_FATAL(is_2D_routing, "connect_to_local_tensix_builder requires 2D routing");

    // In UDM mode, router receiver connects to local relay on the tensix core
    // Get connection specs from relay builder to set up receiver-to-relay connection
    // Relay only has one channel (ROUTER_CHANNEL = 0) for upstream fabric router traffic
    eth_chan_directions local_tensix_dir = tensix_builder.get_direction();
    auto adapter_spec = tensix_builder.build_connection_to_relay_channel();

    // Enable UDM mode and store relay buffer count
    erisc_builder_->udm_mode = true;
    erisc_builder_->local_tensix_relay_num_buffers = adapter_spec.num_buffers_per_channel;

    // Only one local relay connection (we can consider the local relay connection as one ds tensix connection)
    erisc_builder_->num_downstream_tensix_connections = 1;

    auto* adapter_ptr = erisc_builder_->receiver_channel_to_downstream_adapter.get();
    const auto tensix_noc_x = tensix_builder.get_noc_x();
    const auto tensix_noc_y = tensix_builder.get_noc_y();
    adapter_ptr->add_local_tensix_connection(adapter_spec, local_tensix_dir, CoreCoord(tensix_noc_x, tensix_noc_y));

    // Provide router NOC coordinates to relay kernel for sending packets back to router
    tensix_builder.append_relay_router_noc_xy(erisc_builder_->get_noc_x(), erisc_builder_->get_noc_y());
}

void ComputeMeshRouterBuilder::establish_connections_to_router(
    ComputeMeshRouterBuilder& downstream_router,
    const std::function<bool(ConnectionType)>& /*connection_type_filter*/) {
    // Establish VC connections between this router and the specified downstream router
    // This function does NOT iterate through targets - it connects to the single downstream_router passed in
    uint32_t num_vcs = channel_mapping_.get_num_virtual_channels();

    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();

    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        // Compute sender channel on downstream router based on directions and VC
        uint32_t downstream_sender_channel =
            get_downstream_sender_channel(is_2D_routing, downstream_router.get_eth_direction(), vc);

        // Get downstream builder and mapping
        auto* downstream_builder = downstream_router.get_builder_for_vc_channel(vc, downstream_sender_channel);
        auto downstream_mapping = downstream_router.channel_mapping_.get_sender_mapping(vc, downstream_sender_channel);
        uint32_t internal_channel_id = downstream_mapping.internal_sender_channel_id;

        // Setup producer → consumer connection
        // setup_downstream_vc_connection handles type checking internally
        erisc_builder_->setup_downstream_vc_connection(downstream_builder, vc, vc, internal_channel_id);
        // Record connection in registry if present
        if (connection_registry_) {
            RouterConnectionRecord record{
                .source_node = local_node_,
                .source_direction = location_.direction,
                .source_eth_chan = location_.eth_chan,
                .source_vc = vc,
                .source_receiver_channel = 0,
                .dest_node = downstream_router.local_node_,
                .dest_direction = downstream_router.location_.direction,
                .dest_eth_chan = downstream_router.location_.eth_chan,
                .dest_vc = vc,
                .dest_sender_channel = internal_channel_id,
                .connection_type = connection_mapping_.get_downstream_targets(vc, 0).at(0).type};
            connection_registry_->record_connection(record);
        }

        log_debug(
            tt::LogTest,
            "Router at x={}, y={}, Direction={}, FabricNodeId={} :: Connecting VC{} receiver_ch={} to downstream "
            "router at x={}, y={}, Direction={}, VC{}, internal_ch={}",
            get_noc_x(),
            get_noc_y(),
            get_eth_direction(),
            local_node_,
            vc,
            0,
            downstream_builder->get_noc_x(),
            downstream_builder->get_noc_y(),
            downstream_builder->get_direction(),
            vc,
            internal_channel_id);
    }
}

void ComputeMeshRouterBuilder::configure_connection(
    FabricRouterBuilder& peer, uint32_t link_idx, uint32_t num_links, Topology topology, bool is_galaxy) {
    // Validate invariant: FabricBuilder guarantees all routers on a device are the same concrete type
    auto* peer_compute_ptr = dynamic_cast<ComputeMeshRouterBuilder*>(&peer);
    TT_FATAL(
        peer_compute_ptr != nullptr,
        "Router type mismatch: expected ComputeMeshRouterBuilder but got different type. "
        "This indicates a bug in FabricBuilder::create_routers()");
    auto& peer_compute = *peer_compute_ptr;

    TT_FATAL(
        !erisc_builder_->build_in_worker_connection_mode,
        "Tried to connect router to downstream in worker connection mode");

    // Establish INTRA_MESH connections between the two routers (bidirectional)
    auto intra_mesh_filter = [](ConnectionType type) { return type == ConnectionType::INTRA_MESH; };

    establish_connections_to_router(peer_compute, intra_mesh_filter);
    peer_compute.establish_connections_to_router(*this, intra_mesh_filter);

    // Configure NOC VC based on link index (must be same for both routers)
    auto edm_noc_vc = tt::tt_fabric::FabricEriscDatamoverConfig::DEFAULT_NOC_VC +
                      (link_idx % tt::tt_fabric::FabricEriscDatamoverConfig::NUM_EDM_NOC_VCS);
    erisc_builder_->config.edm_noc_vc = edm_noc_vc;
    peer_compute.erisc_builder_->config.edm_noc_vc = edm_noc_vc;

    // Apply core placement optimizations
    core_placement::CorePlacementContext cctx{
        .topology = topology,
        .is_galaxy = is_galaxy,
        .num_links = num_links,
    };
    core_placement::apply_core_placement_optimizations(cctx, *erisc_builder_, *peer_compute.erisc_builder_, link_idx);
}

void ComputeMeshRouterBuilder::configure_local_connections(
    const std::map<RoutingDirection, FabricRouterBuilder*>& local_routers) {
    // Establish local connections (MESH_TO_Z or Z_TO_MESH) to routers on same device
    // We need to look up target routers by direction from the map
    auto local_connection_filter = [](ConnectionType type) {
        return type == ConnectionType::MESH_TO_Z || type == ConnectionType::Z_TO_MESH;
    };

    // Track which target routers we've already connected to (to avoid redundant calls)
    std::set<RoutingDirection> connected_targets;

    // Iterate through all VCs to find local connection targets
    uint32_t num_vcs = channel_mapping_.get_num_virtual_channels();

    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        // Each VC has only 1 receiver channel (index 0)
        constexpr uint32_t receiver_channel = 0;
        auto targets = connection_mapping_.get_downstream_targets(vc, receiver_channel);

        for (const auto& target : targets) {
            // Only handle local connections
            if (!local_connection_filter(target.type)) {
                continue;
            }

            // Check if target direction exists (handles 2-4 router configs)
            TT_FATAL(target.target_direction.has_value(), "Local connection target must have direction specified");

            auto target_dir = target.target_direction.value();
            if (!local_routers.contains(target_dir)) {
                // Target router doesn't exist (e.g., edge device with only 2-3 mesh routers)
                // This is expected for Z routers on edge devices
                continue;
            }

            // Establish connections to this target router (if not already done)
            if (!connected_targets.contains(target_dir)) {
                auto* local_router = local_routers.at(target_dir);
                auto* local_compute_router = dynamic_cast<ComputeMeshRouterBuilder*>(local_router);
                TT_FATAL(local_compute_router != nullptr, "Local router must be a ComputeMeshRouterBuilder");
                establish_connections_to_router(*local_compute_router, local_connection_filter);
                connected_targets.insert(target_dir);
            }
        }
    }
}

void ComputeMeshRouterBuilder::configure_for_dispatch() {
    // Dispatch requires higher context switching frequency to service slow dispatch / UMD / debug tools
    constexpr uint32_t k_DispatchFabricRouterContextSwitchInterval = 16;
    erisc_builder_->set_firmware_context_switch_interval(k_DispatchFabricRouterContextSwitchInterval);
    erisc_builder_->set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::INTERVAL);
}

void ComputeMeshRouterBuilder::compile_ancillary_kernels(tt::tt_metal::Program& program) {
    // Compile tensix builder if present
    if (tensix_builder_.has_value()) {
        tensix_builder_->create_and_compile(program);
    }
}

void ComputeMeshRouterBuilder::create_kernel(tt::tt_metal::Program& program, const KernelCreationContext& ctx) {
    // Build defines
    std::map<std::string, std::string> defines = {};
    if (ctx.is_2D_routing) {
        defines["FABRIC_2D"] = "";

        // FABRIC_2D_VC1_ACTIVE: Set when router has VC1
        bool vc1_active = channel_mapping_.get_num_virtual_channels() > 1;
        if (vc1_active) {
            defines["FABRIC_2D_VC1_ACTIVE"] = "";
        }

        // FABRIC_2D_VC1_SERVICED: Set when router actively services VC1 traffic
        // - Intra-mesh routers service VC1 when full_mesh mode is enabled
        // - Inter-mesh routers service VC1 when pass_through mode is enabled
        const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        const auto& intermesh_config = fabric_context.get_builder_context().get_intermesh_vc_config();

        // VC1 is serviced when:
        // - Intra-mesh router with full mesh VC1, or
        // - Inter-mesh router with pass-through VC1
        bool vc1_serviced = (!is_inter_mesh_ && intermesh_config.requires_vc1_full_mesh) ||
                            (is_inter_mesh_ && intermesh_config.requires_vc1_mesh_pass_through);

        if (vc1_serviced) {
            defines["FABRIC_2D_VC1_SERVICED"] = "";
        }

        // FABRIC_2D_VC0_CROSSOVER_TO_VC1: Set for inter-mesh routers that perform VC0→VC1 crossover
        // Inter-mesh routers crossover incoming VC0 traffic to downstream intra-mesh VC1
        bool vc0_crossover_to_vc1 = is_inter_mesh_ && intermesh_config.requires_vc1_full_mesh;
        if (vc0_crossover_to_vc1) {
            defines["FABRIC_2D_VC0_CROSSOVER_TO_VC1"] = "";
        }
    }

    // Get SOC descriptor for eth core lookup
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto device_id = control_plane.get_physical_chip_id_from_fabric_node_id(local_node_);
    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
    const auto eth_chan = location_.eth_chan;
    auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);

    // Configure for host signal wait
    erisc_builder_->set_wait_for_host_signal(true);

    // Get runtime args (same for all RISC cores)
    const std::vector<uint32_t> rt_args = erisc_builder_->get_runtime_args();

    const auto num_enabled_risc_cores = get_configured_risc_count();

    for (uint32_t risc_id = 0; risc_id < num_enabled_risc_cores; risc_id++) {
        // Get compile-time args and append cluster-wide coordination info
        std::vector<uint32_t> ct_args = erisc_builder_->get_compile_time_args(risc_id);

        const auto is_master_risc_core = (eth_chan == ctx.master_router_chan) && (risc_id == 0);
        ct_args.push_back(is_master_risc_core);
        ct_args.push_back(ctx.master_router_chan);
        ct_args.push_back(ctx.num_local_fabric_routers);
        ct_args.push_back(ctx.router_channels_mask);

        // Determine processor
        auto proc = static_cast<tt::tt_metal::DataMovementProcessor>(risc_id);
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
            tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode() &&
            num_enabled_risc_cores == 1) {
            // Force fabric to run on erisc1 due to stack usage exceeded with MUX on erisc0
            proc = tt::tt_metal::DataMovementProcessor::RISCV_1;
        }

        // Use Os (optimize for size) when VC1 is active to fit in code space
        // Use O3 (optimize for performance) otherwise
        bool vc1_active = erisc_builder_->config.num_used_receiver_channels_per_vc[1] > 0;
        auto opt_level = vc1_active ? tt::tt_metal::KernelBuildOptLevel::Os : tt::tt_metal::KernelBuildOptLevel::O3;

        // Create the kernel
        auto kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
            eth_logical_core,
            tt::tt_metal::EthernetConfig{
                .noc = erisc_builder_->config.risc_configs[risc_id].get_configured_noc(),
                .processor = proc,
                .compile_args = ct_args,
                .defines = defines,
                .opt_level = opt_level});

        tt::tt_metal::SetRuntimeArgs(program, kernel, eth_logical_core, rt_args);
    }

    log_debug(
        tt::LogMetal,
        "Fabric router kernel created: eth_chan={}, direction={}, is_master={}",
        eth_chan,
        get_eth_direction(),
        eth_chan == ctx.master_router_chan);
}

FabricDatamoverBuilderBase* ComputeMeshRouterBuilder::get_builder_for_vc_channel(uint32_t vc, uint32_t channel) const {
    auto mapping = channel_mapping_.get_sender_mapping(vc, channel);
    if (mapping.builder_type == BuilderType::TENSIX) {
        TT_FATAL(tensix_builder_.has_value(), "Tensix builder required but not present");
        return const_cast<FabricTensixDatamoverBuilder*>(&tensix_builder_.value());
    }
    return erisc_builder_.get();
}

}  // namespace tt::tt_fabric
