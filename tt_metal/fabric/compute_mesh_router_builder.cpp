// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_mesh_router_builder.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/third_party/umd/device/api/umd/device/types/core_coordinates.hpp"
#include <tt_stl/assert.hpp>
#include <algorithm>

namespace tt::tt_fabric {

ComputeMeshRouterBuilder::ComputeMeshRouterBuilder(
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder,
    std::optional<FabricTensixDatamoverBuilder> tensix_builder,
    FabricRouterChannelMapping channel_mapping) :
    erisc_builder_(std::move(erisc_builder)),
    tensix_builder_(std::move(tensix_builder)),
    channel_mapping_(std::move(channel_mapping)) {
    TT_FATAL(erisc_builder_ != nullptr, "Erisc builder cannot be null");
}

std::unique_ptr<ComputeMeshRouterBuilder> ComputeMeshRouterBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& fabric_program,
    umd::CoreCoord eth_logical_core,
    FabricNodeId fabric_node_id,
    FabricNodeId remote_fabric_node_id,
    const tt::tt_fabric::FabricEriscDatamoverConfig& edm_config,
    tt::tt_fabric::eth_chan_directions eth_direction,
    bool dispatch_link,
    tt::tt_fabric::chan_id_t eth_chan,
    tt::tt_fabric::Topology topology) {
    bool fabric_tensix_extension_enabled = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
                                           tt::tt_fabric::FabricTensixConfig::DISABLED;
    bool fabric_tensix_extension_mux_mode =
        tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() == tt::tt_fabric::FabricTensixConfig::MUX;
    bool fabric_tensix_extension_udm_mode =
        tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() == tt::tt_fabric::FabricTensixConfig::UDM;

    // Determine if tensix builder will be created (reusable condition)
    bool will_create_tensix_builder = fabric_tensix_extension_enabled && !dispatch_link;
    bool downstream_is_tensix_builder = will_create_tensix_builder && fabric_tensix_extension_mux_mode;

    // Create channel mapping EARLY (needed for computing injection flags)
    auto channel_mapping =
        tt::tt_fabric::FabricRouterChannelMapping(topology, eth_direction, downstream_is_tensix_builder);
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
    if (will_create_tensix_builder) {
        size_t tensix_num_channels = builder_config::get_num_tensix_sender_channels(
            topology, tt::tt_metal::MetalContext::instance().get_fabric_tensix_config());
        auto tensix_to_router_channel_map =
            get_variant_to_router_channel_map(channel_mapping, BuilderType::TENSIX, tensix_num_channels);
        tensix_injection_flags = get_child_builder_variant_sender_channel_injection_flags(
            router_injection_flags, tensix_to_router_channel_map);
    }

    // NOW create erisc builder with computed injection flags
    auto edm_builder =
        std::make_unique<tt::tt_fabric::FabricEriscDatamoverBuilder>(tt::tt_fabric::FabricEriscDatamoverBuilder::build(
            device,
            fabric_program,
            eth_logical_core,
            fabric_node_id,
            remote_fabric_node_id,
            edm_config,
            std::move(erisc_injection_flags),
            false, /* build_in_worker_connection_mode */
            eth_direction,
            downstream_is_tensix_builder));

    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode()) {
        // Enable updates at a fixed interval for link stability and link status updates
        constexpr uint32_t k_BlackholeFabricRouterContextSwitchInterval = 32;
        edm_builder->set_firmware_context_switch_interval(k_BlackholeFabricRouterContextSwitchInterval);
        edm_builder->set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::INTERVAL);
    }

    // Create tensix builder if needed
    std::optional<tt::tt_fabric::FabricTensixDatamoverBuilder> tensix_builder_opt;
    if (will_create_tensix_builder) {
        tensix_builder_opt = tt::tt_fabric::FabricTensixDatamoverBuilder::build(
            device,
            fabric_program,
            fabric_node_id,
            remote_fabric_node_id,
            eth_chan,
            eth_direction,
            std::move(tensix_injection_flags));
    }

    auto router_builder = std::make_unique<tt::tt_fabric::ComputeMeshRouterBuilder>(
        std::move(edm_builder), std::move(tensix_builder_opt), std::move(channel_mapping));

    // Setup the local relay kernel connection if in UDM mode
    if (fabric_tensix_extension_udm_mode && router_builder->has_tensix_builder()) {
        router_builder->connect_to_local_tensix_builder(router_builder->get_tensix_builder());
    }

    return router_builder;
}

void ComputeMeshRouterBuilder::connect_to_downstream_router_over_noc(
    FabricRouterBuilder& other_interface, uint32_t vc) {
    // For now, cast to concrete type since we need access to internals
    // Phase 4 will abstract this better with establish_bidirectional_connection
    auto* other_ptr = dynamic_cast<ComputeMeshRouterBuilder*>(&other_interface);
    TT_FATAL(other_ptr != nullptr, "connect_to_downstream_router_over_noc requires ComputeMeshRouterBuilder");
    ComputeMeshRouterBuilder& other = *other_ptr;
    auto connect_vc = [&](uint32_t vc_index,
                          FabricDatamoverBuilderBase* downstream_builder,
                          uint32_t logical_sender_channel_idx) {
        log_debug(
            tt::LogTest,
            "Router at x={}, y={}, Direction={}, FabricNodeId={} :: Connecting VC{} to downstream router at x={}, "
            "y={}, Direction={}",
            erisc_builder_->get_noc_x(),
            erisc_builder_->get_noc_y(),
            erisc_builder_->get_direction(),
            erisc_builder_->local_fabric_node_id,
            vc_index,
            downstream_builder->get_noc_x(),
            downstream_builder->get_noc_y(),
            downstream_builder->get_direction());

        // auto send_chan = get_sender_channel(is_2D_routing, this->get_direction(), vc_index);
        auto downstream_mapping = other.channel_mapping_.get_sender_mapping(vc_index, logical_sender_channel_idx);
        uint32_t internal_channel_id = downstream_mapping.internal_sender_channel_id;

        // Need to call the templated method with the correct derived type
        // NOTE: erisc_builder_ is hardcoded here because there is currently no scenario where tensix forwards to tensix
        //       however when that is enabled, this code should be updated to point to the src builder dynamically
        if (auto* downstream_erisc_builder = dynamic_cast<FabricEriscDatamoverBuilder*>(downstream_builder)) {
            erisc_builder_->setup_downstream_vc_connection(downstream_erisc_builder, vc_index, internal_channel_id);
        } else if (auto* downstream_tensix_builder = dynamic_cast<FabricTensixDatamoverBuilder*>(downstream_builder)) {
            erisc_builder_->setup_downstream_vc_connection(downstream_tensix_builder, vc_index, internal_channel_id);
        }
    };

    auto get_downstream_builder_for_vc = [&](uint32_t vc_index,
                                             uint32_t sender_channel_idx) -> FabricDatamoverBuilderBase* {
        auto mapping = other.channel_mapping_.get_sender_mapping(vc_index, sender_channel_idx);

        if (mapping.builder_type == BuilderType::TENSIX) {
            TT_FATAL(
                other.tensix_builder_.has_value(),
                "Channel mapping requires TENSIX builder for VC{} channel {}, but tensix builder not present",
                vc_index,
                sender_channel_idx);
            return &other.tensix_builder_.value();
        } else {
            return other.erisc_builder_.get();
        }
    };

    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();
    if (vc == 0) {
        TT_FATAL(
            !erisc_builder_->build_in_worker_connection_mode,
            "Tried to connect router to downstream in worker connection mode");

        // Helper to get the downstream builder for a specific VC based on channel mapping
        uint32_t sender_channel_idx = get_downstream_sender_channel(is_2D_routing, other.get_direction());
        // Connect VC0
        connect_vc(0, get_downstream_builder_for_vc(0, sender_channel_idx), sender_channel_idx);
    }
}

SenderWorkerAdapterSpec ComputeMeshRouterBuilder::build_connection_to_fabric_channel(
    uint32_t vc, uint32_t sender_channel_idx) {
    // This method returns connection info for a sender channel, which can be used by
    // downstream routers to connect to this sender channel
    auto sender_mapping = channel_mapping_.get_sender_mapping(vc, sender_channel_idx);

    if (sender_mapping.builder_type == BuilderType::ERISC) {
        return erisc_builder_->build_connection_to_fabric_channel(sender_mapping.internal_sender_channel_id);
    } else if (sender_mapping.builder_type == BuilderType::TENSIX) {
        TT_FATAL(tensix_builder_.has_value(), "Tensix builder not available but mapping requires it");
        return tensix_builder_->build_connection_to_fabric_channel(sender_mapping.internal_sender_channel_id);
    } else {
        TT_FATAL(false, "Unknown builder type");
    }
}

uint32_t ComputeMeshRouterBuilder::get_downstream_sender_channel(
    const bool is_2D_routing, const eth_chan_directions downstream_direction) const {
    if (!is_2D_routing) {
        return 1;  // 1D: sender channel 1 for forwarding
    }

    // Sender channel 0 is always reserved for the local worker.
    //
    // Sender channels 1–3 correspond to the three upstream neighbors relative
    // to the downstream router's direction.
    //
    // The mapping from receiver direction → sender channel depends on the
    // downstream forwarding direction:
    //
    //   • Downstream = EAST:
    //         WEST  → channel 1
    //         NORTH → channel 2
    //         SOUTH → channel 3
    //
    //   • Downstream = WEST:
    //         EAST  → channel 1
    //         NORTH → channel 2
    //         SOUTH → channel 3
    //
    //   • Downstream = NORTH:
    //         EAST  → channel 1
    //         WEST  → channel 2
    //         SOUTH → channel 3
    //
    //   • Downstream = SOUTH:
    //         EAST  → channel 1
    //         WEST  → channel 2
    //         NORTH → channel 3

    size_t downstream_compact_index_for_upstream;
    if (downstream_direction == eth_chan_directions::EAST) {
        // EAST downstream: WEST(1)→0, NORTH(2)→1, SOUTH(3)→2
        downstream_compact_index_for_upstream = this->get_direction() - 1;
    } else {
        // For other downstream directions: if upstream < downstream, use as-is; else subtract 1
        downstream_compact_index_for_upstream =
            (this->get_direction() < downstream_direction) ? this->get_direction() : (this->get_direction() - 1);
    }

    // Sender channel = 1 + compact index (since channel 0 is for local worker)
    return 1 + downstream_compact_index_for_upstream;
}

eth_chan_directions ComputeMeshRouterBuilder::get_direction() const { return erisc_builder_->get_direction(); }

size_t ComputeMeshRouterBuilder::get_noc_x() const { return erisc_builder_->get_noc_x(); }

size_t ComputeMeshRouterBuilder::get_noc_y() const { return erisc_builder_->get_noc_y(); }

size_t ComputeMeshRouterBuilder::get_configured_risc_count() const {
    return erisc_builder_->get_configured_risc_count();
}

FabricNodeId ComputeMeshRouterBuilder::get_local_fabric_node_id() const { return erisc_builder_->local_fabric_node_id; }

FabricNodeId ComputeMeshRouterBuilder::get_peer_fabric_node_id() const { return erisc_builder_->peer_fabric_node_id; }

std::vector<bool> ComputeMeshRouterBuilder::compute_sender_channel_injection_flags_for_vc(
    Topology topology, eth_chan_directions direction, uint32_t /*vc*/, uint32_t num_channels) {
    std::vector<bool> injection_flags(num_channels, false);

    // Early return for Linear/Mesh - no injection channels marked
    if (topology == Topology::Linear || topology == Topology::Mesh) {
        return injection_flags;
    }

    // VC0: Worker channel (idx 0) is always an injection channel
    injection_flags.at(0) = true;

    if (topology != Topology::Torus) {
        return injection_flags;
    }

    // For Torus: Turn channels are injection channels
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
    for (size_t variant_internal_ch = 0; variant_internal_ch < variant_to_router_channel_map.size();
         ++variant_internal_ch) {
        auto router_channel_opt = variant_to_router_channel_map.at(variant_internal_ch);

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

}  // namespace tt::tt_fabric
