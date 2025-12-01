// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_builder.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/third_party/umd/device/api/umd/device/types/core_coordinates.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

FabricRouterBuilder::FabricRouterBuilder(
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder,
    std::optional<FabricTensixDatamoverBuilder> tensix_builder,
    FabricRouterChannelMapping channel_mapping) :
    erisc_builder_(std::move(erisc_builder)),
    tensix_builder_(std::move(tensix_builder)),
    channel_mapping_(std::move(channel_mapping)) {
    TT_FATAL(erisc_builder_ != nullptr, "Erisc builder cannot be null");
}

std::unique_ptr<FabricRouterBuilder> FabricRouterBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& fabric_program,
    umd::CoreCoord eth_logical_core,
    FabricNodeId fabric_node_id,
    FabricNodeId remote_fabric_node_id,
    const tt::tt_fabric::FabricEriscDatamoverConfig& edm_config,
    tt::tt_fabric::FabricEriscDatamoverType fabric_edm_type,
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

    bool downstream_is_tensix_extension = false;
    if (fabric_tensix_extension_mux_mode && !dispatch_link) {
        downstream_is_tensix_extension = true;
    }

    auto edm_builder =
        std::make_unique<tt::tt_fabric::FabricEriscDatamoverBuilder>(tt::tt_fabric::FabricEriscDatamoverBuilder::build(
            device,
            fabric_program,
            eth_logical_core,
            fabric_node_id,
            remote_fabric_node_id,
            edm_config,
            false, /* build_in_worker_connection_mode */
            fabric_edm_type,
            eth_direction,
            downstream_is_tensix_extension));

    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode()) {
        // Enable updates at a fixed interval for link stability and link status updates
        constexpr uint32_t k_BlackholeFabricRouterContextSwitchInterval = 32;
        edm_builder->set_firmware_context_switch_interval(k_BlackholeFabricRouterContextSwitchInterval);
        edm_builder->set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::INTERVAL);
    }

    // Create tensix builder if needed
    std::optional<tt::tt_fabric::FabricTensixDatamoverBuilder> tensix_builder_opt;
    if (fabric_tensix_extension_enabled) {
        // Only create tensix builder if this channel is not used by dispatch
        if (!dispatch_link) {
            tensix_builder_opt = tt::tt_fabric::FabricTensixDatamoverBuilder::build(
                device, fabric_program, fabric_node_id, remote_fabric_node_id, eth_chan, eth_direction);
        }
    }

    // Create channel mapping
    // Only enable tensix extension in mapping if we actually created a tensix builder
    bool downstream_is_tensix_builder = tensix_builder_opt.has_value() && fabric_tensix_extension_mux_mode;
    auto channel_mapping =
        tt::tt_fabric::FabricRouterChannelMapping(topology, eth_direction, downstream_is_tensix_builder);

    auto router_builder = std::make_unique<tt::tt_fabric::FabricRouterBuilder>(
        std::move(edm_builder), std::move(tensix_builder_opt), std::move(channel_mapping));

    // Setup the local relay kernel connection if in UDM mode
    if (fabric_tensix_extension_udm_mode && router_builder->has_tensix_builder()) {
        router_builder->connect_to_local_tensix_builder(router_builder->get_tensix_builder());
    }

    return router_builder;
}

void FabricRouterBuilder::connect_to_downstream_router_over_noc(
    FabricRouterBuilder& other, uint32_t vc) {
    auto connect_vc = [&](uint32_t vc_index, FabricDatamoverBuilderBase* downstream_builder, uint32_t logical_sender_channel_idx) {
        log_debug(
            tt::LogTest,
            "Router at x={}, y={}, Direction={}, FabricNodeId={} :: Connecting VC{} to downstream router at x={}, y={}, Direction={}",
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
            erisc_builder_->setup_downstream_vc_connection(downstream_erisc_builder, vc_index, internal_channel_id, vc_index == 1);
        } else if (auto* downstream_tensix_builder = dynamic_cast<FabricTensixDatamoverBuilder*>(downstream_builder)) {
            erisc_builder_->setup_downstream_vc_connection(downstream_tensix_builder, vc_index, internal_channel_id, vc_index == 1);
        }
    };

    auto get_downstream_builder_for_vc = [&](uint32_t vc_index, uint32_t sender_channel_idx) -> FabricDatamoverBuilderBase* {
        auto mapping = other.channel_mapping_.get_sender_mapping(vc_index, sender_channel_idx);

        if (mapping.builder_type == BuilderType::TENSIX) {
            TT_FATAL(other.tensix_builder_.has_value(),
                     "Channel mapping requires TENSIX builder for VC{} channel {}, but tensix builder not present",
                     vc_index, sender_channel_idx);
            return &other.tensix_builder_.value();
        } else {
            return other.erisc_builder_.get();
        }
    };

    const auto& fabric_context =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();
    if (vc == 0) {
        TT_FATAL(
            !erisc_builder_->build_in_worker_connection_mode,
            "Tried to connect router to downstream in worker connection mode");

        // Helper to get the downstream builder for a specific VC based on channel mapping

        uint32_t sender_channel_idx = get_downstream_sender_channel(is_2D_routing, other.get_direction());
        // Connect VC0
        connect_vc(0, get_downstream_builder_for_vc(0, sender_channel_idx), sender_channel_idx);
    } else if (vc == 1) {

        // Check if we should connect VC1 for deadlock avoidance
        if (!fabric_context.need_deadlock_avoidance_support(this->get_direction())) {
            return;
        }

        // For 2D routing we can only connect VC1 if the downstream is on the same axis
        bool connect_vc1 = true;
        if (is_2D_routing) {
            auto ds_dir = other.get_direction();

            connect_vc1 =
                (this->get_direction() == eth_chan_directions::EAST && ds_dir == eth_chan_directions::WEST) ||
                (this->get_direction() == eth_chan_directions::WEST && ds_dir == eth_chan_directions::EAST) ||
                (this->get_direction() == eth_chan_directions::NORTH && ds_dir == eth_chan_directions::SOUTH) ||
                (this->get_direction() == eth_chan_directions::SOUTH && ds_dir == eth_chan_directions::NORTH);
        }

        if (connect_vc1) {
            // Get the downstream builder for VC1 based on channel mapping
            // Note: VC1 only ever has one sender channel, so we index with offset 0 into VC1
            constexpr uint32_t sender_channel_index_into_vc1 = 0;
            connect_vc(1, get_downstream_builder_for_vc(1, sender_channel_index_into_vc1), sender_channel_index_into_vc1);
        }
    }
}

SenderWorkerAdapterSpec FabricRouterBuilder::build_connection_to_fabric_channel(
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

uint32_t FabricRouterBuilder::get_downstream_sender_channel(const bool is_2D_routing, const eth_chan_directions downstream_direction) const {

    if (!is_2D_routing) {
        return 1;  // 1D: sender channel 1 for forwarding
    }

    // Sender channel 0 is always reserved for the local worker.
    //
    // Sender channels 1–3 correspond to the three upstream neighbors relative
    // to the downstream router’s direction.
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

eth_chan_directions FabricRouterBuilder::get_direction() const {
    return erisc_builder_->get_direction();
}

size_t FabricRouterBuilder::get_noc_x() const {
    return erisc_builder_->get_noc_x();
}

size_t FabricRouterBuilder::get_noc_y() const {
    return erisc_builder_->get_noc_y();
}

size_t FabricRouterBuilder::get_configured_risc_count() const {
    return erisc_builder_->get_configured_risc_count();
}

FabricNodeId FabricRouterBuilder::get_local_fabric_node_id() const {
    return erisc_builder_->local_fabric_node_id;
}

FabricNodeId FabricRouterBuilder::get_peer_fabric_node_id() const {
    return erisc_builder_->peer_fabric_node_id;
}

void FabricRouterBuilder::connect_to_local_tensix_builder(FabricTensixDatamoverBuilder& tensix_builder) {
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
