// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_builder.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
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
    bool fabric_tensix_extension_enabled,
    bool dispatch_link,
    tt::tt_fabric::chan_id_t eth_chan,
    tt::tt_fabric::Topology topology) {

    bool has_tensix_extension = false;
    if (fabric_tensix_extension_enabled && !dispatch_link) {
        has_tensix_extension = true;
    }

    auto edm_builder = std::make_unique<tt::tt_fabric::FabricEriscDatamoverBuilder>(
        tt::tt_fabric::FabricEriscDatamoverBuilder::build(
            device,
            fabric_program,
            eth_logical_core,
            fabric_node_id,
            remote_fabric_node_id,
            edm_config,
            false, /* build_in_worker_connection_mode */
            fabric_edm_type,
            eth_direction,
            has_tensix_extension));

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
            auto tensix_builder = tt::tt_fabric::FabricTensixDatamoverBuilder::build(
                device, fabric_program, fabric_node_id, remote_fabric_node_id, eth_chan, eth_direction);
            tensix_builder_opt = tensix_builder;
        }
    }

    // Create channel mapping
    auto channel_mapping = tt::tt_fabric::FabricRouterChannelMapping(topology, eth_direction);

    return std::make_unique<tt::tt_fabric::FabricRouterBuilder>(
        std::move(edm_builder), tensix_builder_opt, std::move(channel_mapping));
}


void FabricRouterBuilder::connect_to_downstream_router(
    FabricRouterBuilder& other,
    uint32_t vc,
    [[maybe_unused]] uint32_t sender_channel_idx,
    uint32_t receiver_channel_idx) {
    // auto sender_mapping = channel_mapping_.get_sender_mapping(vc, sender_channel_idx);
    auto receiver_mapping = other.channel_mapping_.get_receiver_mapping(vc, receiver_channel_idx);

    // TT_FATAL(sender_mapping.builder_type == BuilderType::ERISC, "Internal Error. Tried to connect to a downstream router over Ethernet, but the source is not an erisc fabric builder.");
    // erisc_builder_->connect_to_downstream_edm(*other.erisc_builder_);

    // Create FabricDatamoverBuilder variant for the downstream builder
    // If both routers have tensix builders, connect VC0 to tensix (performance optimization)
    // Otherwise, use the receiver_mapping to determine which builder to connect to
    bool both_have_tensix = this->tensix_builder_.has_value() && other.tensix_builder_.has_value();

    // Initialize variants directly - cannot default-construct variant with reference_wrapper
    auto get_downstream_builder = [&]() -> FabricDatamoverBuilder {
        if (both_have_tensix) {
            return std::ref(other.tensix_builder_.value());
        } else if (receiver_mapping.builder_type == BuilderType::TENSIX) {
            TT_FATAL(other.tensix_builder_.has_value(), "Other router's tensix builder not available");
            return std::ref(other.tensix_builder_.value());
        } else {
            return std::ref(*other.erisc_builder_);
        }
    };
    FabricDatamoverBuilder downstream_builder = get_downstream_builder();
    FabricDatamoverBuilder vc1_downstream_builder = std::ref(*other.erisc_builder_);

    // If both routers have tensix builders, use the two-argument version
    // (connects VC0 to tensix, VC1 to erisc)
    // Otherwise, use the single-argument version (connects both VC0 and VC1 to erisc)
    if (both_have_tensix) {
        erisc_builder_->connect_to_downstream_edm(downstream_builder, vc1_downstream_builder);
    } else {
        erisc_builder_->connect_to_downstream_edm(downstream_builder);
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

}  // namespace tt::tt_fabric
