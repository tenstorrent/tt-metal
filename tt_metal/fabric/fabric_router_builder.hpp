// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"

namespace tt::tt_fabric {

/**
 * FabricRouterBuilder
 * 
 * Wrapper class that wraps FabricEriscDatamoverBuilder (always present) and optionally
 * FabricTensixDatamoverBuilder (0 or 1). This wrapper acts as the external interface for
 * router connections, translating between logical channels (VC + sender/receiver indices)
 * and internal builder channels.
 */
class FabricRouterBuilder {
public:
    /**
     * Constructor
     * 
     * @param erisc_builder The erisc datamover builder (always required)
     * @param tensix_builder Optional tensix datamover builder
     * @param channel_mapping The channel mapping object that defines logical to internal channel mappings
     */
    FabricRouterBuilder(
        std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder,
        std::optional<FabricTensixDatamoverBuilder> tensix_builder,
        FabricRouterChannelMapping channel_mapping);

    /**
     * Connect to a local router over NOC (intra-device connection)
     * 
     * Establishes one-way connection: sender channel on this router → receiver channel on other router over NOC
     * 
     * @param other The other router builder to connect to
     * @param vc Virtual channel ID
     * @param sender_channel_idx Logical sender channel index within the VC
     * @param receiver_channel_idx Logical receiver channel index within the VC on the other router
     */
    void connect_to_local_router_over_noc(
        FabricRouterBuilder& other,
        uint32_t vc,
        uint32_t sender_channel_idx,
        uint32_t receiver_channel_idx);

    /**
     * Connect to a remote router over Ethernet (inter-device connection)
     * 
     * Establishes one-way connection: sender channel on this router → receiver channel on other router over Ethernet
     * 
     * @param other The other router builder to connect to
     * @param vc Virtual channel ID
     * @param sender_channel_idx Logical sender channel index within the VC
     * @param receiver_channel_idx Logical receiver channel index within the VC on the other router
     */
    void connect_to_remote_router_over_ethernet(
        FabricRouterBuilder& other,
        uint32_t vc,
        uint32_t sender_channel_idx,
        uint32_t receiver_channel_idx);

    /**
     * Build connection to fabric channel (for sender channels)
     * 
     * @param vc Virtual channel ID
     * @param sender_channel_idx Logical sender channel index within the VC
     * @return SenderWorkerAdapterSpec for external connections
     */
    SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc, uint32_t sender_channel_idx);

    /**
     * Build connection to fabric receiver channel (for receiver channels, if needed)
     * 
     * @param vc Virtual channel ID
     * @param receiver_channel_idx Logical receiver channel index within the VC
     * @return Connection specification (implementation depends on use case)
     */
    // Note: Receiver channel connection building may not be needed in all cases
    // This is a placeholder for future use

    // Getters/delegators for wrapped builder properties
    eth_chan_directions get_direction() const;
    size_t get_noc_x() const;
    size_t get_noc_y() const;
    size_t get_configured_risc_count() const;
    FabricNodeId get_local_fabric_node_id() const;
    FabricNodeId get_peer_fabric_node_id() const;

    // Access to wrapped builders (if needed for advanced use cases)
    FabricEriscDatamoverBuilder& get_erisc_builder() { return *erisc_builder_; }
    const FabricEriscDatamoverBuilder& get_erisc_builder() const { return *erisc_builder_; }
    bool has_tensix_builder() const { return tensix_builder_.has_value(); }
    FabricTensixDatamoverBuilder& get_tensix_builder() {
        TT_FATAL(tensix_builder_.has_value(), "Tensix builder not available");
        return tensix_builder_.value();
    }
    const FabricTensixDatamoverBuilder& get_tensix_builder() const {
        TT_FATAL(tensix_builder_.has_value(), "Tensix builder not available");
        return tensix_builder_.value();
    }

private:
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder_;
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_;
    FabricRouterChannelMapping channel_mapping_;

    // Helper methods for connection logic
    void connect_sender_to_receiver(
        const InternalSenderChannelMapping& sender_mapping,
        const InternalReceiverChannelMapping& receiver_mapping,
        FabricRouterBuilder& other,
        bool is_noc_connection);
};

}  // namespace tt::tt_fabric

