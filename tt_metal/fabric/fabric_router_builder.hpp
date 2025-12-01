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
     * Build a FabricRouterBuilder with all necessary components
     *
     * @param device The device to build on
     * @param fabric_program The fabric program
     * @param eth_logical_core The ethernet logical core
     * @param fabric_node_id The local fabric node ID
     * @param remote_fabric_node_id The remote fabric node ID
     * @param edm_config The EDM configuration
     * @param fabric_edm_type The fabric EDM type
     * @param eth_direction The ethernet direction
     * @param dispatch_link Whether this is a dispatch link
     * @param eth_chan The ethernet channel (for tensix builder)
     * @param topology The fabric topology
     * @return A unique_ptr to the constructed FabricRouterBuilder
     */
    static std::unique_ptr<FabricRouterBuilder> build(
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
        tt::tt_fabric::Topology topology);

    /**
     * Connect the downstream router over noc or Ethernet. Iterates through all VCs and channels
     * between the routers and connects them.
     *
     * Establishes one-way connection
     *
     * @param other The other router builder to connect to
     * @param vc Virtual channel ID
     */
    void connect_to_downstream_router_over_noc(FabricRouterBuilder& other, uint32_t vc);

    /**
     * Build connection to fabric channel (for sender channels)
     *
     * @param vc Virtual channel ID
     * @param sender_channel_idx Logical sender channel index within the VC
     * @return SenderWorkerAdapterSpec for external connections
     */
    SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc, uint32_t sender_channel_idx);


    uint32_t get_downstream_sender_channel(bool is_2D_routing, eth_chan_directions downstream_direction) const;

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
    /**
     * Connect the local tensix builder to the erisc builder in UDM mode
     * This sets up the receiver-to-relay connection for the local tensix relay interface
     *
     * @param tensix_builder The tensix builder to connect
     */
    void connect_to_local_tensix_builder(FabricTensixDatamoverBuilder& tensix_builder);

    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder_;
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_;
    FabricRouterChannelMapping channel_mapping_;
};

}  // namespace tt::tt_fabric
