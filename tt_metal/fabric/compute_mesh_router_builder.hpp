// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"

namespace tt::tt_fabric {

/**
 * ComputeMeshRouterBuilder
 *
 * Implementation of FabricRouterBuilder for compute mesh routers.
 * Wraps FabricEriscDatamoverBuilder (always present) and optionally
 * FabricTensixDatamoverBuilder (0 or 1). This wrapper acts as the external interface for
 * router connections, translating between logical channels (VC + sender/receiver indices)
 * and internal builder channels.
 */
class ComputeMeshRouterBuilder : public FabricRouterBuilder {
public:
    /**
     * Build a ComputeMeshRouterBuilder with all necessary components.
     * Handles its own config lookup based on location and fabric context.
     *
     * @param device The device to build on
     * @param program The fabric program
     * @param local_node The local fabric node ID
     * @param location Router location (eth_chan, remote_node, direction, is_dispatch)
     * @return A unique_ptr to the constructed ComputeMeshRouterBuilder
     */
    static std::unique_ptr<ComputeMeshRouterBuilder> build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricNodeId local_node,
        const RouterLocation& location);

    // ============ FabricRouterBuilder Interface Implementation ============

    void configure_connection(
        FabricRouterBuilder& peer, uint32_t link_idx, uint32_t num_links, Topology topology, bool is_galaxy) override;

    void configure_for_dispatch() override;

    void compile_ancillary_kernels(tt::tt_metal::Program& program) override;

    void create_kernel(tt::tt_metal::Program& program, const KernelCreationContext& ctx) override;

    // ============ Compute-Mesh Specific Methods ============

    /**
     * Build connection to fabric channel (for sender channels)
     *
     * @param vc Virtual channel ID
     * @param sender_channel_idx Logical sender channel index within the VC
     * @return SenderWorkerAdapterSpec for external connections
     */
    SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc, uint32_t sender_channel_idx);

    uint32_t get_downstream_sender_channel(bool is_2D_routing, eth_chan_directions downstream_direction) const;

    eth_chan_directions get_eth_direction() const;
    size_t get_noc_x() const;
    size_t get_noc_y() const;
    size_t get_configured_risc_count() const;

    // ============ Compute-Mesh Specific Accessors ============

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
    // Private constructor - use build() factory method
    ComputeMeshRouterBuilder(
        FabricNodeId local_node,
        const RouterLocation& location,
        std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder,
        std::optional<FabricTensixDatamoverBuilder> tensix_builder,
        FabricRouterChannelMapping channel_mapping);

    /**
     * Compute which sender channels are traffic injection channels for a specific VC.
     * Injection channels are channels where traffic originates (not forwarded):
     * - VC0: Worker channel (channel 0) is always an injection channel
     * - VC1+: No worker channel; channel 0 maps to VC0's channel 1 semantics
     * - In Torus topology, "turn channels" are injection channels (where traffic changes direction)
     *
     * @param topology The fabric topology
     * @param direction The router's direction
     * @param vc The virtual channel to compute flags for
     * @param num_channels Number of channels in this VC
     * @return Array indicating which sender channels are injection channels for this VC
     */
    static std::vector<bool> compute_sender_channel_injection_flags_for_vc(
        Topology topology, eth_chan_directions direction, uint32_t vc, uint32_t num_channels);

    /**
     * Map router-level injection flags to a child builder variant's channel space.
     * This is a generic helper that doesn't know which builder variant it's serving.
     *
     * @param router_injection_flags Injection flags indexed by router's semantic channel IDs
     * @param variant_to_router_channel_map Maps variant's internal channel index to optional router channel index
     *        (nullopt for internal-only channels)
     * @return Injection flags for the variant builder (false for internal-only channels)
     */
    static std::vector<bool> get_child_builder_variant_sender_channel_injection_flags(
        const std::vector<bool>& router_injection_flags,
        const std::vector<std::optional<size_t>>& variant_to_router_channel_map);

    /**
     * Build a reverse mapping from a builder variant's internal channels to router's external facing channel IDs.
     * For kernel internal channels that aren't externally facing, the mapping will be nullopt.
     * Iterates through the channel mapping to find which logical channels are handled by the given builder type,
     * then maps their internal channel IDs to router channel IDs.
     *
     * @param channel_mapping The channel mapping to use (already knows topology and mode)
     * @param builder_type Which builder variant (ERISC or TENSIX)
     * @param variant_num_sender_channels Number of sender channels the variant has
     * @return Vector where index is the variant's internal channel ID, value is optional router channel ID
     *         (nullopt for internal-only channels not exposed to external topology)
     */
    static std::vector<std::optional<size_t>> get_variant_to_router_channel_map(
        const FabricRouterChannelMapping& channel_mapping,
        BuilderType builder_type,
        size_t variant_num_sender_channels);

    /**
     * Connect the local tensix builder to the erisc builder in UDM mode
     * This sets up the receiver-to-relay connection for the local tensix relay interface
     *
     * @param tensix_builder The tensix builder to connect
     */
    void connect_to_local_tensix_builder(FabricTensixDatamoverBuilder& tensix_builder);

    /**
     * Connect the downstream router over noc or Ethernet. Iterates through all VCs and channels
     * between the routers and connects them.
     *
     * Establishes one-way connection
     *
     * @param other The other router builder to connect to
     * @param vc Virtual channel ID
     */
    void connect_to_downstream_router_over_noc(ComputeMeshRouterBuilder& other, uint32_t vc);

    // Compute-mesh specific state
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder_;
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_;
    FabricRouterChannelMapping channel_mapping_;
};

}  // namespace tt::tt_fabric
