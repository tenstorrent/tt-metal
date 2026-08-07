// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/builder/connection_registry.hpp"

namespace tt::tt_fabric {

// Forward declarations
class FabricDatamoverBuilderBase;
class ControlPlane;

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
     * @param chip_facts The chip's routing facts (edge capabilities, ring predicates), bound at chip scope
     * @param connection_registry Optional registry to record connections for testing
     * @return A unique_ptr to the constructed ComputeMeshRouterBuilder
     */
    static std::unique_ptr<ComputeMeshRouterBuilder> build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricNodeId local_node,
        const RouterLocation& location,
        const ChipRoutingFacts& chip_facts,
        std::shared_ptr<ConnectionRegistry> connection_registry = nullptr);

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

    uint32_t get_downstream_sender_channel(
        bool is_2D_routing, eth_chan_directions downstream_direction, uint32_t vc) const;

    eth_chan_directions get_eth_direction() const;
    size_t get_noc_x() const;
    size_t get_noc_y() const;
    size_t get_configured_risc_count() const;

    /**
     * Get the builder for a specific VC/channel combination
     * Encapsulates channel mapping + builder resolution
     *
     * @param vc Virtual channel index
     * @param channel Sender channel index within the VC
     * @return Pointer to the appropriate builder (erisc or tensix)
     */
    FabricDatamoverBuilderBase* get_builder_for_vc_channel(uint32_t vc, uint32_t channel) const;

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
        RouterVcShape vc_shape,
        RouterTurnSet turns_by_vc,
        bool downstream_is_tensix_builder,
        std::shared_ptr<ConnectionRegistry> connection_registry);

    /**
     * Generic helper to establish connections from this router to a downstream router.
     * Iterates through all VCs and sender channels, establishing every direction-matching target.
     *
     * @param downstream_router The target router to connect to
     */
    void establish_connections_to_router(ComputeMeshRouterBuilder& downstream_router);

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
     * Iterates the shape's (vc, channel) pairs, keeps the ones the given builder variant owns
     * (builder_type_for_vc), and maps their internal channel IDs to router channel IDs.
     *
     * @param vc_shape The router's per-VC channel shape
     * @param downstream_is_tensix_builder Whether VC0 channels are tensix-owned (MUX mode)
     * @param builder_type Which builder variant (ERISC or TENSIX)
     * @param variant_num_sender_channels Number of sender channels the variant has
     * @return Vector where index is the variant's internal channel ID, value is optional router channel ID
     *         (nullopt for internal-only channels not exposed to external topology)
     */
    static std::vector<std::optional<size_t>> get_variant_to_router_channel_map(
        const RouterVcShape& vc_shape,
        bool downstream_is_tensix_builder,
        BuilderType builder_type,
        size_t variant_num_sender_channels);

    /**
     * Connect the local tensix builder to the erisc builder in UDM mode
     * This sets up the receiver-to-relay connection for the local tensix relay interface
     *
     * @param tensix_builder The tensix builder to connect
     */
    void connect_to_local_tensix_builder(FabricTensixDatamoverBuilder& tensix_builder);

    // Compute-mesh specific state
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder_;
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_;
    RouterVcShape vc_shape_;
    RouterTurnSet turns_by_vc_;
    bool downstream_is_tensix_builder_ = false;
    std::shared_ptr<ConnectionRegistry> connection_registry_;
    bool is_inter_mesh_;  // True if this router connects different meshes
};

}  // namespace tt::tt_fabric
