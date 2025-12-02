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
 *
 * Compute mesh routers support:
 * - VC0 for normal traffic
 * - Local worker channels for compute workloads
 * - Optional Tensix mux extension
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

    void connect_to_downstream_router_over_noc(FabricRouterBuilder& other, uint32_t vc) override;

    void configure_connection(
        FabricRouterBuilder& peer, uint32_t link_idx, uint32_t num_links, Topology topology, bool is_galaxy) override;

    void configure_for_dispatch() override;

    void compile_ancillary_kernels(tt::tt_metal::Program& program) override;

    void create_kernel(tt::tt_metal::Program& program, const KernelCreationContext& ctx) override;

    // ============ Compute-Mesh Specific Methods ============

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
     */
    static std::vector<bool> compute_sender_channel_injection_flags_for_vc(
        Topology topology, eth_chan_directions direction, uint32_t vc, uint32_t num_channels);

    /**
     * Map router-level injection flags to a child builder variant's channel space.
     */
    static std::vector<bool> get_child_builder_variant_sender_channel_injection_flags(
        const std::vector<bool>& router_injection_flags,
        const std::vector<std::optional<size_t>>& variant_to_router_channel_map);

    /**
     * Build a reverse mapping from a builder variant's internal channels to router's external facing channel IDs.
     */
    static std::vector<std::optional<size_t>> get_variant_to_router_channel_map(
        const FabricRouterChannelMapping& channel_mapping,
        BuilderType builder_type,
        size_t variant_num_sender_channels);

    /**
     * Connect the local tensix builder to the erisc builder in UDM mode
     */
    void connect_to_local_tensix_builder(FabricTensixDatamoverBuilder& tensix_builder);

    // Compute-mesh specific state
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder_;
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_;
    FabricRouterChannelMapping channel_mapping_;
};

}  // namespace tt::tt_fabric
