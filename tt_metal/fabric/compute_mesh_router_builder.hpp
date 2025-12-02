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
     * Constructor
     *
     * @param erisc_builder The erisc datamover builder (always required)
     * @param tensix_builder Optional tensix datamover builder
     * @param channel_mapping The channel mapping object that defines logical to internal channel mappings
     * @param eth_chan The ethernet channel ID for this router
     */
    ComputeMeshRouterBuilder(
        std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder,
        std::optional<FabricTensixDatamoverBuilder> tensix_builder,
        FabricRouterChannelMapping channel_mapping,
        chan_id_t eth_chan);

    /**
     * Build a ComputeMeshRouterBuilder with all necessary components
     *
     * @param device The device to build on
     * @param fabric_program The fabric program
     * @param eth_logical_core The ethernet logical core
     * @param fabric_node_id The local fabric node ID
     * @param remote_fabric_node_id The remote fabric node ID
     * @param edm_config The EDM configuration
     * @param eth_direction The ethernet direction
     * @param dispatch_link Whether this is a dispatch link
     * @param eth_chan The ethernet channel (for tensix builder)
     * @param topology The fabric topology
     * @return A unique_ptr to the constructed ComputeMeshRouterBuilder
     */
    static std::unique_ptr<ComputeMeshRouterBuilder> build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& fabric_program,
        umd::CoreCoord eth_logical_core,
        FabricNodeId fabric_node_id,
        FabricNodeId remote_fabric_node_id,
        const tt::tt_fabric::FabricEriscDatamoverConfig& edm_config,
        tt::tt_fabric::eth_chan_directions eth_direction,
        bool dispatch_link,
        tt::tt_fabric::chan_id_t eth_chan,
        tt::tt_fabric::Topology topology);

    // ============ FabricRouterBuilder Interface Implementation ============

    void connect_to_downstream_router_over_noc(FabricRouterBuilder& other, uint32_t vc) override;

    void configure_link(
        FabricRouterBuilder& peer, uint32_t link_idx, uint32_t num_links, Topology topology, bool is_galaxy) override;

    void compile_ancillary_kernels(tt::tt_metal::Program& program) override;

    void create_kernel(tt::tt_metal::Program& program, const KernelCreationContext& ctx) override;

    SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc, uint32_t sender_channel_idx) override;

    uint32_t get_downstream_sender_channel(bool is_2D_routing, eth_chan_directions downstream_direction) const override;

    // Property getters
    eth_chan_directions get_direction() const override;
    size_t get_noc_x() const override;
    size_t get_noc_y() const override;
    size_t get_configured_risc_count() const override;
    FabricNodeId get_local_fabric_node_id() const override;
    FabricNodeId get_peer_fabric_node_id() const override;

    // Builder access (erisc_builder is in interface, tensix is compute-mesh specific)
    FabricEriscDatamoverBuilder& get_erisc_builder() override { return *erisc_builder_; }
    const FabricEriscDatamoverBuilder& get_erisc_builder() const override { return *erisc_builder_; }

    // Tensix builder access - compute mesh specific, not in interface
    bool has_tensix_builder() const { return tensix_builder_.has_value(); }
    FabricTensixDatamoverBuilder& get_tensix_builder() {
        TT_FATAL(tensix_builder_.has_value(), "Tensix builder not available");
        return tensix_builder_.value();
    }
    const FabricTensixDatamoverBuilder& get_tensix_builder() const {
        TT_FATAL(tensix_builder_.has_value(), "Tensix builder not available");
        return tensix_builder_.value();
    }

    // Mesh type query
    bool is_switch_mesh() const override { return false; }  // This is compute mesh

private:
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

    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder_;
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_;
    FabricRouterChannelMapping channel_mapping_;
    chan_id_t eth_chan_;  // Cached for kernel creation (master router determination)
};

}  // namespace tt::tt_fabric
