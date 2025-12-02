// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/fabric/fabric_router_types.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

class FabricBuilderContext;

/**
 * FabricRouterBuilder - Abstract interface for fabric router builders
 *
 * This interface abstracts the differences between compute mesh and switch mesh
 * router builders, allowing polymorphic behavior while maintaining a consistent API.
 *
 * Implementations:
 * - ComputeMeshRouterBuilder: For compute mesh routers (with tensix support)
 * - SwitchMeshRouterBuilder: For switch mesh routers (future, routing-only)
 *
 * Usage:
 *   auto router = FabricRouterBuilder::create(device, program, local_node, location, spec);
 *   router->connect_to_downstream_router_over_noc(*other_router, vc);
 */
class FabricRouterBuilder {
public:
    virtual ~FabricRouterBuilder() = default;

    /**
     * Factory method to create the appropriate router builder based on spec
     *
     * @param device The device to build on
     * @param program The fabric program
     * @param local_node The local fabric node ID
     * @param location Router location (eth_chan, remote_node, direction, is_dispatch)
     * @param spec Router build specification (determines builder type)
     * @return A unique_ptr to the appropriate FabricRouterBuilder implementation
     */
    static std::unique_ptr<FabricRouterBuilder> create(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricNodeId local_node,
        const RouterLocation& location,
        const RouterBuildSpec& spec);

    // ============ Connection Methods ============

    /**
     * Connect to downstream router over NOC (one-way connection)
     *
     * @param other The other router builder to connect to
     * @param vc Virtual channel ID
     */
    virtual void connect_to_downstream_router_over_noc(FabricRouterBuilder& other, uint32_t vc) = 0;

    /**
     * Build connection to fabric channel (for sender channels)
     *
     * @param vc Virtual channel ID
     * @param sender_channel_idx Logical sender channel index within the VC
     * @return SenderWorkerAdapterSpec for external connections
     */
    virtual SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc, uint32_t sender_channel_idx) = 0;

    /**
     * Get downstream sender channel index
     *
     * @param is_2D_routing Whether 2D routing is enabled
     * @param downstream_direction The downstream direction
     * @return The sender channel index for downstream connections
     */
    virtual uint32_t get_downstream_sender_channel(
        bool is_2D_routing, eth_chan_directions downstream_direction) const = 0;

    // ============ Property Getters ============

    virtual eth_chan_directions get_direction() const = 0;
    virtual size_t get_noc_x() const = 0;
    virtual size_t get_noc_y() const = 0;
    virtual size_t get_configured_risc_count() const = 0;
    virtual FabricNodeId get_local_fabric_node_id() const = 0;
    virtual FabricNodeId get_peer_fabric_node_id() const = 0;

    // ============ Builder Access ============
    // These provide access to underlying builders for operations that haven't
    // been abstracted yet. Phase 4 will move more logic into the interface.

    virtual FabricEriscDatamoverBuilder& get_erisc_builder() = 0;
    virtual const FabricEriscDatamoverBuilder& get_erisc_builder() const = 0;

    virtual bool has_tensix_builder() const = 0;
    virtual FabricTensixDatamoverBuilder& get_tensix_builder() = 0;
    virtual const FabricTensixDatamoverBuilder& get_tensix_builder() const = 0;

    // ============ Mesh Type Query ============

    /**
     * Check if this is a switch mesh router
     * @return true if this is a switch mesh router, false for compute mesh
     */
    virtual bool is_switch_mesh() const = 0;

    // ============ Static Utilities ============

    /**
     * Determine if bubble flow control should be enabled based on topology
     */
    static bool is_bubble_flow_control_enabled(Topology topology) {
        return topology == Topology::Ring || topology == Topology::Torus;
    }
};

}  // namespace tt::tt_fabric
