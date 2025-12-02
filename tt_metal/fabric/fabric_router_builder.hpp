// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>               // RoutingDirection
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>  // FabricNodeId
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>         // Topology
#include <hostdevcommon/fabric_common.h>                                // chan_id_t, eth_chan_directions

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

// Forward declarations - avoid heavyweight includes in interface header
struct FabricEriscDatamoverConfig;
struct SenderWorkerAdapterSpec;

// ============ Router Location ============

/**
 * RouterLocation
 *
 * Minimal per-router topological info - "where is this router?"
 * Common state (device, program, local_node) is cached at FabricBuilder level.
 * Use aggregate initialization directly.
 */
struct RouterLocation {
    chan_id_t eth_chan;
    FabricNodeId remote_node;
    RoutingDirection direction;  // Use RoutingDirection, convert to eth_chan_directions when needed
    bool is_dispatch_link;
};

// ============ Kernel Creation Context ============

/**
 * KernelCreationContext
 *
 * Cluster-wide coordination info needed for kernel creation.
 * Computed by FabricBuilder (which sees all routers), passed to each router.
 */
struct KernelCreationContext {
    bool is_2D_routing;
    chan_id_t master_router_chan;
    size_t num_local_fabric_routers;
    uint32_t router_channels_mask;
};

// ============ Router Builder Abstract Base Class ============

/**
 * FabricRouterBuilder - Abstract base class for fabric router builders
 *
 * This base class provides:
 * - Common state shared by all router types (local node, location info)
 * - Non-virtual getters for common properties
 * - Pure virtual methods for derived-specific behavior
 *
 * Implementations:
 * - ComputeMeshRouterBuilder: For compute mesh routers (with tensix support)
 * - SwitchMeshRouterBuilder: For switch mesh routers (future, routing-only)
 *
 * Usage:
 *   auto router = FabricRouterBuilder::create(device, program, local_node, location);
 *   router->connect_to_downstream_router_over_noc(*other_router, vc);
 */
class FabricRouterBuilder {
public:
    virtual ~FabricRouterBuilder() = default;

    /**
     * Factory method to create the appropriate router builder.
     * Determines router type (compute mesh vs switch mesh) internally based on fabric context.
     *
     * @param device The device to build on
     * @param program The fabric program
     * @param local_node The local fabric node ID
     * @param location Router location (eth_chan, remote_node, direction, is_dispatch)
     * @return A unique_ptr to the appropriate FabricRouterBuilder implementation
     */
    static std::unique_ptr<FabricRouterBuilder> create(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricNodeId local_node,
        const RouterLocation& location);

    // ============ Connection Methods ============

    /**
     * Connect to downstream router over NOC (one-way connection)
     *
     * @param other The other router builder to connect to
     * @param vc Virtual channel ID
     */
    virtual void connect_to_downstream_router_over_noc(FabricRouterBuilder& other, uint32_t vc) = 0;

    /**
     * Configure connection between this router and a peer router.
     * Handles NOC VC configuration and core placement optimizations for both routers.
     *
     * @param peer The peer router builder
     * @param link_idx The link index within the direction
     * @param num_links Total number of links in this direction
     * @param topology The fabric topology
     * @param is_galaxy Whether this is a galaxy cluster
     */
    virtual void configure_connection(
        FabricRouterBuilder& peer, uint32_t link_idx, uint32_t num_links, Topology topology, bool is_galaxy) = 0;

    /**
     * Configure this router for dispatch link operation.
     * Dispatch links require specific settings (e.g., higher context switching frequency).
     * Each router type decides how (or if) to configure for dispatch.
     */
    virtual void configure_for_dispatch() = 0;

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

    // ============ Common Property Getters (non-virtual) ============

    FabricNodeId get_local_fabric_node_id() const { return local_node_; }
    FabricNodeId get_peer_fabric_node_id() const { return location_.remote_node; }
    chan_id_t get_eth_channel() const { return location_.eth_chan; }
    RoutingDirection get_routing_direction() const { return location_.direction; }
    bool is_dispatch_link() const { return location_.is_dispatch_link; }
    const RouterLocation& get_location() const { return location_; }

    // ============ Derived-Specific Property Getters ============

    virtual eth_chan_directions get_eth_direction() const = 0;
    virtual size_t get_noc_x() const = 0;
    virtual size_t get_noc_y() const = 0;
    virtual size_t get_configured_risc_count() const = 0;

    // ============ Build Methods ============

    /**
     * Compile any ancillary kernels this router needs (e.g., tensix mux).
     * Each router type decides what ancillary kernels it requires.
     *
     * @param program The program to compile kernels into
     */
    virtual void compile_ancillary_kernels(tt::tt_metal::Program& program) = 0;

    /**
     * Create the main router kernel for this router.
     * Each router type controls its own kernel source, compile args, runtime args, etc.
     *
     * @param program The program to create the kernel in
     * @param ctx Cluster-wide coordination info (master router, channel mask, etc.)
     */
    virtual void create_kernel(tt::tt_metal::Program& program, const KernelCreationContext& ctx) = 0;

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

protected:
    // Protected constructor - only derived classes can construct
    FabricRouterBuilder(FabricNodeId local_node, const RouterLocation& location) :
        local_node_(local_node), location_(location) {}

    // Common state shared by all router types
    FabricNodeId local_node_;  // Same for all routers on a device
    RouterLocation location_;  // Per-router topological info (eth_chan, remote_node, direction, is_dispatch)
};

}  // namespace tt::tt_fabric
