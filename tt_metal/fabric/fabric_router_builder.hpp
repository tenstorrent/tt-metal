// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>               // RoutingDirection
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>  // FabricNodeId
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>         // Topology
#include <hostdevcommon/fabric_common.h>                                // chan_id_t

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

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
 *   router->configure_connection(*other_router, link_idx, num_links, topology, is_galaxy);
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
     * Configure connection between this router and a peer router.
     * Handles bidirectional VC connections, NOC VC configuration, and core placement optimizations.
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

    // ============ Common Property Getters (non-virtual) ============

    FabricNodeId get_local_fabric_node_id() const { return local_node_; }
    FabricNodeId get_peer_fabric_node_id() const { return location_.remote_node; }
    chan_id_t get_eth_channel() const { return location_.eth_chan; }
    RoutingDirection get_routing_direction() const { return location_.direction; }
    bool is_dispatch_link() const { return location_.is_dispatch_link; }
    const RouterLocation& get_location() const { return location_; }

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

protected:
    // Protected constructor - only derived classes can construct
    FabricRouterBuilder(FabricNodeId local_node, const RouterLocation& location) :
        local_node_(local_node), location_(location) {}

    // Common state shared by all router types
    FabricNodeId local_node_;  // Same for all routers on a device
    RouterLocation location_;  // Per-router topological info (eth_chan, remote_node, direction, is_dispatch)
};

}  // namespace tt::tt_fabric
