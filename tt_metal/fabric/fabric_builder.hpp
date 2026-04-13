// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "hostdevcommon/fabric_common.h"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

class FabricContext;
class FabricBuilderContext;

/**
 * FabricBuilder
 *
 * Transient orchestrator class that owns router builders during the build process.
 *
 * Lifecycle:
 *   1. Construct with device, program, and contexts
 *   2. Call build phases in order:
 *      discover_channels() -> create_routers() -> connect_routers() ->
 *      compile_ancillary_kernels() -> create_kernels()
 *   3. FabricBuilder is destroyed, router builders are destroyed
 *
 * ABI note: This class has virtual methods (added for plugin subclassing support).
 * This means it carries a vptr and its layout differs from a non-virtual version.
 * FabricBuilder is only constructed via fabric_init.cpp (never embedded in other
 * structs or passed across shared library boundaries by value), so the ABI impact
 * is limited to callers that previously constructed FabricBuilder directly. The
 * register_fabric_builder_factory() API is the intended extension point.
 */
class FabricBuilder {
public:
    FabricBuilder(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, FabricContext& fabric_context);
    virtual ~FabricBuilder() = default;

    /**
     * Discover active ethernet channels and neighbors for this device.
     * Must be called before create_routers().
     */
    virtual void discover_channels();

    /**
     * Create all router builders for this device.
     * Override to dispatch to custom FabricRouterBuilder subclasses.
     */
    virtual void create_routers();

    /**
     * Connect routers using topology-appropriate connections.
     */
    virtual void connect_routers();

    /**
     * Compile ancillary kernels (e.g., tensix mux) for each router.
     */
    virtual void compile_ancillary_kernels();

    /**
     * Create the main ERISC router kernels.
     */
    virtual void create_kernels();

    /**
     * Check if any routers were created.
     */
    bool has_routers() const { return !routers_.empty(); }

    /**
     * Get the number of routers created.
     */
    size_t get_num_routers() const { return routers_.size(); }

protected:
    /**
     * RouterConnectionPair - Internal struct for router connection info
     */
    struct RouterConnectionPair {
        chan_id_t chan1;
        chan_id_t chan2;
        uint32_t link_idx;
        uint32_t num_links;
    };

    /**
     * Get pairs of routers to connect based on topology.
     */
    std::vector<RouterConnectionPair> get_router_connection_pairs() const;

    /**
     * Configure local connections between routers on this device.
     *
     * Generic connection establishment: iterates through all routers and
     * establishes connections to local targets based on their connection mappings.
     * Each router's mapping determines which local routers to connect to.
     *
     * Handles variable router configurations (e.g., 2-4 mesh routers on edge devices).
     *
     * Called by connect_routers() after inter-device connections are established.
     */
    void configure_local_connections(
        const std::map<FabricRouterBuilder*, std::map<RoutingDirection, FabricRouterBuilder*>>&
            routers_by_direction_map);

    /**
     * Compile kernels for directions that have no router/eth channel.
     *
     * In UDM mode, edge devices (e.g., corner of a mesh) don't have neighbors
     * in all 4 directions, but still need mux cores for inter-mux forwarding.
     * These kernels are not associated with any router.
     *
     * No-op for non-UDM modes or devices with neighbors in all directions.
     */
    void compile_kernels_for_missing_directions();

    tt::tt_metal::IDevice* device_;
    tt::tt_metal::Program& program_;
    FabricContext& fabric_context_;
    FabricBuilderContext& builder_context_;

    // Fabric node ID for this device (derived from device_->id())
    FabricNodeId local_node_;

    // Topology info (initialized in constructor)
    bool wrap_around_mesh_ = false;
    bool device_has_dispatch_tunnel_ = false;

    // Owned routers - destroyed when FabricBuilder is destroyed
    std::unordered_map<chan_id_t, std::unique_ptr<FabricRouterBuilder>> routers_;

    // Cached during discover_channels(), used by create_routers() and connect_routers()
    std::unordered_map<RoutingDirection, std::vector<chan_id_t>> channels_by_direction_;
    std::unordered_map<RoutingDirection, FabricNodeId> chip_neighbors_;
    std::unordered_set<chan_id_t> dispatch_links_;

    // Master router channel (first in map)
    chan_id_t master_router_chan_ = 0;
};

// ============ Plugin Factory Registration ============

/**
 * Factory function type for creating custom FabricBuilder implementations.
 *
 * External code (e.g., custom fabric builder plugins) can register a factory function
 * that will be called instead of constructing the default FabricBuilder. This enables
 * out-of-tree builder implementations without compile-time coupling.
 *
 * The factory receives the same arguments as the FabricBuilder constructor and must
 * return a non-null, fully-functional FabricBuilder (or subclass) that implements all
 * build phases: discover_channels, create_routers, connect_routers,
 * compile_ancillary_kernels, create_kernels.
 *
 * Thread safety: Registration must happen before any call to
 * create_and_compile_tt_fabric_program(). This is a program-startup operation.
 * Concurrent registration or registration during fabric init is a programming error.
 * Double-registration without clearing (nullptr) first will assert.
 *
 * Note: This header is currently internal (not in the public api/ install set).
 * Out-of-tree consumers must include it from the tt-metal source tree. A public
 * forwarding header under api/tt-metalium/experimental/fabric/ may be added in the
 * future if this API stabilizes.
 *
 * Usage:
 *   // In plugin initialization (before fabric init):
 *   tt::tt_fabric::register_fabric_builder_factory(
 *       [](tt::tt_metal::IDevice* device, tt::tt_metal::Program& program,
 *          tt::tt_fabric::FabricContext& ctx) -> std::unique_ptr<tt::tt_fabric::FabricBuilder> {
 *           return std::make_unique<MyCustomFabricBuilder>(device, program, ctx);
 *       });
 */
using FabricBuilderFactory = std::function<std::unique_ptr<FabricBuilder>(
    tt::tt_metal::IDevice*, tt::tt_metal::Program&, FabricContext&)>;

/**
 * Register a custom FabricBuilder factory. When set, fabric initialization will use this
 * factory instead of constructing the default FabricBuilder. Pass nullptr to clear.
 * Asserts if a factory is already registered (clear first with nullptr before re-registering).
 * Must be called before any fabric init. The factory must return a non-null pointer.
 */
void register_fabric_builder_factory(FabricBuilderFactory factory);

/**
 * Returns the currently registered factory, or nullptr if none is registered.
 */
const FabricBuilderFactory& get_fabric_builder_factory();

}  // namespace tt::tt_fabric
