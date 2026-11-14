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
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
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
 * register_fabric_build_session_factory() API is the intended extension point.
 */
class FabricBuilder {
public:
    FabricBuilder(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricContext& fabric_context,
        CoreType dispatch_core_type);
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
     * Check if any routers were created. Virtual so a builder that does not keep its routers in
     * routers_ (e.g. a plugin that emits kernels from its own graph) is not mistaken for an empty one.
     */
    virtual bool has_routers() const { return !routers_.empty(); }

    /**
     * Get the number of routers created.
     */
    virtual size_t get_num_routers() const { return routers_.size(); }

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
    CoreType dispatch_core_type_;

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

    // The per-chip facts threaded into every per-router build: edge capabilities classified once
    // at discovery, ring predicates bound once in the constructor. Only peer-chip facts are live
    // ControlPlane queries downstream.
    ChipRoutingFacts chip_facts_;

    // Master router channel (first in map)
    chan_id_t master_router_chan_ = 0;
};

// ============ Plugin Registration ============

/**
 * FabricBuildSession
 *
 * Holds a builder's state for one fabric launch. Fabric init creates one session per launch,
 * once and on one thread, before it compiles any device, and destroys it when fabric is torn
 * down. A builder that needs a whole-system view (e.g. lowering a topology graph and allocating
 * every router before emitting any kernel) does that work when its session is created; the
 * per-device builders then only read it. Tearing fabric down and launching it again, with the
 * same or a different config, gets a fresh session.
 *
 * make_device_builder() is called once per device, concurrently from the per-device compile
 * tasks, so it must be safe to call from several threads at once.
 */
class FabricBuildSession {
public:
    virtual ~FabricBuildSession() = default;

    virtual std::unique_ptr<FabricBuilder> make_device_builder(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricContext& fabric_context,
        CoreType dispatch_core_type) const = 0;
};

/**
 * Creates the session for one fabric launch. Receives the launch's FabricContext and must
 * return a non-null session.
 */
using FabricBuildSessionFactory = std::function<std::unique_ptr<FabricBuildSession>(FabricContext&)>;

/**
 * Register a build session factory. When set, every fabric launch builds through the session it
 * creates instead of the default FabricBuilder. Pass nullptr to clear. Asserts if a factory is
 * already registered (clear first). Must be called before fabric init, not during it.
 *
 * Note: This header is internal (not in the public api/ install set). Out-of-tree consumers
 * include it from the tt-metal source tree.
 */
void register_fabric_build_session_factory(FabricBuildSessionFactory factory);

/**
 * Create the build session for one fabric launch: the registered factory's, or a session that
 * builds with the default FabricBuilder when none is registered.
 */
std::unique_ptr<FabricBuildSession> create_fabric_build_session(FabricContext& fabric_context);

/**
 * Per-device factory. Kept for builders that need no launch-wide state: registering a factory
 * registers a session factory whose sessions call it for each device.
 *
 * Usage:
 *   tt::tt_fabric::register_fabric_builder_factory(
 *       [](tt::tt_metal::IDevice* device, tt::tt_metal::Program& program,
 *          tt::tt_fabric::FabricContext& ctx, CoreType dispatch_core_type)
 *           -> std::unique_ptr<tt::tt_fabric::FabricBuilder> {
 *           return std::make_unique<MyCustomFabricBuilder>(device, program, ctx, dispatch_core_type);
 *       });
 */
using FabricBuilderFactory = std::function<std::unique_ptr<FabricBuilder>(
    tt::tt_metal::IDevice*, tt::tt_metal::Program&, FabricContext&, CoreType)>;

/**
 * Register a per-device factory (see FabricBuilderFactory). Same rules as
 * register_fabric_build_session_factory(), which it calls: pass nullptr to clear.
 */
void register_fabric_builder_factory(FabricBuilderFactory factory);

}  // namespace tt::tt_fabric
