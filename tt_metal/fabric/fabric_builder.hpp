// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
 * Provides clear build phases and ownership semantics.
 *
 * Lifecycle:
 *   1. Construct with device, program, and contexts
 *   2. Call build phases in order:
 *      discover_channels() -> create_routers() -> connect_routers() ->
 *      compile_ancillary_kernels() -> create_kernels() -> finalize_build_state()
 *   3. FabricBuilder is destroyed, routers are destroyed (they've served their purpose)
 *
 * Usage:
 *   FabricBuilder builder(device, program, fabric_context);
 *   builder.discover_channels();
 *   builder.create_routers();
 *   builder.connect_routers();
 *   builder.compile_ancillary_kernels();
 *   builder.create_kernels();
 *   builder.finalize_build_state();
 */
class FabricBuilder {
public:
    FabricBuilder(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, FabricContext& fabric_context);

    /**
     * Discover active ethernet channels and neighbors for this device.
     * Populates: channels_by_direction_, chip_neighbors_, dispatch_links_
     * Must be called before create_routers().
     */
    void discover_channels();

    /**
     * Create all router builders for this device.
     * Uses cached discovery data from discover_channels().
     */
    void create_routers();

    /**
     * Connect routers using topology-appropriate connections.
     * Uses cached channels_by_direction_ from create_routers().
     */
    void connect_routers();

    /**
     * Compile ancillary kernels (e.g., tensix mux) for each router.
     */
    void compile_ancillary_kernels();

    /**
     * Create the main ERISC router kernels.
     */
    void create_kernels();

    /**
     * Record build state in BuilderContext.
     */
    void finalize_build_state();

    /**
     * Check if any routers were created.
     */
    bool has_routers() const { return !routers_.empty(); }

    /**
     * Get the number of routers created.
     */
    size_t get_num_routers() const { return routers_.size(); }

private:
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
     * Uses member variables: channels_by_direction_, chip_neighbors_, wrap_around_mesh_
     */
    std::vector<RouterConnectionPair> get_router_connection_pairs() const;

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

}  // namespace tt::tt_fabric
