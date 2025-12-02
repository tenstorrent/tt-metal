// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>
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
 * RouterConnectionPair
 *
 * Represents a pair of routers to be connected, with link information.
 * Used by get_router_connection_pairs() to separate topology logic from connection logic.
 */
struct RouterConnectionPair {
    chan_id_t chan1;
    chan_id_t chan2;
    uint32_t link_idx;
    uint32_t num_links;
};

/**
 * Get pairs of routers to connect based on topology
 *
 * Pure function that encapsulates all topology logic (2D vs 1D, direction pairs).
 *
 * @param channels_by_direction Map of direction to list of eth channels
 * @param fabric_context The fabric context (for topology info)
 * @param chip_neighbors Map of directions to neighbor info
 * @param wrap_around_mesh Whether the mesh wraps around
 * @return Vector of RouterConnectionPair describing which routers to connect
 */
std::vector<RouterConnectionPair> get_router_connection_pairs(
    const std::unordered_map<RoutingDirection, std::vector<chan_id_t>>& channels_by_direction,
    const FabricContext& fabric_context,
    const std::unordered_map<RoutingDirection, FabricNodeId>& chip_neighbors,
    bool wrap_around_mesh);

/**
 * FabricBuilder
 *
 * Transient orchestrator class that owns router builders during the build process.
 * Provides clear build phases and ownership semantics.
 *
 * Lifecycle:
 *   1. Construct with device, program, and contexts
 *   2. Call build phases in order: create_routers(), connect_routers(), compile_ancillary_kernels(), create_kernels()
 *   3. Call finalize_build_state() to record state
 *   4. FabricBuilder is destroyed, routers are destroyed (they've served their purpose)
 *
 * Usage:
 *   FabricBuilder builder(device, program, fabric_context);
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
     * Create all router builders for this device.
     * Discovers active channels and creates appropriate builders.
     * Caches channels_by_direction_ for use in connect_routers().
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
    tt::tt_metal::IDevice* device_;
    tt::tt_metal::Program& program_;
    FabricContext& fabric_context_;
    FabricBuilderContext& builder_context_;

    // Fabric node ID for this device (derived from device_->id())
    FabricNodeId local_node_;

    // Owned routers - destroyed when FabricBuilder is destroyed
    std::unordered_map<chan_id_t, std::unique_ptr<FabricRouterBuilder>> routers_;

    // Cached during create_routers(), used by connect_routers()
    std::unordered_map<RoutingDirection, std::vector<chan_id_t>> channels_by_direction_;

    // Cached neighbor info
    std::unordered_map<RoutingDirection, FabricNodeId> chip_neighbors_;
    bool wrap_around_mesh_ = false;

    // Master router channel (first in map)
    chan_id_t master_router_chan_ = 0;
};

}  // namespace tt::tt_fabric
