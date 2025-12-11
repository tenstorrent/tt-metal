// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>  // FabricType
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include "erisc_datamover_builder.hpp"
#include <vector>
#include <limits>
#include <memory>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"

namespace tt::tt_fabric {

// Forward declaration
class FabricBuilderContext;

/**
 * IntermeshVCMode - Defines intermesh VC requirements
 */
enum class IntermeshVCMode : uint8_t {
    DISABLED,                      // No intermesh VC (single mesh or no intermesh connectivity)
    EDGE_ONLY,                     // Intermesh VC on edge nodes only (traffic sinks at mesh boundary)
    FULL_MESH,                     // Intermesh VC throughout mesh (traffic can traverse nodes within mesh)
    FULL_MESH_WITH_PASS_THROUGH    // Intermesh VC with inter-mesh pass-through (e.g., A→B→C routing)
};

/**
 * IntermeshRouterType - Distinguishes types of intermesh routers
 *
 * Different intermesh router types have different channel requirements:
 * - Z_INTERMESH: Vertical device stacking, requires 4 VC1 sender channels (3 mesh + Z)
 * - XY_INTERMESH: Horizontal inter-mesh, requires 3 VC1 sender channels (mesh only)
 */
enum class IntermeshRouterType : uint8_t {
    NONE,          // No intermesh connectivity
    Z_INTERMESH,   // Z routers (vertical device stacking)
    XY_INTERMESH   // XY intermesh routers (horizontal inter-mesh)
};

/**
 * IntermeshVCConfig - System-level intermesh VC configuration
 *
 * Determined during FabricContext initialization based on:
 * - Number of meshes in MeshGraph
 * - Intermesh connectivity topology
 * - Whether traffic traverses within meshes or passes through intermediate meshes
 *
 * Modes:
 * - DISABLED: No intermesh connectivity
 * - EDGE_ONLY: VC1 on edge nodes only, traffic sinks at mesh boundary
 * - FULL_MESH: VC1 throughout mesh, traffic can traverse nodes within target mesh
 * - FULL_MESH_WITH_PASS_THROUGH: VC1 with inter-mesh routing (A→B→C)
 */
struct IntermeshVCConfig {
    IntermeshVCMode mode = IntermeshVCMode::DISABLED;
    IntermeshRouterType router_type = IntermeshRouterType::NONE;  // Type of intermesh router (Z vs XY)
    bool requires_vc1 = false;                      // True if VC1 needed for intermesh
    bool requires_vc1_full_mesh = false;            // True if VC1 needed throughout mesh (not just edges)
    bool requires_vc1_mesh_pass_through = false;    // True if VC1 must support inter-mesh pass-through

    IntermeshVCConfig() = default;

    static IntermeshVCConfig disabled() {
        return IntermeshVCConfig();
    }

    static IntermeshVCConfig edge_only() {
        IntermeshVCConfig config;
        config.mode = IntermeshVCMode::EDGE_ONLY;
        config.requires_vc1 = true;
        return config;
    }

    static IntermeshVCConfig full_mesh() {
        IntermeshVCConfig config;
        config.mode = IntermeshVCMode::FULL_MESH;
        config.requires_vc1 = true;
        config.requires_vc1_full_mesh = true;
        return config;
    }

    static IntermeshVCConfig full_mesh_with_pass_through() {
        IntermeshVCConfig config;
        config.mode = IntermeshVCMode::FULL_MESH_WITH_PASS_THROUGH;
        config.requires_vc1 = true;
        config.requires_vc1_full_mesh = true;
        config.requires_vc1_mesh_pass_through = true;
        return config;
    }
};

/**
 * FabricContext
 *
 * Query interface for fabric topology and configuration.
 * Owns FabricBuilderContext for build-time state (lazy initialized).
 *
 * Split design:
 * - FabricContext: Topology queries, packet specs, mesh type queries (immutable after init)
 * - FabricBuilderContext: EDM configs, per-device state, tensix config (build-time, mutable)
 */
class FabricContext {
public:
    static constexpr auto routing_directions = {
        RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W};

    explicit FabricContext(tt::tt_fabric::FabricConfig fabric_config);
    ~FabricContext();

    // Non-copyable, non-movable
    FabricContext(const FabricContext&) = delete;
    FabricContext& operator=(const FabricContext&) = delete;
    FabricContext(FabricContext&&) = delete;
    FabricContext& operator=(FabricContext&&) = delete;

    // ============ Topology Queries ============
    bool is_wrap_around_mesh(MeshId mesh_id) const;
    tt::tt_fabric::Topology get_fabric_topology() const { return topology_; }
    bool is_2D_routing_enabled() const { return is_2D_routing_enabled_; }
    bool is_bubble_flow_control_enabled() const { return bubble_flow_control_enabled_; }
    bool need_deadlock_avoidance_support(eth_chan_directions direction) const;

    // ============ Mesh Type Queries ============
    // Stub: returns false for now (all meshes are compute meshes)
    // TODO: Implement when switch mesh support lands
    bool is_switch_mesh(MeshId mesh_id) const;

    // ============ Z Router Queries ============
    // Check if a device has a Z router
    // Stub for Phase 3: returns false (will be implemented in Phase 5)
    // TODO(Phase 5): Implement proper Z router detection
    bool has_z_router_on_device(ChipId device_id) const;

    // ============ Tensix Config Query ============
    // Returns true if tensix is enabled (MUX or UDM mode)
    // Queried from MetalContext at init time
    bool is_tensix_enabled() const { return tensix_enabled_; }

    // ============ Intermesh VC Configuration ============
    const IntermeshVCConfig& get_intermesh_vc_config() const { return intermesh_vc_config_; }
    bool requires_intermesh_vc() const { return intermesh_vc_config_.requires_vc1; }
    bool requires_intermesh_vc_full_mesh() const { return intermesh_vc_config_.requires_vc1_full_mesh; }
    bool requires_intermesh_vc_mesh_pass_through() const { return intermesh_vc_config_.requires_vc1_mesh_pass_through; }

    // ============ Packet Specs ============
    size_t get_fabric_packet_header_size_bytes() const { return packet_header_size_bytes_; }
    size_t get_fabric_max_payload_size_bytes() const { return max_payload_size_bytes_; }
    size_t get_fabric_channel_buffer_size_bytes() const { return channel_buffer_size_bytes_; }

    // ============ Builder Context Access ============
    // For build-time operations (config selection, per-device state, router addresses)
    // Lazy initialization on first access
    FabricBuilderContext& get_builder_context();
    const FabricBuilderContext& get_builder_context() const;
    bool has_builder_context() const { return builder_context_ != nullptr; }

    // ============ Static Utilities ============
    static tt::tt_fabric::Topology get_topology_from_config(tt::tt_fabric::FabricConfig fabric_config);

private:
    std::unordered_map<MeshId, bool> check_for_wrap_around_mesh() const;
    size_t compute_packet_header_size_bytes() const;
    size_t compute_max_payload_size_bytes() const;
    IntermeshVCConfig compute_intermesh_vc_config() const;

    tt::tt_fabric::FabricConfig fabric_config_{};
    tt::tt_fabric::Topology topology_{};

    bool is_2D_routing_enabled_ = false;
    bool bubble_flow_control_enabled_ = false;
    bool tensix_enabled_ = false;

    std::unordered_map<MeshId, bool> wrap_around_mesh_;

    size_t packet_header_size_bytes_ = 0;
    size_t max_payload_size_bytes_ = 0;
    size_t channel_buffer_size_bytes_ = 0;

    IntermeshVCConfig intermesh_vc_config_{};

    // Builder context (lazy init on first access)
    mutable std::unique_ptr<FabricBuilderContext> builder_context_;
};

}  // namespace tt::tt_fabric
