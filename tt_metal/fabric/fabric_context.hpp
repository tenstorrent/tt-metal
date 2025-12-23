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
#include <map>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"

namespace tt::tt_fabric {

// Forward declaration
class FabricBuilderContext;


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

    // ============ Packet Specs ============
    size_t get_fabric_packet_header_size_bytes() const { return packet_header_size_bytes_; }
    size_t get_fabric_max_payload_size_bytes() const { return max_payload_size_bytes_; }
    size_t get_fabric_channel_buffer_size_bytes() const { return channel_buffer_size_bytes_; }

    // ============ Dynamic Header Sizing (Phase 1) ============
    uint32_t get_1d_max_hops() const { return max_1d_hops_; }
    uint32_t get_1d_pkt_hdr_extension_words() const {
        return (max_1d_hops_ <= 16) ? 0 : 1;  // Phase 1: 0 or 1 only
    }
    uint32_t get_2d_pkt_hdr_route_buffer_size() const { return routing_2d_buffer_size_; }

    // Get all fabric defines for kernel compilation (used by tt_metal.cpp)
    std::map<std::string, std::string> get_fabric_kernel_defines() const;

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

    // Topology-based sizing
    uint32_t get_max_1d_hops_from_topology() const;
    uint32_t get_max_2d_hops_from_topology() const;
    uint32_t compute_2d_pkt_hdr_route_buffer_size(uint32_t max_hops) const;
    void compute_packet_specifications();

    // Header size helpers (use explicit template instantiation for type safety)
    size_t get_1d_header_size(uint32_t extension_words) const;
    size_t get_2d_header_size(uint32_t route_buffer_size) const;
    size_t get_udm_header_size(uint32_t route_buffer_size) const;

    tt::tt_fabric::FabricConfig fabric_config_{};
    tt::tt_fabric::Topology topology_{};

    bool is_2D_routing_enabled_ = false;
    bool bubble_flow_control_enabled_ = false;
    bool tensix_enabled_ = false;

    std::unordered_map<MeshId, bool> wrap_around_mesh_;

    size_t packet_header_size_bytes_ = 0;
    size_t max_payload_size_bytes_ = 0;
    size_t channel_buffer_size_bytes_ = 0;

    // Phase 1: Dynamic header sizing based on topology
    uint32_t max_1d_hops_ = 32;             // Default: 32 hops (64B header)
    uint32_t max_2d_hops_ = 0;              // Computed from mesh dimensions
    uint32_t routing_2d_buffer_size_ = 32;  // Default: 32B (96B header)

    // Builder context (lazy init on first access)
    mutable std::unique_ptr<FabricBuilderContext> builder_context_;
};

}  // namespace tt::tt_fabric
