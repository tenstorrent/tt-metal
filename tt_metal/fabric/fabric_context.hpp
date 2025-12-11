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

    // ============ Tensix Config Query ============
    // Returns true if tensix is enabled (MUX or UDM mode)
    // Queried from MetalContext at init time
    bool is_tensix_enabled() const { return tensix_enabled_; }

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
    static bool is_2D_topology(tt::tt_fabric::Topology topology);

private:
    std::unordered_map<MeshId, bool> check_for_wrap_around_mesh() const;
    size_t compute_packet_header_size_bytes() const;
    size_t compute_max_payload_size_bytes() const;

    tt::tt_fabric::FabricConfig fabric_config_{};
    tt::tt_fabric::Topology topology_{};

    bool is_2D_routing_enabled_ = false;
    bool bubble_flow_control_enabled_ = false;
    bool tensix_enabled_ = false;

    std::unordered_map<MeshId, bool> wrap_around_mesh_;

    size_t packet_header_size_bytes_ = 0;
    size_t max_payload_size_bytes_ = 0;
    size_t channel_buffer_size_bytes_ = 0;

    // Builder context (lazy init on first access)
    mutable std::unique_ptr<FabricBuilderContext> builder_context_;
};

}  // namespace tt::tt_fabric
