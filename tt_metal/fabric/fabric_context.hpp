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

    explicit FabricContext(
        tt::tt_fabric::FabricConfig fabric_config, const FabricRouterConfig& router_config = FabricRouterConfig{});
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

    // ============ Dynamic Header Sizing ============
    uint32_t get_1d_pkt_hdr_extension_words() const {
        TT_FATAL(!is_2D_routing_enabled(), "Cannot query 1D extension words in 2D routing mode");
        return routing_1d_extension_words_;
    }
    uint32_t get_2d_pkt_hdr_route_buffer_size() const {
        TT_FATAL(is_2D_routing_enabled(), "Cannot query 2D route buffer size in 1D routing mode");
        return routing_2d_buffer_size_;
    }

    // Returns empty map if routing mode is undefined
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
    // ============ Packet Sizing Constants (Implementation Details) ============

    // Implementation limits (memory map & buffer size constraints)
    struct Limits {
        // 1D: Max hops supported by current L1 memory map
        //     ROUTING_PATH_SIZE_1D = 256 bytes / 8 bytes per entry = 32 chips max
        static constexpr uint32_t MAX_1D_HOPS = 32;

        // 2D: Max route buffer size (optimized to 35, fits in 96B header)
        //     Each byte in route buffer encodes 1 hop, so MAX_2D_HOPS = MAX_2D_ROUTE_BUFFER_SIZE
        //     35-byte buffer fits in 96B header (61B base + 35B buffer = 96B, zero padding waste)
        static constexpr uint32_t MAX_2D_ROUTE_BUFFER_SIZE = 35;
        static constexpr uint32_t MAX_2D_HOPS = MAX_2D_ROUTE_BUFFER_SIZE;
    };

    // 1D routing: hops per routing word (base word supports 16 hops)
    static constexpr uint32_t ROUTING_1D_HOPS_PER_WORD = 16;

    // 2D routing: buffer tiers optimized to maximize capacity per header size
    // Header base = 61B, aligned to 16B boundaries
    // Strategy: One tier per header size (max capacity) to avoid bloat
    //   80B:  61+19=80  (max capacity)
    //   96B:  61+35=96  (max capacity)
    // Fabric context automatically selects smallest header that fits required hop count
    struct Routing2DBufferTier {
        uint32_t max_hops;
        uint32_t buffer_size;
    };
    static constexpr Routing2DBufferTier ROUTING_2D_BUFFER_TIERS[] = {
        // NOTE: 80B header size de-stabilized some Mesh benchmarks for 8X4 mesh, so disabling for now
        //{19, 19},  // 80B header - max capacity
        {35, 35}  // 96B header - max capacity
    };

    // ============ Private Implementation ============
    std::unordered_map<MeshId, bool> check_for_wrap_around_mesh() const;
    size_t compute_packet_header_size_bytes() const;
    size_t compute_max_payload_size_bytes() const;
    size_t validate_and_apply_packet_size(size_t requested_size) const;

    // Compute and validate routing mode from topology
    void compute_routing_mode();

    // Topology-based sizing
    uint32_t get_max_1d_hops_from_topology() const;
    uint32_t get_max_2d_hops_from_topology() const;
    uint32_t compute_1d_pkt_hdr_extension_words(uint32_t max_hops) const;
    uint32_t compute_2d_pkt_hdr_route_buffer_size(uint32_t max_hops) const;
    void compute_packet_specifications();

    // Header size helpers (use explicit template instantiation for type safety)
    size_t get_1d_header_size(uint32_t extension_words) const;
    size_t get_2d_header_size(uint32_t route_buffer_size) const;
    size_t get_udm_header_size(uint32_t route_buffer_size) const;

    tt::tt_fabric::FabricConfig fabric_config_{};
    tt::tt_fabric::Topology topology_{};
    FabricRouterConfig router_config_{};

    bool is_2D_routing_enabled_ = false;
    bool bubble_flow_control_enabled_ = false;
    bool tensix_enabled_ = false;

    std::unordered_map<MeshId, bool> wrap_around_mesh_;

    size_t packet_header_size_bytes_ = 0;
    size_t max_payload_size_bytes_ = 0;
    size_t channel_buffer_size_bytes_ = 0;

    // Dynamic header sizing (set by compute_packet_specifications based on mode)
    uint32_t max_1d_hops_ = 0;                 // Valid only in 1D mode
    uint32_t routing_1d_extension_words_ = 0;  // Valid only in 1D mode
    uint32_t max_2d_hops_ = 0;                 // Valid only in 2D mode
    uint32_t routing_2d_buffer_size_ = 0;      // Valid only in 2D mode

    uint16_t routing_mode_ = 0;  // ROUTING_MODE_UNDEFINED by default

    // Builder context (lazy init on first access)
    mutable std::unique_ptr<FabricBuilderContext> builder_context_;
};

}  // namespace tt::tt_fabric
