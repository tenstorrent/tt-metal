// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>  // FabricType
#include <umd/device/types/cluster_descriptor_types.hpp>   // ChipId
#include <tt_stl/assert.hpp>
#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <map>
#include <type_traits>
#include <utility>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

// Forward declarations
class ControlPlane;
class FabricBuilderContext;

// A packet header FabricContext can choose.
struct PacketHeaderConfig {
    bool is_2d = false;
    bool udm = false;
    uint32_t extension_words = 0;    // 1D only
    uint32_t route_buffer_size = 0;  // 2D only
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
        RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W, RoutingDirection::Z};

    // The references are held for the lifetime of this object. They are owned by the same MetalEnv
    // as control_plane, which owns this object. Constructed while control_plane is still being
    // built, so the constructor must not look up MetalContext.
    FabricContext(
        ControlPlane& control_plane,
        const tt_metal::Hal& hal,
        const tt::Cluster& cluster,
        const llrt::RunTimeOptions& rtoptions,
        tt::tt_fabric::FabricConfig fabric_config,
        const FabricRouterConfig& router_config = FabricRouterConfig{});
    ~FabricContext();

    // Non-copyable, non-movable
    FabricContext(const FabricContext&) = delete;
    FabricContext& operator=(const FabricContext&) = delete;
    FabricContext(FabricContext&&) = delete;
    FabricContext& operator=(FabricContext&&) = delete;

    // ============ Dependencies ============
    ControlPlane& get_control_plane() { return control_plane_; }
    const ControlPlane& get_control_plane() const { return control_plane_; }
    const tt_metal::Hal& get_hal() const { return hal_; }
    const tt::Cluster& get_cluster() const { return cluster_; }
    const llrt::RunTimeOptions& get_rtoptions() const { return rtoptions_; }

    // ============ Topology Queries ============
    bool is_wrap_around_mesh(MeshId mesh_id) const;
    tt::tt_fabric::Topology get_fabric_topology() const { return topology_; }
    bool is_2D_routing_enabled() const { return is_2D_routing_enabled_; }
    bool is_bubble_flow_control_enabled() const { return bubble_flow_control_enabled_; }
    // Whether a router on this axis needs protected (bubble) flow control compiled in.
    //
    // Express meshes derive it from whether the axis actually carries protected rings, since express
    // chords can close a ring on an axis that is not a torus. Other meshes keep the topology-derived
    // answer unchanged.
    bool need_deadlock_avoidance_support(
        const ControlPlane& control_plane, const FabricNodeId& fabric_node_id, eth_chan_directions direction) const;
    bool is_ubb_galaxy() const { return is_ubb_galaxy_; }

    // ============ Mesh Type Queries ============
    // Stub: returns false for now (all meshes are compute meshes)
    // TODO: Implement when switch mesh support lands
    bool is_switch_mesh(MeshId mesh_id) const;

    // ============ Z Router Queries ============
    // Do not add a raw "does this node/mesh have a Z edge" query here. Scanning connectivity for
    // RoutingDirection::Z answers a weaker question than any of its callers actually ask:
    //   - selecting the 2D action-map format
    //                                    -> is_2D_routing_enabled(), independent of whether the mesh
    //                                       has a Z edge
    //   - sizing the VC0 downstream fan  -> get_vc0_downstream_edm_count(is_2D, express_routing_enabled)
    //   - intermesh / express edge role  -> classify_fabric_edge() and ZPortRole, per edge

    // ============ Tensix Config Query ============
    // Read from the ControlPlane at construction
    tt::tt_fabric::FabricTensixConfig get_fabric_tensix_config() const { return fabric_tensix_config_; }
    // Returns true if tensix is enabled (MUX or UDM mode)
    bool is_tensix_enabled() const { return fabric_tensix_config_ != tt::tt_fabric::FabricTensixConfig::DISABLED; }

    // ============ Packet Specs ============
    // The run's packet header, which visit_packet_header_type maps to its type.
    const PacketHeaderConfig& get_packet_header_config() const { return packet_header_config_; }
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

    // Configuration-wide kernel defines. Exact 2D mesh shape is runtime routing metadata; express
    // capacity is enabled for all local kernels when any locally bound mesh uses express routing.
    // Returns empty map if routing mode is undefined.
    std::map<std::string, std::string> get_fabric_kernel_defines(const ControlPlane& control_plane) const;

    // ============ Builder Context Access ============
    // For build-time operations (config selection, per-device state, router addresses)
    // Lazy initialization on first access
    FabricBuilderContext& get_builder_context();
    const FabricBuilderContext& get_builder_context() const;
    bool has_builder_context() const { return builder_context_ != nullptr; }

    // ============ Static Utilities ============
    static tt::tt_fabric::Topology get_topology_from_config(tt::tt_fabric::FabricConfig fabric_config);

    // ============ Packet Header Sizing ============
    // The header sizes FabricContext can choose. visit_packet_header_type (below) maps each to its type.

    // Implementation limits (memory map & buffer size constraints)
    struct Limits {
        // 1D: Max hops supported by L1 memory map
        //     ROUTING_PATH_SIZE_1D = 1024 bytes / 16 bytes per entry = 64 chips max (63 hops)
        static constexpr uint32_t MAX_1D_HOPS = 63;

        // Maximum supported 2D shape is 64x4 in either orientation, so packets need 68 action bytes.
        static constexpr uint32_t MAX_2D_ACTION_MAP_BYTES =
            Routing2DCodec::MAX_AXIS_SIZE + Routing2DCodec::MAX_ORTHOGONAL_AXIS_SIZE;
    };

    // 1D routing: hops per routing word (base word supports 16 hops)
    static constexpr uint32_t ROUTING_1D_HOPS_PER_WORD = RoutingFieldsConstants::LowLatency::BASE_HOPS;

    // 2D routing: buffer tiers optimized to maximize capacity per header size
    // Header base = 60 B, aligned to 16 B boundaries
    // Strategy: One tier per header size (max capacity) to avoid bloat
    //   96 B: 60 + 36, 112 B: 60 + 52, 128 B: 60 + 68
    // FabricContext selects the smallest header that holds the topology's action map.
    struct Routing2DHeaderTier {
        uint32_t max_action_map_bytes;
        uint32_t route_buffer_size;
    };
    static constexpr Routing2DHeaderTier ROUTING_2D_HEADER_TIERS[] = {
        // NOTE: 80B header size de-stabilized some Mesh benchmarks for 8X4 mesh, so disabling for now
        //{20, 20},  // 80B header - max capacity (60+20=80)
        {36, 36},                                                           // 96B header - max capacity (60+36=96)
        {52, 52},                                                           // 112B header - max capacity (60+52=112)
        {Limits::MAX_2D_ACTION_MAP_BYTES, Limits::MAX_2D_ACTION_MAP_BYTES}  // 128B header
    };

    // Precondition: max_hops validated by compute_packet_specifications()
    static constexpr uint32_t compute_1d_pkt_hdr_extension_words(uint32_t max_hops) {
        // Base routing word supports 16 hops; extension words add 16 hops each
        // ExtensionWords=0: 1-16 hops (48B header)
        // ExtensionWords=1: 17-32 hops (64B header)
        return (max_hops - 1) / ROUTING_1D_HOPS_PER_WORD;
    }

    // Precondition: required_route_bytes is validated by compute_packet_specifications().
    static constexpr uint32_t compute_2d_pkt_hdr_route_buffer_size(uint32_t required_route_bytes) {
        for (const auto& tier : ROUTING_2D_HEADER_TIERS) {
            if (required_route_bytes <= tier.max_action_map_bytes) {
                return tier.route_buffer_size;
            }
        }
        return Limits::MAX_2D_ACTION_MAP_BYTES;
    }

private:
    // ============ Private Implementation ============
    std::unordered_map<MeshId, bool> check_for_wrap_around_mesh(const ControlPlane& control_plane) const;
    PacketHeaderConfig compute_packet_header_config(const ControlPlane& control_plane) const;
    size_t compute_max_payload_size_bytes(const tt_metal::Hal& hal, tt::ARCH arch) const;
    size_t validate_and_apply_packet_size(const tt_metal::Hal& hal, tt::ARCH arch, size_t requested_size) const;

    // Compute and validate routing mode from topology
    void compute_routing_mode();

    // Topology-based sizing
    uint32_t get_max_1d_hops_from_topology(const ControlPlane& control_plane) const;
    uint32_t get_max_2d_action_map_bytes_from_topology(const ControlPlane& control_plane) const;
    void compute_packet_specifications(const ControlPlane& control_plane, const tt_metal::Hal& hal, tt::ARCH arch);

    ControlPlane& control_plane_;
    const tt_metal::Hal& hal_;
    const tt::Cluster& cluster_;
    const llrt::RunTimeOptions& rtoptions_;

    tt::tt_fabric::FabricConfig fabric_config_{};
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config_ = tt::tt_fabric::FabricTensixConfig::DISABLED;
    tt::tt_fabric::Topology topology_{};
    FabricRouterConfig router_config_{};

    bool is_2D_routing_enabled_ = false;
    bool bubble_flow_control_enabled_ = false;
    bool is_ubb_galaxy_ = false;

    std::unordered_map<MeshId, bool> wrap_around_mesh_;

    PacketHeaderConfig packet_header_config_;
    size_t packet_header_size_bytes_ = 0;
    size_t max_payload_size_bytes_ = 0;
    size_t channel_buffer_size_bytes_ = 0;

    // Dynamic header sizing (set by compute_packet_specifications based on mode)
    uint32_t max_1d_hops_ = 0;                 // Valid only in 1D mode
    uint32_t routing_1d_extension_words_ = 0;  // Valid only in 1D mode
    uint32_t routing_2d_buffer_size_ = 0;      // Valid only in 2D mode

    uint16_t routing_mode_ = 0;  // ROUTING_MODE_UNDEFINED by default

    // Builder context (lazy init on first access)
    mutable std::unique_ptr<FabricBuilderContext> builder_context_;
};

namespace packet_header_detail {

constexpr uint32_t NUM_1D_CONFIGS =
    FabricContext::compute_1d_pkt_hdr_extension_words(FabricContext::Limits::MAX_1D_HOPS) + 1;
constexpr uint32_t NUM_2D_TIERS = std::size(FabricContext::ROUTING_2D_HEADER_TIERS);

constexpr std::array<PacketHeaderConfig, NUM_1D_CONFIGS + 2 * NUM_2D_TIERS> make_packet_header_configs() {
    std::array<PacketHeaderConfig, NUM_1D_CONFIGS + 2 * NUM_2D_TIERS> configs{};
    size_t i = 0;
    for (uint32_t extension_words = 0; extension_words < NUM_1D_CONFIGS; extension_words++) {
        configs[i++] = {.is_2d = false, .udm = false, .extension_words = extension_words};
    }
    for (bool udm : {false, true}) {
        for (const auto& tier : FabricContext::ROUTING_2D_HEADER_TIERS) {
            configs[i++] = {.is_2d = true, .udm = udm, .route_buffer_size = tier.route_buffer_size};
        }
    }
    return configs;
}

// The 2D chooser returns a tier's route buffer size for every action map up to the limit only if the tiers
// grow strictly, each buffer holds its tier's action map, and the last tier reaches the limit.
constexpr bool routing_2d_header_tiers_are_valid() {
    const auto& tiers = FabricContext::ROUTING_2D_HEADER_TIERS;
    for (size_t i = 0; i < std::size(tiers); i++) {
        if (tiers[i].route_buffer_size < tiers[i].max_action_map_bytes) {
            return false;
        }
        if (i > 0 && (tiers[i].max_action_map_bytes <= tiers[i - 1].max_action_map_bytes ||
                      tiers[i].route_buffer_size <= tiers[i - 1].route_buffer_size)) {
            return false;
        }
    }
    return tiers[std::size(tiers) - 1].max_action_map_bytes == FabricContext::Limits::MAX_2D_ACTION_MAP_BYTES;
}
static_assert(routing_2d_header_tiers_are_valid(), "ROUTING_2D_HEADER_TIERS must grow and cover the 2D limit");

// Every input the choosers accept picks a header in PACKET_HEADER_CONFIGS with room for it.
constexpr bool choosers_pick_listed_headers() {
    for (uint32_t hops = 1; hops <= FabricContext::Limits::MAX_1D_HOPS; hops++) {
        const uint32_t extension_words = FabricContext::compute_1d_pkt_hdr_extension_words(hops);
        if (extension_words >= NUM_1D_CONFIGS ||
            (extension_words + 1) * FabricContext::ROUTING_1D_HOPS_PER_WORD < hops) {
            return false;
        }
    }
    for (uint32_t bytes = 1; bytes <= FabricContext::Limits::MAX_2D_ACTION_MAP_BYTES; bytes++) {
        const uint32_t route_buffer_size = FabricContext::compute_2d_pkt_hdr_route_buffer_size(bytes);
        bool listed = false;
        for (const auto& tier : FabricContext::ROUTING_2D_HEADER_TIERS) {
            listed = listed || tier.route_buffer_size == route_buffer_size;
        }
        if (!listed || route_buffer_size < bytes) {
            return false;
        }
    }
    return true;
}
static_assert(choosers_pick_listed_headers(), "A packet header chooser picks a header that is not listed or too small");

}  // namespace packet_header_detail

// Every packet header FabricContext can choose: 1D with each extension word count up to what MAX_1D_HOPS needs,
// then each 2D tier without and with UDM.
inline constexpr auto PACKET_HEADER_CONFIGS = packet_header_detail::make_packet_header_configs();

namespace packet_header_detail {

template <size_t I>
constexpr auto packet_header_type_at() {
    constexpr PacketHeaderConfig config = PACKET_HEADER_CONFIGS[I];
    if constexpr (!config.is_2d) {
        return std::type_identity<LowLatencyPacketHeaderT<config.extension_words>>{};
    } else if constexpr (config.udm) {
        return std::type_identity<UDMHybridMeshPacketHeaderT<config.route_buffer_size>>{};
    } else {
        return std::type_identity<HybridMeshPacketHeaderT<config.route_buffer_size>>{};
    }
}

inline size_t find_packet_header_config(const PacketHeaderConfig& config) {
    TT_FATAL(!config.udm || config.is_2d, "UDM mode only supports 2D routing");
    for (size_t i = 0; i < PACKET_HEADER_CONFIGS.size(); i++) {
        const auto& candidate = PACKET_HEADER_CONFIGS[i];
        if (candidate.is_2d == config.is_2d && candidate.udm == config.udm &&
            (config.is_2d ? candidate.route_buffer_size == config.route_buffer_size
                          : candidate.extension_words == config.extension_words)) {
            return i;
        }
    }
    if (config.is_2d) {
        TT_THROW("Unsupported 2D route buffer size: {}", config.route_buffer_size);
    }
    TT_THROW("Unsupported extension words: {}", config.extension_words);
}

template <typename F, size_t... I>
decltype(auto) visit(const PacketHeaderConfig& config, F& f, std::index_sequence<I...>) {
    using Result = std::invoke_result_t<F&, decltype(packet_header_type_at<0>())>;
    static_assert(
        (std::is_same_v<Result, std::invoke_result_t<F&, decltype(packet_header_type_at<I>())>> && ...),
        "visit_packet_header_type: f must return the same type for every packet header");
    static constexpr Result (*cases[])(F&) = {[](F& g) -> Result { return g(packet_header_type_at<I>()); }...};
    return cases[find_packet_header_config(config)](f);
}

}  // namespace packet_header_detail

// Calls f(std::type_identity<H>{}) with the packet header type H for config and returns its result. f must return
// the same type for every header; void is allowed. Throws for a configuration FabricContext can't choose.
template <typename F>
decltype(auto) visit_packet_header_type(const PacketHeaderConfig& config, F&& f) {
    return packet_header_detail::visit(config, f, std::make_index_sequence<PACKET_HEADER_CONFIGS.size()>{});
}

}  // namespace tt::tt_fabric
