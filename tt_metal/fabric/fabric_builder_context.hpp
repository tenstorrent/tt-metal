// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include "erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/channel_trimming_import.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include <vector>
#include <memory>
#include <array>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace tt::tt_fabric {

class FabricContext;

/**
 * PublishedAllocatorState
 *
 * Per-router allocator state published after phase 1 of fabric build.
 * Contains the channel layout (base addresses and buffer counts) for both
 * sender and receiver channels across all VCs. This data allows a peer
 * device to construct a FabricRemoteChannelsAllocator that accurately
 * reflects the actual (possibly asymmetric) allocation on the remote side.
 */
struct PublishedAllocatorState {
    // Per-VC receiver channels (what a peer's sender needs to know)
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        receiver_channels_base_address{};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        receiver_channels_num_buffers{};
    std::array<size_t, builder_config::MAX_NUM_VCS> num_used_receiver_channels_per_vc{};

    // Per-VC sender channels (what a peer's receiver needs to know)
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        sender_channels_base_address{};
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        sender_channels_num_buffers{};
    std::array<size_t, builder_config::MAX_NUM_VCS> num_used_sender_channels_per_vc{};

    // Base class construction data (needed to construct FabricRemoteChannelsAllocator)
    Topology topology = Topology::Linear;
    FabricEriscDatamoverOptions options{};
    std::vector<MemoryRegion> memory_regions;

    // Serialization support for MPI exchange
    std::vector<uint8_t> serialize() const;
    static PublishedAllocatorState deserialize(const std::vector<uint8_t>& data);
};

/**
 * Key type for the published allocator state registry.
 * Identifies a specific router by (chip_id, eth_channel).
 */
using AllocatorStateKey = std::pair<ChipId, chan_id_t>;

struct AllocatorStateKeyHash {
    size_t operator()(const AllocatorStateKey& key) const {
        auto h1 = std::hash<ChipId>{}(key.first);
        auto h2 = std::hash<chan_id_t>{}(key.second);
        return h1 ^ (h2 << 32) ^ (h2 >> 32);
    }
};


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
 * Shared state for the internal barrier in compile_fabric's two-phase build.
 * Constructor and destructor are out-of-line (in fabric_init.cpp) because
 * FabricBuildPhase1Result contains unique_ptr<FabricBuilder> (incomplete here).
 */
struct FabricBuildBarrier {
    const std::vector<tt::tt_metal::IDevice*>& all_devices;
    std::vector<FabricBuildPhase1Result> phase1_results;
    std::atomic<size_t> phase1_count{0};
    std::mutex barrier_mutex;
    std::condition_variable barrier_cv;
    bool phase1_published = false;

    explicit FabricBuildBarrier(const std::vector<tt::tt_metal::IDevice*>& devices);
    ~FabricBuildBarrier();
};

/**
 * FabricBuilderContext
 *
 * Build-time state and config selection for fabric initialization.
 * Owned by FabricContext (lazy initialized on first access).
 *
 * Responsibilities:
 * - Pre-built EDM config templates
 * - Per-device build state (master router channels, initialized router counts)
 * - Tensix config (lazy init after routing tables configured)
 * - Config selection via get_fabric_router_config()
 * - Router address information
 */
class FabricBuilderContext {
public:
    explicit FabricBuilderContext(const FabricContext& fabric_context);
    ~FabricBuilderContext();

    // Non-copyable, non-movable (owned by FabricContext)
    FabricBuilderContext(const FabricBuilderContext&) = delete;
    FabricBuilderContext& operator=(const FabricBuilderContext&) = delete;
    FabricBuilderContext(FabricBuilderContext&&) = delete;
    FabricBuilderContext& operator=(FabricBuilderContext&&) = delete;

    // ============ Access to Parent Context ============
    const FabricContext& get_fabric_context() const { return fabric_context_; }

    // ============ Config Selection ============
    // Returns the appropriate EDM config based on tensix config and direction
    FabricEriscDatamoverConfig& get_fabric_router_config(
        FabricTensixConfig fabric_tensix_config = FabricTensixConfig::DISABLED,
        eth_chan_directions direction = eth_chan_directions::EAST) const;

    // ============ Max Channel Counts ============
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& get_max_sender_channels_per_vc() const {
        return max_sender_channels_per_vc_;
    }
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& get_max_receiver_channels_per_vc() const {
        return max_receiver_channels_per_vc_;
    }

    // ============ Tensix Config ============
    void initialize_tensix_config();
    FabricTensixDatamoverConfig& get_tensix_config() const;
    bool has_tensix_config() const { return tensix_config_ != nullptr; }

    // ============ Per-Device Build State ============
    void set_num_fabric_initialized_routers(ChipId chip_id, size_t num_routers);
    uint32_t get_num_fabric_initialized_routers(ChipId chip_id) const;

    void set_fabric_master_router_chan(ChipId chip_id, chan_id_t chan_id);
    chan_id_t get_fabric_master_router_chan(ChipId chip_id) const;

    // ============ Router Address Info ============
    std::vector<size_t> get_fabric_router_addresses_to_clear() const;
    std::pair<uint32_t, uint32_t> get_fabric_router_sync_address_and_status() const;
    std::optional<std::pair<uint32_t, EDMStatus>> get_fabric_router_ready_address_and_signal() const;
    std::pair<uint32_t, uint32_t> get_fabric_router_termination_address_and_signal() const;

    // ============ Diagnostic Buffer Map ============
    /** Returns the diagnostic buffer locations for all routers.
     *  The layout is identical across all router cores in the fabric. */
    FabricRouterDiagnosticBufferMap get_telemetry_and_metadata_buffer_map() const {
        TT_FATAL(router_config_ != nullptr, "Error, fabric router config is uninitialized");
        return router_config_->get_telemetry_and_metadata_buffer_map();
    }

    // ============ Channel Trimming Overrides ============
    const std::optional<ChannelTrimmingOverrideMap>& get_channel_trimming_overrides() const {
        return channel_trimming_overrides_;
    }

    // ============ Intermesh VC Configuration ============
    const IntermeshVCConfig& get_intermesh_vc_config() const { return intermesh_vc_config_; }
    bool requires_intermesh_vc() const { return intermesh_vc_config_.requires_vc1; }
    bool requires_intermesh_vc_full_mesh() const { return intermesh_vc_config_.requires_vc1_full_mesh; }
    bool requires_intermesh_vc_mesh_pass_through() const { return intermesh_vc_config_.requires_vc1_mesh_pass_through; }

    // ============ Published Allocator State Registry ============
    // Used for peer state exchange between build phases.
    // Publishing happens on the main thread after phase 1 completes — no mutex needed.
    void publish_allocator_state(ChipId chip_id, chan_id_t eth_chan, PublishedAllocatorState state);
    const PublishedAllocatorState& get_published_allocator_state(ChipId chip_id, chan_id_t eth_chan) const;
    bool has_published_allocator_state(ChipId chip_id, chan_id_t eth_chan) const;
    // Return any published allocator state (for cases where any router's state suffices).
    // Falls back to the default allocator state if no per-router states have been published yet
    // (e.g., during early initialization before builders are created).
    const PublishedAllocatorState& get_any_published_allocator_state() const;

    // Return the default allocator state computed from the template config.
    // Available immediately after FabricBuilderContext construction (before builders exist).
    const PublishedAllocatorState& get_default_allocator_state() const { return default_allocator_state_; }

    // ============ Build Barrier ============
    // Thread-safe barrier for coordinating the two-phase fabric build.
    // Created lazily on first call; shared across all threads in one build invocation.
    struct FabricBuildBarrier& get_or_create_build_barrier(
        const std::vector<tt::tt_metal::IDevice*>& all_devices);
    void clear_build_barrier();

private:

    IntermeshVCConfig compute_intermesh_vc_config() const;

    friend class FabricContext;

    const FabricContext& fabric_context_;

    // Channel trimming overrides loaded from profile YAML (if specified)
    std::optional<ChannelTrimmingOverrideMap> channel_trimming_overrides_;

    IntermeshVCConfig intermesh_vc_config_;

    // Computed max channel counts based on actual router types in this fabric
    std::array<std::size_t, builder_config::MAX_NUM_VCS> max_sender_channels_per_vc_{};
    std::array<std::size_t, builder_config::MAX_NUM_VCS> max_receiver_channels_per_vc_{};

    // Pre-built EDM config templates
    std::unique_ptr<FabricEriscDatamoverConfig> router_config_;
    // Router config with mux extension for each direction (E, W, N, S)
    std::array<std::unique_ptr<FabricEriscDatamoverConfig>, eth_chan_directions::COUNT> router_with_mux_config_;

    // Tensix config (lazy init after routing tables configured)
    std::unique_ptr<FabricTensixDatamoverConfig> tensix_config_;

    // Per-device state
    size_t num_devices_ = 0;
    static constexpr chan_id_t UNINITIALIZED_MASTER_ROUTER_CHAN = std::numeric_limits<chan_id_t>::max();
    static constexpr uint32_t UNINITIALIZED_ROUTERS = std::numeric_limits<uint32_t>::max();
    std::vector<chan_id_t> master_router_chans_;
    std::vector<uint32_t> num_initialized_routers_;

    // Helper to create EDM config with given options
    std::unique_ptr<FabricEriscDatamoverConfig> create_edm_config(
        FabricTensixConfig fabric_tensix_config = FabricTensixConfig::DISABLED,
        eth_chan_directions direction = eth_chan_directions::EAST) const;

    // Helper to compute max channel counts for this fabric instance
    void compute_max_channel_counts();

    // Default allocator state computed from template config at construction time.
    // Available before builders exist (used by populate_fabric_connection_info during init).
    PublishedAllocatorState default_allocator_state_;

    // Published allocator state registry — keyed by (chip_id, eth_channel).
    // Published on main thread after phase 1; read on worker threads in phase 2.
    std::unordered_map<AllocatorStateKey, PublishedAllocatorState, AllocatorStateKeyHash> published_allocator_state_;

    // Build barrier for two-phase fabric build coordination.
    std::unique_ptr<FabricBuildBarrier> build_barrier_;
    std::mutex build_barrier_mutex_;
};

}  // namespace tt::tt_fabric
