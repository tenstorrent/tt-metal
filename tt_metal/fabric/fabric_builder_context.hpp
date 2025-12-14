// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include "erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include <vector>
#include <memory>
#include <array>
#include <limits>
#include <optional>

namespace tt::tt_fabric {

class FabricContext;


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
    ~FabricBuilderContext() = default;

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

    // ============ Intermesh VC Configuration ============
    const IntermeshVCConfig& get_intermesh_vc_config() const { return intermesh_vc_config_; }
    bool requires_intermesh_vc() const { return intermesh_vc_config_.requires_vc1; }
    bool requires_intermesh_vc_full_mesh() const { return intermesh_vc_config_.requires_vc1_full_mesh; }
    bool requires_intermesh_vc_mesh_pass_through() const { return intermesh_vc_config_.requires_vc1_mesh_pass_through; }

private:
    static IntermeshVCConfig compute_intermesh_vc_config();

    friend class FabricContext;

    const FabricContext& fabric_context_;

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
};

}  // namespace tt::tt_fabric
