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

private:
    friend class FabricContext;

    const FabricContext& fabric_context_;

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
};

}  // namespace tt::tt_fabric
