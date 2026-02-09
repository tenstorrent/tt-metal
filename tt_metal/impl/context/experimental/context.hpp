// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "hostdevcommon/common_values.hpp"

namespace tt::tt_metal::experimental {

class ContextDescriptor {
private:
    friend class Context;

    // Dispatch
    int num_cqs_;
    int l1_small_size_;
    int trace_region_size_;
    int worker_l1_size_;
    DispatchCoreConfig dispatch_core_config_;
    bool is_mock_device_;
    std::string mock_cluster_desc_path_;

    // Fabric
    tt::tt_fabric::FabricConfig fabric_config_;
    tt::tt_fabric::FabricReliabilityMode reliability_mode_;
    std::optional<uint8_t> num_routing_planes_;
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config_;
    tt::tt_fabric::FabricUDMMode fabric_udm_mode_;
    tt::tt_fabric::FabricManagerMode fabric_manager_;
    tt::tt_fabric::FabricRouterConfig router_config_;

public:
    explicit ContextDescriptor(
        int num_cqs = 1,
        int l1_small_size = DEFAULT_L1_SMALL_SIZE,
        int trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        int worker_l1_size = DEFAULT_WORKER_L1_SIZE,
        DispatchCoreConfig dispatch_core_config = DispatchCoreConfig{},
        bool is_mock_device = false,
        const std::string& mock_cluster_desc_path = "",
        tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::optional<uint8_t> num_routing_planes = std::nullopt,
        tt::tt_fabric::FabricTensixConfig fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::DISABLED,
        tt::tt_fabric::FabricUDMMode fabric_udm_mode = tt::tt_fabric::FabricUDMMode::DISABLED,
        tt::tt_fabric::FabricManagerMode fabric_manager = tt::tt_fabric::FabricManagerMode::DEFAULT,
        tt::tt_fabric::FabricRouterConfig router_config = tt::tt_fabric::FabricRouterConfig{});

    ~ContextDescriptor() = default;
    ContextDescriptor(const ContextDescriptor&) = delete;
    ContextDescriptor& operator=(const ContextDescriptor&) = delete;
    ContextDescriptor(ContextDescriptor&&) noexcept;
    ContextDescriptor& operator=(ContextDescriptor&&) noexcept;

    int get_num_cqs() const;
    int get_l1_small_size() const;
    int get_trace_region_size() const;
    int get_worker_l1_size() const;
    DispatchCoreConfig get_dispatch_core_config() const;
    bool get_is_mock_device() const;
    std::string get_mock_cluster_desc_path() const;

    tt::tt_fabric::FabricConfig get_fabric_config() const;
    tt::tt_fabric::FabricReliabilityMode get_reliability_mode() const;
    std::optional<uint8_t> get_num_routing_planes() const;
    tt::tt_fabric::FabricTensixConfig get_fabric_tensix_config() const;
    tt::tt_fabric::FabricUDMMode get_fabric_udm_mode() const;
    tt::tt_fabric::FabricManagerMode get_fabric_manager() const;
    tt::tt_fabric::FabricRouterConfig get_router_config() const;
};

}  // namespace tt::tt_metal::experimental
