// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <cstdint>
#include <string_view>

#include <tt-metalium/dispatch_core_common.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include "context/metal_context.hpp"

namespace tt {
class Cluster;
}  // namespace tt

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal {

class Hal;
class MetalContext;
class DeviceManager;

class ContextDescriptor {
public:
    ContextDescriptor(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode,
        tt::tt_fabric::FabricTensixConfig fabric_tensix_config,
        tt::tt_fabric::FabricUDMMode fabric_udm_mode,
        tt::tt_fabric::FabricManagerMode fabric_manager,
        tt::tt_fabric::FabricRouterConfig router_config = tt::tt_fabric::FabricRouterConfig{},
        int num_cqs = 1,
        int l1_small_size = 0,
        int trace_region_size = 0,
        int worker_l1_size = 0,
        const tt::tt_metal::DispatchCoreConfig& dispatch_core_config = {},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        std::string_view mock_cluster_desc_path = "") :
        num_cqs_(num_cqs),
        l1_small_size_(l1_small_size),
        trace_region_size_(trace_region_size),
        worker_l1_size_(worker_l1_size),
        dispatch_core_config_(dispatch_core_config),
        l1_bank_remap_(l1_bank_remap),
        mock_cluster_desc_path_(mock_cluster_desc_path),
        fabric_config_(fabric_config),
        reliability_mode_(reliability_mode),
        fabric_tensix_config_(fabric_tensix_config),
        fabric_udm_mode_(fabric_udm_mode),
        fabric_manager_(fabric_manager),
        router_config_(router_config) {}

    // Intended for internal use by MetalContext to pass in HAL/Cluster/RuntimeOptions dependencies for init
    ContextDescriptor(
        const Hal& hal,
        Cluster& cluster,
        const llrt::RunTimeOptions& rtoptions,
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode,
        tt::tt_fabric::FabricTensixConfig fabric_tensix_config,
        tt::tt_fabric::FabricUDMMode fabric_udm_mode,
        tt::tt_fabric::FabricManagerMode fabric_manager,
        tt::tt_fabric::FabricRouterConfig router_config = tt::tt_fabric::FabricRouterConfig{},
        int num_cqs = 1,
        int l1_small_size = 0,
        int trace_region_size = 0,
        int worker_l1_size = 0,
        const tt::tt_metal::DispatchCoreConfig& dispatch_core_config = {},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        std::string_view mock_cluster_desc_path = "") :
        hal_(&hal),
        cluster_(&cluster),
        rtoptions_(&rtoptions),
        num_cqs_(num_cqs),
        l1_small_size_(l1_small_size),
        trace_region_size_(trace_region_size),
        worker_l1_size_(worker_l1_size),
        dispatch_core_config_(dispatch_core_config),
        l1_bank_remap_(l1_bank_remap),
        mock_cluster_desc_path_(mock_cluster_desc_path),
        fabric_config_(fabric_config),
        reliability_mode_(reliability_mode),
        fabric_tensix_config_(fabric_tensix_config),
        fabric_udm_mode_(fabric_udm_mode),
        fabric_manager_(fabric_manager),
        router_config_(router_config) {}

    ContextDescriptor() = default;

    const Hal& hal() const { return *hal_; }
    Cluster& cluster() const { return *cluster_; }
    const llrt::RunTimeOptions& rtoptions() const { return *rtoptions_; }

    int num_cqs() const { return num_cqs_; }
    int l1_small_size() const { return l1_small_size_; }
    int trace_region_size() const { return trace_region_size_; }
    int worker_l1_size() const { return worker_l1_size_; }
    const DispatchCoreConfig& dispatch_core_config() const { return dispatch_core_config_; }
    bool is_mock_device() const { return !mock_cluster_desc_path_.empty(); }
    std::string_view mock_cluster_desc_path() const { return mock_cluster_desc_path_; }
    const tt::stl::Span<const std::uint32_t>& l1_bank_remap() const { return l1_bank_remap_; }

    tt::tt_fabric::FabricConfig fabric_config() const { return fabric_config_; }
    tt::tt_fabric::FabricReliabilityMode reliability_mode() const { return reliability_mode_; }
    std::optional<uint8_t> num_routing_planes() const { return num_routing_planes_; }
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config() const { return fabric_tensix_config_; }
    tt::tt_fabric::FabricUDMMode fabric_udm_mode() const { return fabric_udm_mode_; }
    tt::tt_fabric::FabricManagerMode fabric_manager() const { return fabric_manager_; }
    const tt::tt_fabric::FabricRouterConfig& router_config() const { return router_config_; }

private:
    friend class MetalContext;
    friend class DeviceManager;

    // Dependencies
    const Hal* hal_ = nullptr;
    Cluster* cluster_ = nullptr;
    const llrt::RunTimeOptions* rtoptions_ = nullptr;

    // Dispatch
    int num_cqs_ = 1;
    int l1_small_size_ = 0;
    int trace_region_size_ = 0;
    int worker_l1_size_ = 0;
    DispatchCoreConfig dispatch_core_config_;
    tt::stl::Span<const std::uint32_t> l1_bank_remap_;
    std::string_view mock_cluster_desc_path_;

    // Fabric
    tt::tt_fabric::FabricConfig fabric_config_ = tt::tt_fabric::FabricConfig::DISABLED;
    tt::tt_fabric::FabricReliabilityMode reliability_mode_ =
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    std::optional<uint8_t> num_routing_planes_;
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config_ = tt::tt_fabric::FabricTensixConfig::DISABLED;
    tt::tt_fabric::FabricUDMMode fabric_udm_mode_ = tt::tt_fabric::FabricUDMMode::DISABLED;
    tt::tt_fabric::FabricManagerMode fabric_manager_ = tt::tt_fabric::FabricManagerMode::DEFAULT;
    tt::tt_fabric::FabricRouterConfig router_config_;
};

}  // namespace tt::tt_metal
