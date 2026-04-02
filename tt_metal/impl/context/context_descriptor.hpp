// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <cstdint>
#include <string>

#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include "context/metal_env_accessor.hpp"
#include "context/metal_env_impl.hpp"
#include "context_types.hpp"

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

// Used internally to share runtime configuration
// Provides both lower level (MetalEnv) and runtime configuration (MetalContext)
class ContextDescriptor : public MetalEnvDescriptor {
public:
    // MetalEnv is provided for access to the cluster, hal, and rtoptions
    // It's settings may or may not match the settings (e.g. fabric config) in this descriptor yet
    // as this contextdescriptor is passed around during initialization
    ContextDescriptor(
        MetalEnv* env,
        MetalContext* metal_context,
        int num_cqs = 1,
        size_t l1_small_size = 0,
        size_t trace_region_size = 0,
        size_t worker_l1_size = 0,
        const tt::tt_metal::DispatchCoreConfig& dispatch_core_config = {},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        const std::string& mock_cluster_desc_path = "") :
        MetalEnvDescriptor(
            mock_cluster_desc_path.empty() ? std::optional<std::string>(std::nullopt)
                                           : std::optional<std::string>(mock_cluster_desc_path)),
        env_(env),
        metal_context_(metal_context),
        num_cqs_(num_cqs),
        l1_small_size_(l1_small_size),
        trace_region_size_(trace_region_size),
        worker_l1_size_(worker_l1_size),
        dispatch_core_config_(dispatch_core_config),
        l1_bank_remap_(l1_bank_remap) {}

    ContextDescriptor() = default;

    MetalEnvImpl& env_impl() const {
        TT_FATAL(env_ != nullptr, "Missing MetalEnv for this ContextDescriptor");
        return MetalEnvAccessor(*env_).impl();
    }
    MetalEnv& env() const {
        TT_FATAL(env_ != nullptr, "Missing MetalEnv for this ContextDescriptor");
        return *env_;
    }
    MetalContext& metal_context() const {
        TT_FATAL(metal_context_ != nullptr, "Missing MetalContext for this ContextDescriptor");
        return *metal_context_;
    }

    int num_cqs() const { return num_cqs_; }
    size_t l1_small_size() const { return l1_small_size_; }
    size_t trace_region_size() const { return trace_region_size_; }
    size_t worker_l1_size() const { return worker_l1_size_; }
    const DispatchCoreConfig& dispatch_core_config() const { return dispatch_core_config_; }
    const tt::stl::Span<const std::uint32_t>& l1_bank_remap() const { return l1_bank_remap_; }

    const Hal& hal() const { return env_impl().get_hal(); }
    Cluster& cluster() const { return env_impl().get_cluster(); }
    const llrt::RunTimeOptions& rtoptions() const { return env_impl().get_rtoptions(); }

    tt_fabric::FabricConfig fabric_config() const { return env_impl().get_fabric_config(); }
    tt_fabric::FabricReliabilityMode fabric_reliability_mode() const {
        return env_impl().get_fabric_reliability_mode();
    }
    tt_fabric::FabricTensixConfig fabric_tensix_config() const { return env_impl().get_fabric_tensix_config(); }
    tt_fabric::FabricUDMMode fabric_udm_mode() const { return env_impl().get_fabric_udm_mode(); }
    tt_fabric::FabricManagerMode fabric_manager() const { return env_impl().get_fabric_manager(); }
    const tt::tt_fabric::FabricRouterConfig& fabric_router_config() const {
        return env_impl().get_fabric_router_config();
    }

    // Dependencies
    MetalEnv* env_ = nullptr;
    MetalContext* metal_context_ = nullptr;

    // Dispatch
    int num_cqs_ = 1;
    size_t l1_small_size_ = 0;
    size_t trace_region_size_ = 0;
    size_t worker_l1_size_ = 0;
    DispatchCoreConfig dispatch_core_config_;
    tt::stl::Span<const std::uint32_t> l1_bank_remap_;
};

}  // namespace tt::tt_metal
