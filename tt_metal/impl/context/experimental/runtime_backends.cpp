// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/context/experimental/runtime_backends.hpp"
#include "impl/context/experimental/runtime.hpp"
#include <tt-logger/tt-logger.hpp>
#include <llrt/rtoptions.hpp>
#include <tt-metalium/hal.hpp>
#include <llrt/tt_cluster.hpp>
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"

#include <unordered_map>

namespace tt::tt_metal::experimental {

namespace {

// Returns a default mock cluster descriptor filename for the detected hardware architecture.
// Falls back to single-chip configuration.
std::string get_default_mock_cluster_desc() {
    tt::ARCH arch = tt::tt_metal::get_physical_architecture();
    TT_FATAL(arch != tt::ARCH::Invalid, "No TT hardware detected - cannot auto-detect architecture for mock mode");

    static const std::unordered_map<tt::ARCH, std::string> default_descriptors = {
        {tt::ARCH::WORMHOLE_B0, "wormhole_N150.yaml"},
        {tt::ARCH::BLACKHOLE, "blackhole_P150.yaml"},
    };

    auto it = default_descriptors.find(arch);
    TT_FATAL(it != default_descriptors.end(), "No default mock cluster descriptor for arch {}", static_cast<int>(arch));
    return it->second;
}

}  // namespace

SiliconRuntimeBackend::SiliconRuntimeBackend(std::shared_ptr<MetaliumObject> metalium_object) :
    metalium_object_(std::move(metalium_object)) {}

void SiliconRuntimeBackend::initialize(const ContextDescriptor& descriptor) {
    log_info(tt::LogMetal, "Initializing Silicon Runtime Backend Num CQ: {}", descriptor.get_num_cqs());
    if (!metalium_object_) {
        TT_THROW("MetaliumObject must be provided to SiliconRuntimeBackend");
    }
}

void SiliconRuntimeBackend::teardown() {
    log_info(tt::LogMetal, "Tearing Down Silicon Runtime Backend");
    metalium_object_.reset();
}

void MockDeviceRuntimeBackend::initialize(const ContextDescriptor& descriptor) {
    log_info(tt::LogMetal, "Initializing Mock Device Runtime Backend Num CQ: {}", descriptor.get_num_cqs());

    // Create RuntimeOptions configured for mock device
    rtoptions_ = std::make_unique<tt::llrt::RunTimeOptions>();

    // Use the provided mock cluster descriptor, or auto-detect from hardware
    std::string mock_desc = descriptor.get_mock_cluster_desc_path();
    if (mock_desc.empty()) {
        mock_desc = get_default_mock_cluster_desc();
    }
    rtoptions_->set_mock_cluster_desc(mock_desc);

    // Create HAL and Cluster for mock device
    const bool is_base_routing_fw_enabled =
        tt::Cluster::is_base_routing_fw_enabled(tt::Cluster::get_cluster_type_from_cluster_desc(*rtoptions_));
    const auto platform_arch = tt::tt_metal::get_platform_architecture(*rtoptions_);

    hal_ = std::make_shared<tt::tt_metal::Hal>(
        platform_arch,
        is_base_routing_fw_enabled,
        rtoptions_->get_enable_2_erisc_mode(),
        tt::tt_metal::get_profiler_dram_bank_size_for_hal_allocation(*rtoptions_));
    rtoptions_->ParseAllFeatureEnv(*hal_);
    cluster_ = std::make_shared<tt::Cluster>(*rtoptions_, *hal_);

    log_info(tt::LogMetal, "Mock Device Runtime Backend initialized using cluster descriptor: {}", mock_desc);
}

void MockDeviceRuntimeBackend::teardown() {
    log_info(tt::LogMetal, "Tearing Down Mock Device Runtime Backend");
    cluster_.reset();
    hal_.reset();
    rtoptions_.reset();
}

}  // namespace tt::tt_metal::experimental
