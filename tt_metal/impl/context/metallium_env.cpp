// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "metallium_env.hpp"
#include "firmware_capability.hpp"
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

MetalliumEnv::MetalliumEnv() = default;

MetalliumEnv::~MetalliumEnv() { this->destroy(); }

void MetalliumEnv::initialize(const std::shared_ptr<MetalliumEnvDescriptor>& descriptor) {
    TT_FATAL(!initialized_, "MetalliumEnv already initialized");
    this->initialize_base_objects(descriptor);
    initialized_ = true;
}

void MetalliumEnv::destroy() {
    if (!initialized_) {
        return;
    }
    cluster_.reset();
    hal_.reset();
    rtoptions_.reset();
    initialized_ = false;
}

llrt::RunTimeOptions& MetalliumEnv::get_rtoptions() const {
    TT_FATAL(rtoptions_ != nullptr, "MetalliumEnv not initialized");
    return *rtoptions_;
}

const tt::tt_metal::Hal& MetalliumEnv::get_hal() const {
    TT_FATAL(hal_ != nullptr, "MetalliumEnv not initialized");
    return *hal_;
}

tt::Cluster& MetalliumEnv::get_cluster() const {
    TT_FATAL(cluster_ != nullptr, "MetalliumEnv not initialized");
    return *cluster_;
}

bool MetalliumEnv::is_initialized() const { return initialized_; }

void MetalliumEnv::initialize_base_objects(const std::shared_ptr<MetalliumEnvDescriptor>& descriptor) {
    this->rtoptions_ = std::make_unique<llrt::RunTimeOptions>();

    if (descriptor->is_mock_device()) {
        this->rtoptions_->set_mock_cluster_desc(std::string(descriptor->mock_cluster_desc_path()));
    }

    const bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(*this->rtoptions_));
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);

    cluster_ = std::make_unique<Cluster>(*this->rtoptions_);
    this->verify_fw_capabilities();
    this->hal_ = std::make_unique<Hal>(
        platform_arch,
        is_base_routing_fw_enabled,
        this->rtoptions_->get_enable_2_erisc_mode(),
        get_profiler_dram_bank_size_for_hal_allocation(*this->rtoptions_));

    this->rtoptions_->ParseAllFeatureEnv(*hal_);
    this->cluster_->set_hal(hal_.get());
}

void MetalliumEnv::verify_fw_capabilities() {
    FirmwareCapabilityRequest req;
    req.enable_2_erisc_mode = this->rtoptions_->get_enable_2_erisc_mode();

    FirmwareCapabilityResult res;
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);
    if (!check_firmware_capabilities(platform_arch, {.eth_fw = cluster_->get_ethernet_firmware_version()}, req, res)) {
        this->rtoptions_->set_enable_2_erisc_mode(res.enable_2_erisc_mode);
    }
}

}  // namespace tt::tt_metal
