// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "metalium_env_impl.hpp"
#include "firmware_capability.hpp"
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include <tt-metalium/experimental/context/context_descriptor.hpp>
#include <utility>

namespace tt::tt_metal {

MetaliumEnv::MetaliumEnvImpl::MetaliumEnvImpl(MetaliumEnvDescriptor descriptor) : descriptor_(std::move(descriptor)) {
    initialize_base_objects();
    verify_fw_capabilities();
}

MetaliumEnv::MetaliumEnvImpl::~MetaliumEnvImpl() { destroy_early(); }

llrt::RunTimeOptions& MetaliumEnv::MetaliumEnvImpl::get_rtoptions() { return *rtoptions_; }
const Hal& MetaliumEnv::MetaliumEnvImpl::get_hal() { return *hal_; }
Cluster& MetaliumEnv::MetaliumEnvImpl::get_cluster() { return *cluster_; }
const MetaliumEnvDescriptor& MetaliumEnv::MetaliumEnvImpl::get_descriptor() const { return descriptor_; }
bool MetaliumEnv::MetaliumEnvImpl::is_initialized() const { return initialized_; }

void MetaliumEnv::MetaliumEnvImpl::destroy_early() noexcept {
    if (!initialized_) {
        return;
    }
    cluster_.reset();
    hal_.reset();
    rtoptions_.reset();
    initialized_ = false;
}

void MetaliumEnv::MetaliumEnvImpl::initialize_base_objects() {
    this->rtoptions_ = std::make_unique<llrt::RunTimeOptions>();

    if (descriptor_.is_mock_device()) {
        this->rtoptions_->set_mock_cluster_desc(std::string(descriptor_.mock_cluster_desc_path()));
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

void MetaliumEnv::MetaliumEnvImpl::verify_fw_capabilities() {
    FirmwareCapabilityRequest req;
    req.enable_2_erisc_mode = this->rtoptions_->get_enable_2_erisc_mode();

    FirmwareCapabilityResult res;
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);
    if (!check_firmware_capabilities(platform_arch, {.eth_fw = cluster_->get_ethernet_firmware_version()}, req, res)) {
        this->rtoptions_->set_enable_2_erisc_mode(res.enable_2_erisc_mode);
    }
    initialized_ = true;
}

MetaliumEnv::MetaliumEnv(MetaliumEnvDescriptor descriptor) :
    impl_(std::make_unique<MetaliumEnv::MetaliumEnvImpl>(std::move(descriptor))) {}

MetaliumEnv::~MetaliumEnv() { this->destroy(); }

void MetaliumEnv::destroy() { impl_->destroy_early(); }

bool MetaliumEnv::is_initialized() const { return impl_->is_initialized(); }

const MetaliumEnvDescriptor& MetaliumEnv::get_descriptor() const { return impl_->get_descriptor(); }

}  // namespace tt::tt_metal
