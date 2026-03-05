// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/context/metalium_env.hpp>
#include "firmware_capability.hpp"
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include <tt-metalium/experimental/context/context_descriptor.hpp>
#include "impl/context/metalium_env_accessor.hpp"

namespace tt::tt_metal {

MetaliumEnv::MetaliumEnv(MetaliumEnvDescriptor descriptor) : initialized_(true), descriptor_(std::move(descriptor)) {
    this->initialize_base_objects();
}

MetaliumEnv::~MetaliumEnv() { this->destroy(); }

void MetaliumEnv::destroy() {
    if (!initialized_) {
        return;
    }
    cluster_.reset();
    hal_.reset();
    rtoptions_.reset();
    initialized_ = false;
}

llrt::RunTimeOptions& MetaliumEnv::get_rtoptions() const {
    TT_FATAL(rtoptions_ != nullptr, "MetaliumEnv not initialized");
    return *rtoptions_;
}

const tt::tt_metal::Hal& MetaliumEnv::get_hal() const {
    TT_FATAL(hal_ != nullptr, "MetaliumEnv not initialized");
    return *hal_;
}

tt::Cluster& MetaliumEnv::get_cluster() const {
    TT_FATAL(cluster_ != nullptr, "MetaliumEnv not initialized");
    return *cluster_;
}

bool MetaliumEnv::is_initialized() const { return initialized_; }

void MetaliumEnv::acquire(int context_id) {
    int expected = NO_OWNER;
    TT_FATAL(
        owning_context_id_.compare_exchange_strong(expected, context_id, std::memory_order_acq_rel),
        "MetaliumEnv is already in use by MetalContext (context_id={}). Cannot acquire for context_id={}.",
        expected,
        context_id);
}

void MetaliumEnv::release(int context_id) {
    int expected = context_id;
    TT_FATAL(
        owning_context_id_.compare_exchange_strong(expected, NO_OWNER, std::memory_order_acq_rel),
        "MetaliumEnv release mismatch: expected context_id={}, actual context_id={}.",
        context_id,
        expected);
}

bool MetaliumEnv::is_acquired() const { return owning_context_id_.load(std::memory_order_acquire) != NO_OWNER; }

const MetaliumEnvDescriptor& MetaliumEnv::get_descriptor() const { return descriptor_; }

void MetaliumEnv::initialize_base_objects() {
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

void MetaliumEnv::verify_fw_capabilities() {
    FirmwareCapabilityRequest req;
    req.enable_2_erisc_mode = this->rtoptions_->get_enable_2_erisc_mode();

    FirmwareCapabilityResult res;
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);
    if (!check_firmware_capabilities(platform_arch, {.eth_fw = cluster_->get_ethernet_firmware_version()}, req, res)) {
        this->rtoptions_->set_enable_2_erisc_mode(res.enable_2_erisc_mode);
    }
}

MetaliumEnvAccessor::MetaliumEnvAccessor(MetaliumEnv& metalium_env) noexcept : metalium_env_(&metalium_env) {}
MetaliumEnvAccessor::MetaliumEnvAccessor(const MetaliumEnv& metalium_env) noexcept :
    metalium_env_(const_cast<MetaliumEnv*>(&metalium_env)) {}
MetaliumEnvAccessor::MetaliumEnvAccessor(std::unique_ptr<MetaliumEnv> metalium_env) noexcept :
    owned_env_(std::move(metalium_env)), metalium_env_(owned_env_.get()) {}

}  // namespace tt::tt_metal
