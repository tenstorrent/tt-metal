
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pthread.h>
#include "metal_env_impl.hpp"
#include "firmware_capability.hpp"
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <utility>

namespace tt::tt_metal {

std::mutex MetalEnv::MetalEnvImpl::s_registry_mutex_;
std::set<MetalEnv::MetalEnvImpl*> MetalEnv::MetalEnvImpl::s_registry_;
std::once_flag MetalEnv::MetalEnvImpl::s_atfork_registered_;

MetalEnvDescriptor::MetalEnvDescriptor(const std::string& mock_cluster_desc_path) :
    mock_cluster_desc_path_(
        mock_cluster_desc_path.empty() ? std::nullopt : std::optional<std::string>(mock_cluster_desc_path)) {}
MetalEnvDescriptor::MetalEnvDescriptor(std::optional<std::string> mock_cluster_desc_path) :
    mock_cluster_desc_path_(std::move(mock_cluster_desc_path)) {}

void MetalEnv::MetalEnvImpl::prefork_check_all() {
    std::lock_guard<std::mutex> lock(s_registry_mutex_);
    for (auto* impl : s_registry_) {
        impl->check_use_count_zero();
    }
}

MetalEnv::MetalEnvImpl::MetalEnvImpl(MetalEnvDescriptor descriptor) : descriptor_(std::move(descriptor)) {
    initialize_base_objects();
    verify_fw_capabilities();

    std::call_once(s_atfork_registered_, []() { pthread_atfork(prefork_check_all, nullptr, nullptr); });

    std::lock_guard<std::mutex> lock(s_registry_mutex_);
    s_registry_.insert(this);
}

MetalEnv::MetalEnvImpl::~MetalEnvImpl() {
    {
        std::lock_guard<std::mutex> lock(s_registry_mutex_);
        s_registry_.erase(this);
    }
    check_use_count_zero();
    cluster_.reset();
    hal_.reset();
    rtoptions_.reset();
}

void MetalEnv::MetalEnvImpl::acquire() { use_count_.fetch_add(1, std::memory_order_acq_rel); }
void MetalEnv::MetalEnvImpl::release() { use_count_.fetch_sub(1, std::memory_order_acq_rel); }

bool MetalEnv::MetalEnvImpl::check_use_count_zero() const {
    const int use_count = use_count_.load(std::memory_order_acquire);
    if (use_count > 0) {
        log_error(
            tt::LogMetal,
            "MetalEnv has {} outstanding reference(s) at teardown or fork. All objects using the MetalEnv (e.g. open "
            "devices) must stop using it before the MetalEnv is destroyed or the process forks.",
            use_count);
        return false;
    }
    return true;
}

llrt::RunTimeOptions& MetalEnv::MetalEnvImpl::get_rtoptions() { return *rtoptions_; }
const Hal& MetalEnv::MetalEnvImpl::get_hal() { return *hal_; }
Cluster& MetalEnv::MetalEnvImpl::get_cluster() { return *cluster_; }
const MetalEnvDescriptor& MetalEnv::MetalEnvImpl::get_descriptor() const { return descriptor_; }

void MetalEnv::MetalEnvImpl::initialize_base_objects() {
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

void MetalEnv::MetalEnvImpl::verify_fw_capabilities() {
    FirmwareCapabilityRequest req;
    req.enable_2_erisc_mode = this->rtoptions_->get_enable_2_erisc_mode();

    FirmwareCapabilityResult res;
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);
    if (!check_firmware_capabilities(platform_arch, {.eth_fw = cluster_->get_ethernet_firmware_version()}, req, res)) {
        this->rtoptions_->set_enable_2_erisc_mode(res.enable_2_erisc_mode);
    }
}

MetalEnv::MetalEnv(MetalEnvDescriptor descriptor) :
    impl_(std::make_unique<MetalEnv::MetalEnvImpl>(std::move(descriptor))) {}

MetalEnv::~MetalEnv() { this->impl_.reset(); }

const MetalEnvDescriptor& MetalEnv::get_descriptor() const { return impl_->get_descriptor(); }

tt::ARCH MetalEnv::get_arch() const { return impl_->get_cluster().arch(); }
std::string MetalEnv::get_arch_name() const { return tt::get_string_lowercase(get_arch()); }
uint32_t MetalEnv::get_num_pcie_devices() const { return impl_->get_cluster().number_of_pci_devices(); }
uint32_t MetalEnv::get_l1_size() const {
    return impl_->get_hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
}
uint32_t MetalEnv::get_dram_alignment() const { return impl_->get_hal().get_alignment(HalMemType::DRAM); }
uint32_t MetalEnv::get_l1_alignment() const { return impl_->get_hal().get_alignment(HalMemType::L1); }
uint32_t MetalEnv::get_arch_num_circular_buffers() const { return impl_->get_hal().get_arch_num_circular_buffers(); }
uint32_t MetalEnv::get_max_worker_l1_unreserved_size() const {
    size_t l1_end = impl_->get_hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                    impl_->get_hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    return l1_end - impl_->get_hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
}
float MetalEnv::get_eps() const { return impl_->get_hal().get_eps(); }
float MetalEnv::get_nan() const { return impl_->get_hal().get_nan(); }
float MetalEnv::get_inf() const { return impl_->get_hal().get_inf(); }

}  // namespace tt::tt_metal
