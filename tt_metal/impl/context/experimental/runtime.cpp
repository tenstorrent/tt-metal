// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <llrt/rtoptions.hpp>
#include "impl/context/experimental/runtime.hpp"
#include "impl/context/experimental/runtime_backends.hpp"

#include <memory>
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"
#include "tt_cluster.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::experimental {

class MetaliumObject::Impl {
public:
    mutable std::mutex mutex_;
    std::atomic<bool> is_initialized_ = false;

    // Listed in the order of initialization
    tt::llrt::RunTimeOptions rtoptions_;
    std::shared_ptr<tt::tt_metal::Hal> hal_;
    std::shared_ptr<tt::Cluster> cluster_;
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context_;

    // Reset when setting Fabric configuration
    std::shared_ptr<tt::tt_fabric::ControlPlane> control_plane_;

    // Maybe null -- lazy initialized
    std::shared_ptr<distributed::multihost::DistributedContext> compute_only_distributed_context_;

    // Check if it's safe to destroy the MetaliumObject
    bool safe_to_destroy() const noexcept {
        // Distributed context is not owned by this object.
        // compute_only_distributed_context_ is lazy initialized and may be zero.
        return hal_.use_count() == 1 && cluster_.use_count() == 1 && control_plane_.use_count() <= 1;
    }

    bool create() {
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (this->is_initialized_) {
            log_error(tt::LogMetal, "MetaliumObject already initialized");
            return false;
        }

        const bool is_base_routing_fw_enabled =
            tt::Cluster::is_base_routing_fw_enabled(tt::Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
        const auto platform_arch = tt::tt_metal::get_platform_architecture(rtoptions_);

        const auto initialize_objects = [&]() {
            hal_ = std::make_shared<tt::tt_metal::Hal>(
                platform_arch,
                is_base_routing_fw_enabled,
                rtoptions_.get_enable_2_erisc_mode(),
                tt::tt_metal::get_profiler_dram_bank_size_for_hal_allocation(rtoptions_));
            rtoptions_.ParseAllFeatureEnv(*hal_);
            cluster_ = std::make_shared<tt::Cluster>(rtoptions_, *hal_);
            distributed_context_ = distributed::multihost::DistributedContext::get_current_world();
        };

        initialize_objects();

        // Reinit objects if needed because of the circular dependency between the tt::Cluster and the HAL.
        // Ideally, we can create the tt::Cluster and query the firmware version **before* creating the HAL.
        if (!cluster_->verify_eth_fw_capability()) {
            rtoptions_.set_enable_2_erisc_mode(false);
            initialize_objects();
        }

        this->is_initialized_ = true;
        return true;
    }

    bool destroy() {
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (!this->is_initialized_) {
            return true;
        }

        if (!this->safe_to_destroy()) {
            TT_THROW(
                "Cannot destroy MetaliumObject while still in use. Use counts - hal: {}, cluster: {}, "
                "distributed_context: {}, control_plane: {}, compute_only: {}",
                hal_.use_count(),
                cluster_.use_count(),
                distributed_context_.use_count(),
                control_plane_.use_count(),
                compute_only_distributed_context_.use_count());
            return false;
        }

        // Destroy in reverse order of initialization
        this->compute_only_distributed_context_.reset();
        this->distributed_context_.reset();
        this->control_plane_.reset();
        this->cluster_.reset();
        this->hal_.reset();
        this->is_initialized_ = false;
        return true;
    }
};

std::shared_ptr<MetaliumObject> MetaliumObject::create() {
    auto metalium_object = std::shared_ptr<MetaliumObject>(new MetaliumObject());
    metalium_object->impl_->create();
    return metalium_object;
}

MetaliumObject::MetaliumObject() { impl_ = std::make_unique<Impl>(); }

MetaliumObject::~MetaliumObject() { this->impl_->destroy(); }

std::shared_ptr<tt::Cluster> MetaliumObject::cluster() { return this->impl_->cluster_; }

std::shared_ptr<tt::tt_metal::Hal> MetaliumObject::hal() { return this->impl_->hal_; }

tt::llrt::RunTimeOptions& MetaliumObject::rtoptions() { return this->impl_->rtoptions_; }

std::shared_ptr<tt::tt_fabric::ControlPlane> MetaliumObject::get_control_plane() { return this->impl_->control_plane_; }

Context::Context(std::shared_ptr<ContextDescriptor> descriptor, std::shared_ptr<MetaliumObject> metalium_object) :
    impl_(nullptr), descriptor_(std::move(descriptor)) {
    if (!descriptor_) {
        TT_THROW("ContextDescriptor must not be null");
    }

    if (descriptor_->is_mock_device_) {
        // Mock context
        impl_ = std::make_unique<MockDeviceRuntimeBackend>();
    } else {
        // Silicon context - requires MetaliumObject as an explicit dependency
        if (!metalium_object) {
            TT_THROW("Silicon context requires a MetaliumObject");
        }
        impl_ = std::make_unique<SiliconRuntimeBackend>(std::move(metalium_object));
    }

    impl_->initialize(*descriptor_);
}

Context::~Context() {
    if (impl_) {
        impl_->teardown();
        impl_.reset();
    }
    descriptor_.reset();
}

bool Context::is_mock_device() const { return descriptor_ && descriptor_->is_mock_device_; }

std::shared_ptr<ContextDescriptor> Context::get_descriptor() const { return descriptor_; }

std::shared_ptr<MetaliumObject> Context::get_metalium_object() {
    if (is_mock_device()) {
        return nullptr;
    }
    auto* silicon_backend = dynamic_cast<SiliconRuntimeBackend*>(impl_.get());
    return silicon_backend ? silicon_backend->get_metalium_object() : nullptr;
}

}  // namespace tt::tt_metal::experimental
