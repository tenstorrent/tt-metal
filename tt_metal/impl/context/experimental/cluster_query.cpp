// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/runtime.hpp>

#include <mutex>
#include <stdexcept>

#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt-metalium/hal.hpp"
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "impl/context/common.hpp"

namespace tt::tt_metal::experimental {

#define LOCK_ASSERT_INITIALIZED_FUNCTION()                           \
    std::lock_guard<std::mutex> lock(mutex_);                        \
    if (!initialized_) {                                             \
        throw std::runtime_error("ClusterQuery is not initialized"); \
    }

class ClusterQuery::Impl {
public:
    tt::llrt::RunTimeOptions rtoptions_;

    // These pointers must not be null if the Runtime is active
    std::unique_ptr<tt::tt_metal::Hal> hal_;
    std::unique_ptr<tt::Cluster> cluster_;
    std::unique_ptr<tt::tt_fabric::ControlPlane> control_plane_;
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context_;
    std::shared_ptr<distributed::multihost::DistributedContext> compute_only_distributed_context_;

    mutable std::mutex mutex_;
    bool initialized_ = false;

    bool initialize() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            log_critical(tt::LogMetal, "Cannot initialize ClusterQuery: already initialized");
            return false;
        }

        const bool is_base_routing_fw_enabled =
            tt::Cluster::is_base_routing_fw_enabled(tt::Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
        const auto platform_arch = tt::tt_metal::get_platform_architecture(rtoptions_);

        const auto initialize_objects = [&]() {
            hal_ = std::make_unique<tt::tt_metal::Hal>(
                platform_arch,
                is_base_routing_fw_enabled,
                rtoptions_.get_enable_2_erisc_mode(),
                tt::tt_metal::get_profiler_dram_bank_size_for_hal_allocation(rtoptions_));
            rtoptions_.ParseAllFeatureEnv(*hal_);
            cluster_ = std::make_unique<tt::Cluster>(rtoptions_, *hal_);
            distributed_context_ = distributed::multihost::DistributedContext::get_current_world();
        };

        initialize_objects();

        // Requires reinit with features disabled
        // This will maintain backward compatibility with clusters that have legacy firmware but it will cause a
        // slowdown during the first init
        if (!cluster_->verify_eth_fw_capability()) {
            rtoptions_.set_enable_2_erisc_mode(false);
            cluster_.reset();
            hal_.reset();
            initialize_objects();
        }

        initialized_ = true;

        return true;
    }

    bool teardown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            log_critical(tt::LogMetal, "Cannot teardown ClusterQuery: not initialized");
            return false;
        }
        if (Runtime::instance().has_bound_context()) {
            log_critical(tt::LogMetal, "Cannot teardown ClusterQuery: Runtime has a bound context");
            return false;
        }
        cluster_.reset();
        hal_.reset();
        initialized_ = false;
        return true;
    }

    bool is_initialized() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_;
    }

    tt::Cluster& cluster() {
        LOCK_ASSERT_INITIALIZED_FUNCTION();
        return *cluster_;
    }

    const tt::Cluster& cluster() const {
        LOCK_ASSERT_INITIALIZED_FUNCTION();
        return *cluster_;
    }

    tt::tt_metal::Hal& hal() {
        LOCK_ASSERT_INITIALIZED_FUNCTION();
        return *hal_;
    }

    const tt::tt_metal::Hal& hal() const {
        LOCK_ASSERT_INITIALIZED_FUNCTION();
        return *hal_;
    }

    tt::llrt::RunTimeOptions& rtoptions() { return rtoptions_; }

    const tt::llrt::RunTimeOptions& rtoptions() const { return rtoptions_; }

    const distributed::multihost::DistributedContext& full_world_distributed_context() const {
        TT_FATAL(distributed_context_, "Distributed context not initialized.");
        return *distributed_context_;
    }

    const distributed::multihost::DistributedContext& global_distributed_context() {
        // If control plane is not initilazed, return the global distributed context
        if (!control_plane_) {
            return *distributed_context_;
        }
        // Lazy initilazation of compute only distributed context
        if (!compute_only_distributed_context_) {
            compute_only_distributed_context_ =
                tt::tt_metal::construct_compute_only_distributed_context(*control_plane_);
        }
        return *compute_only_distributed_context_;
    }

    std::shared_ptr<distributed::multihost::DistributedContext> get_distributed_context_ptr() {
        TT_FATAL(distributed_context_, "Distributed context not initialized.");
        return distributed_context_;
    }
};

#undef LOCK_ASSERT_INITIALIZED_FUNCTION

ClusterQuery::ClusterQuery() : impl_(std::make_unique<Impl>()) {}

ClusterQuery::~ClusterQuery() = default;

ClusterQuery& ClusterQuery::instance() {
    static ClusterQuery instance;
    return instance;
}

bool ClusterQuery::initialize() { return impl_->initialize(); }

bool ClusterQuery::teardown() { return impl_->teardown(); }

bool ClusterQuery::is_initialized() const { return impl_->is_initialized(); }

bool ClusterQuery::is_runtime_active() const { return Runtime::instance().has_bound_context(); }

int ClusterQuery::get_num_visible_devices() const { return impl_->cluster().number_of_user_devices(); }

int ClusterQuery::get_num_pcie_devices() const { return impl_->cluster().number_of_pci_devices(); }

bool ClusterQuery::is_galaxy_cluster() const { return impl_->cluster().is_galaxy_cluster(); }

int ClusterQuery::get_pcie_device_id(int device_id) const {
    return impl_->cluster().get_associated_mmio_device(device_id);
}

// Internal functions

tt::Cluster& ClusterQuery::cluster() { return impl_->cluster(); }

const tt::Cluster& ClusterQuery::cluster() const { return impl_->cluster(); }

tt::tt_metal::Hal& ClusterQuery::hal() { return impl_->hal(); }

const tt::tt_metal::Hal& ClusterQuery::hal() const { return impl_->hal(); }

tt::llrt::RunTimeOptions& ClusterQuery::rtoptions() { return impl_->rtoptions(); }

const tt::llrt::RunTimeOptions& ClusterQuery::rtoptions() const { return impl_->rtoptions(); }

}  // namespace tt::tt_metal::experimental
