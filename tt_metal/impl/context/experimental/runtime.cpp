// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mutex>

#include <tt-metalium/experimental/context.hpp>
#include <tt-metalium/experimental/runtime.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-logger/tt-logger.hpp>

#include <llrt/tt_cluster.hpp>

#include "debug/dprint_server.hpp"
#include "debug/inspector/data.hpp"
#include "debug/inspector/inspector.hpp"
#include "debug/noc_logging.hpp"
#include "debug/noc_debugging.hpp"
#include "debug/watcher_server.hpp"
#include "dispatch/data_collector.hpp"
#include "profiler/profiler_state_manager.hpp"

namespace tt::tt_metal::experimental {

class Runtime::Impl {
private:
    // Debug Tools
    std::unique_ptr<tt::tt_metal::inspector::Data> inspector_data_;
    std::unique_ptr<tt::tt_metal::DPrintServer> dprint_server_;
    std::unique_ptr<tt::tt_metal::WatcherServer> watcher_server_;
    std::unique_ptr<tt::tt_metal::ProfilerStateManager> profiler_state_manager_;
    std::unique_ptr<tt::tt_metal::DataCollector> data_collector_;
    std::unique_ptr<tt::tt_metal::NOCDebugState> noc_debug_state_;

    void initialize_fw() {}

    void teardown_fw() {
        // Items are commented out because they are causing MetalContext
        // auto& system_query = ClusterQuery::instance();
        // auto& cluster = system_query.cluster();

        // if (cluster.get_target_device_type() != tt::TargetDevice::Mock) {
        //     cluster.set_internal_routing_info_for_ethernet_cores(false);
        // }

        // if (dprint_server_) {
        //     if (cluster.get_target_device_type() != tt::TargetDevice::Mock) {
        //         dprint_server_->detach_devices();
        //     }
        //     dprint_server_.reset();
        // }

        // if (watcher_server_) {
        //     if (cluster.get_target_device_type() != tt::TargetDevice::Mock) {
        //         watcher_server_->detach_devices();
        //     }
        //     watcher_server_.reset();
        // }

        // if (cluster.get_target_device_type() != tt::TargetDevice::Mock) {
        //     for (ChipId device_id : all_devices) {
        //         assert_cores(device_id);

        //         cluster->l1_barrier(device_id);
        //     }
        // }

        noc_debug_state_.reset();

        if (data_collector_) {
            data_collector_->DumpData();
            data_collector_.reset();
        }

        profiler_state_manager_.reset();

        // Inspector::clear_all_core_info();
        // inspector_data_.reset();
    }

public:
    std::shared_ptr<Context> context_;
    mutable std::mutex mutex_;

    bool bind_context(const std::shared_ptr<Context>& context) {
        // ClusterQuery must be initialized before binding a context because we need it to access the device
        if (!ClusterQuery::instance().is_initialized()) {
            log_critical(tt::LogMetal, "Cannot bind context: ClusterQuery is not initialized");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (context_) {
            log_critical(tt::LogMetal, "Cannot bind context: a context is already bound");
            return false;
        }
        context_ = context;

        log_info(tt::LogMetal, "Context bound");
        initialize_fw();

        return true;
    }

    bool unbind_context() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!context_) {
            return false;
        }

        teardown_fw();
        context_.reset();

        return true;
    }

    bool has_bound_context() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return context_ != nullptr;
    }

    std::shared_ptr<Context> get_context() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return context_;
    }
};

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {}

Runtime::~Runtime() = default;

Runtime& Runtime::instance() {
    static Runtime instance;
    return instance;
}

bool Runtime::bind_context(const std::shared_ptr<Context>& context) { return impl_->bind_context(context); }

bool Runtime::unbind_context() { return impl_->unbind_context(); }

bool Runtime::has_bound_context() const { return impl_->has_bound_context(); }

std::shared_ptr<Context> Runtime::get_context() const { return impl_->get_context(); }

}  // namespace tt::tt_metal::experimental
