// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mutex>

#include <tt-metalium/experimental/context_descriptor.hpp>
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

#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_query_manager.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_settings.hpp"

namespace tt::tt_metal::experimental {

class RuntimeBackend {
public:
    virtual ~RuntimeBackend() = default;

    virtual void initialize(const ContextDescriptor& descriptor) = 0;
    virtual void teardown() = 0;
};

class SiliconRuntime : public RuntimeBackend {
private:
    // Debug Tools
    std::unique_ptr<tt::tt_metal::inspector::Data> inspector_data_;
    std::unique_ptr<tt::tt_metal::DPrintServer> dprint_server_;
    std::unique_ptr<tt::tt_metal::WatcherServer> watcher_server_;
    std::unique_ptr<tt::tt_metal::ProfilerStateManager> profiler_state_manager_;
    std::unique_ptr<tt::tt_metal::DataCollector> data_collector_;
    std::unique_ptr<tt::tt_metal::NOCDebugState> noc_debug_state_;

    // Dispatch
    std::unique_ptr<tt::tt_metal::dispatch_core_manager> dispatch_core_manager_;
    std::unique_ptr<tt::tt_metal::DispatchQueryManager> dispatch_query_manager_;
    std::array<std::unique_ptr<tt::tt_metal::DispatchMemMap>, static_cast<size_t>(CoreType::COUNT)> dispatch_mem_map_;

public:
    void initialize(const ContextDescriptor& descriptor) override {
        auto& cluster_query = MetaliumObject::instance();
        dispatch_core_manager_ =
            std::make_unique<dispatch_core_manager>(descriptor.get_dispatch_core_config(), descriptor.get_num_cqs());
        // dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(context.get_num_cqs());

        tt_metal::DispatchSettings::initialize(cluster_query.cluster());
        dispatch_mem_map_[enchantum::to_underlying(CoreType::WORKER)] =
            std::make_unique<tt::tt_metal::DispatchMemMap>(CoreType::WORKER, descriptor.get_num_cqs());
        dispatch_mem_map_[enchantum::to_underlying(CoreType::ETH)] =
            std::make_unique<tt::tt_metal::DispatchMemMap>(CoreType::ETH, descriptor.get_num_cqs());
        noc_debug_state_ = std::make_unique<tt::tt_metal::NOCDebugState>();
    }

    void teardown() override {
        // Reverse order of initialize
        noc_debug_state_.reset();

        dispatch_query_manager_.reset();
        dispatch_core_manager_.reset();
    }
};

class MockRuntime : public RuntimeBackend {
public:
    void initialize(const ContextDescriptor& /*descriptor*/) override {}

    void teardown() override {}
};

class Context::Impl {
private:
    mutable std::mutex mutex_;
    std::shared_ptr<ContextDescriptor> descriptor_;
    std::unique_ptr<RuntimeBackend> backend_;

    static std::unique_ptr<RuntimeBackend> create_backend() {
        if (tt::tt_metal::get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
            return std::make_unique<MockRuntime>();
        }
        return std::make_unique<SiliconRuntime>();
    }

public:
    bool set_descriptor(const std::shared_ptr<ContextDescriptor>& descriptor) {
        // MetaliumObject must be initialized before binding a context
        if (!MetaliumObject::instance().is_initialized()) {
            log_critical(tt::LogMetal, "Cannot bind context: MetaliumObject is not initialized");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (descriptor_) {
            log_critical(tt::LogMetal, "Cannot bind context: a context is already bound");
            return false;
        }

        backend_ = create_backend();

        descriptor_ = descriptor;
        descriptor_->set_bound(true);

        backend_->initialize(*descriptor_);

        return true;
    }

    bool remove_descriptor() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!descriptor_) {
            return false;
        }

        if (backend_) {
            backend_->teardown();
            backend_.reset();
        }

        descriptor_->set_bound(false);
        descriptor_.reset();
        return true;
    }

    bool has_descriptor() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return descriptor_ != nullptr;
    }

    std::shared_ptr<ContextDescriptor> get_descriptor() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return descriptor_;
    }
};

Context::Context() : impl_(std::make_unique<Impl>()) {}

Context::~Context() = default;

Context& Context::instance() {
    static Context instance;
    return instance;
}

bool Context::set_descriptor(const std::shared_ptr<ContextDescriptor>& context_descriptor) {
    return impl_->set_descriptor(context_descriptor);
}

bool Context::remove_descriptor() { return impl_->remove_descriptor(); }

bool Context::has_descriptor() const { return impl_->has_descriptor(); }

std::shared_ptr<ContextDescriptor> Context::get_descriptor() const { return impl_->get_descriptor(); }

}  // namespace tt::tt_metal::experimental
