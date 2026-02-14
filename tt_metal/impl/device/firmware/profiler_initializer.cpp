// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler_initializer.hpp"

#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <device.hpp>
#include "device/device_impl.hpp"
#include "impl/context/context_descriptor.hpp"

#include <tt_metal_profiler.hpp>
#include "profiler/profiler_state.hpp"
#include "profiler/profiler_state_manager.hpp"

namespace tt::tt_metal {

ProfilerInitializer::ProfilerInitializer(
    std::shared_ptr<const ContextDescriptor> descriptor,
    bool skip_remote_devices,
    ProfilerStateManager* profiler_state_manager) :
    FirmwareInitializer(std::move(descriptor)),
    skip_remote_devices_(skip_remote_devices),
    profiler_state_manager_(profiler_state_manager) {}

void ProfilerInitializer::init(
    [[maybe_unused]] const std::vector<Device*>& devices,
    [[maybe_unused]] const std::unordered_set<InitializerKey>& init_done) {
#if defined(TRACY_ENABLE)
    devices_ = devices;

    if (!getDeviceProfilerState()) {
        return;
    }

    for (auto* dev : devices_) {
        // For Galaxy init, we only need to loop over mmio devices
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }
        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        detail::InitDeviceProfiler(dev);
        log_info(tt::LogMetal, "Profiler started on device {}", dev->id());
        if (!skip_remote_devices_) {
            for (const auto& tunnel : tunnels_from_mmio) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    auto it = std::find_if(devices_.begin(), devices_.end(), [&](Device* d) {
                        return d->id() == mmio_controlled_device_id;
                    });
                    if (it != devices_.end()) {
                        detail::InitDeviceProfiler(*it);
                        log_info(tt::LogMetal, "Profiler started on remote device {}", (*it)->id());
                    }
                }
            }
        }
    }
    detail::ProfilerSync(ProfilerSyncState::INIT);

    if (profiler_state_manager_ && rtoptions_.get_experimental_noc_debug_dump_enabled()) {
        tt::tt_metal::LaunchIntervalBasedProfilerReadThread(std::vector<IDevice*>(devices_.begin(), devices_.end()));
    }
#endif

    initialized_ = true;
}

void ProfilerInitializer::configure() {}

void ProfilerInitializer::teardown() {
    // Read profiler results from dispatch cores
    for (auto* dev : devices_) {
        detail::ReadDeviceProfilerResults(static_cast<IDevice*>(dev), ProfilerReadState::ONLY_DISPATCH_CORES);
    }
    detail::ProfilerSync(ProfilerSyncState::CLOSE_DEVICE);

    devices_.clear();
    initialized_ = false;
}

void ProfilerInitializer::post_teardown() {
    if (getDeviceProfilerState()) {
        // Device profiling data is dumped here instead of MetalContext::teardown() because MetalContext::teardown() is
        // called as a std::atexit() function, and ProfilerStateManager::cleanup_device_profilers() cannot be safely
        // called from a std::atexit() function because it creates new threads, which is unsafe during program
        // termination.
        profiler_state_manager_->cleanup_device_profilers();
    }
}

bool ProfilerInitializer::is_initialized() const { return initialized_; }

}  // namespace tt::tt_metal
