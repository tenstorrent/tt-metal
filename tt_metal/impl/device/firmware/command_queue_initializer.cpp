// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_initializer.hpp"

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <tt_metal.hpp>
#include "impl/context/context_descriptor.hpp"
#include "device/device_impl.hpp"

#include <tt_metal_profiler.hpp>

namespace tt::tt_metal {

CommandQueueInitializer::CommandQueueInitializer(
    std::shared_ptr<const ContextDescriptor> descriptor, bool skip_remote_devices) :
    FirmwareInitializer(std::move(descriptor)), skip_remote_devices_(skip_remote_devices) {}

void CommandQueueInitializer::init(
    const std::vector<Device*>& devices, [[maybe_unused]] const std::unordered_set<InitializerKey>& init_done) {
    devices_ = devices;

    if (descriptor_->is_mock_device()) {
        log_info(tt::LogMetal, "Skipping host-side CQ initialization for mock devices");
        initialized_ = true;
        return;
    }

    // Initialize host-side state for each MMIO device and its tunnel devices
    for (auto* dev : devices_) {
        // Only process MMIO devices at this level; tunnel devices are handled inside the loop
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }
        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        initialize_host(dev);
        if (!skip_remote_devices_) {
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Process tunnel devices from farthest to closest
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    log_debug(tt::LogMetal, "Tunnel {} Device {} Tunnel Stop: {}", t, mmio_controlled_device_id, ts);
                    auto it = std::find_if(devices_.begin(), devices_.end(), [&](Device* d) {
                        return d->id() == mmio_controlled_device_id;
                    });
                    if (it != devices_.end()) {
                        initialize_host(*it);
                    }
                }
            }
        }
    }

    initialized_ = true;
}

void CommandQueueInitializer::configure() {}

void CommandQueueInitializer::teardown() {
    devices_.clear();
    initialized_ = false;
}

bool CommandQueueInitializer::is_initialized() const { return initialized_; }

void CommandQueueInitializer::initialize_host(Device* dev) const {
    detail::ClearProfilerControlBuffer(dev);

    // Create system memory writer for this device to have an associated interface to hardware command
    // queue (i.e. hugepage). Need to do this before FW init so we know what dispatch cores to reset.
    if (using_fast_dispatch()) {
        detail::DispatchStateCheck(true);
        dev->init_command_queue_host();
    } else {
        detail::DispatchStateCheck(false);
        TT_ASSERT(dev->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }
}

bool CommandQueueInitializer::using_fast_dispatch() const { return rtoptions_.get_fast_dispatch(); }

}  // namespace tt::tt_metal
