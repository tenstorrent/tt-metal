// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_kernel_initializer.hpp"

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <tt_metal.hpp>
#include "common/executor.hpp"
#include "device/firmware/fabric_firmware_initializer.hpp"
#include "impl/context/context_descriptor.hpp"
#include "device/device_impl.hpp"

#include "dispatch/topology.hpp"

#include "llrt/hal/generated/dev_msgs.hpp"

namespace tt::llrt::internal_ {
void wait_until_cores_done(
    ChipId device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms);
}  // namespace tt::llrt::internal_

namespace tt::tt_metal {

void DispatchKernelInitializer::init(
    const std::vector<Device*>& devices, [[maybe_unused]] const std::unordered_set<InitializerKey>& init_done) {
    if (!using_fast_dispatch()) {
        return;
    }

    devices_ = devices;

    // Skip firmware initialization for mock devices
    if (descriptor_->is_mock_device()) {
        log_info(tt::LogMetal, "Skipping dispatch firmware initialization for mock devices");
        return;
    }

    TT_ASSERT(
        init_done.contains(FabricFirmwareInitializer::key()),
        "Fabric firmware must be initialized before dispatch firmware");

    // Compile dispatch kernels if using fast dispatch.
    // Note: Host-side init (profiler buffer clear, CQ host init) is a prerequisite
    // that must be performed by the caller before this point.
    if (using_fast_dispatch()) {
        compile_dispatch_kernels();
    }
}

void DispatchKernelInitializer::configure() {
    if (!using_fast_dispatch()) {
        return;
    }

    init_device_command_queues();
    initialized_ = true;
}

void DispatchKernelInitializer::teardown() {
    if (!using_fast_dispatch()) {
        return;
    }

    // Mock devices don't have sysmem_manager, skip FD teardown
    if (descriptor_->is_mock_device()) {
        return;
    }

    terminate_command_queues();
    wait_for_dispatch_cores();

    process_termination_signals();

    devices_.clear();
    initialized_ = false;
}

bool DispatchKernelInitializer::is_initialized() const { return initialized_; }

void DispatchKernelInitializer::compile_dispatch_kernels() {
    // Compile dispatch firmware: populate static args, create CQ programs, compile.

    if (descriptor_->is_mock_device()) {
        return;
    }

    // Generate static args
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }

        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        populate_cq_static_args(dev);
        for (const auto& tunnel : tunnels_from_mmio) {
            for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                uint32_t mmio_controlled_device_id = tunnel[ts];
                auto it = std::find_if(
                    devices_.begin(), devices_.end(), [&](IDevice* d) { return d->id() == mmio_controlled_device_id; });
                if (it != devices_.end()) {
                    populate_cq_static_args(*it);
                }
            }
        }
    }

    // Create command queue programs
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }

        create_cq_program(dev);
        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        for (const auto& tunnel : tunnels_from_mmio) {
            for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                uint32_t mmio_controlled_device_id = tunnel[ts];
                auto it = std::find_if(
                    devices_.begin(), devices_.end(), [&](IDevice* d) { return d->id() == mmio_controlled_device_id; });
                if (it != devices_.end()) {
                    create_cq_program(*it);
                }
            }
        }
    }

    // Compile all programs
    compile_cq_programs();
}

void DispatchKernelInitializer::init_device_command_queues() {
    // Initialize device-side command queues (parallelized per MMIO device).

    if (descriptor_->is_mock_device()) {
        return;
    }

    std::vector<std::shared_future<void>> events;
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }
        ChipId mmio_device_id = dev->id();
        events.emplace_back(detail::async([this, dev, mmio_device_id]() {
            auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(mmio_device_id);
            dev->init_command_queue_device();
            log_debug(tt::LogMetal, "Command Queue initialized on Device {}", dev->id());
            for (const auto& tunnel : tunnels_from_mmio) {
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    auto it = std::find_if(devices_.begin(), devices_.end(), [&](IDevice* d) {
                        return d->id() == mmio_controlled_device_id;
                    });
                    if (it != devices_.end()) {
                        (*it)->init_command_queue_device();
                        log_info(tt::LogMetal, "Command Queue initialized on Device {}", (*it)->id());
                    }
                }
            }
        }));
    }
    for (const auto& event : events) {
        event.get();
    }
}

void DispatchKernelInitializer::terminate_command_queues() {
    for (auto* dev : devices_) {
        auto* device = dynamic_cast<Device*>(dev);
        TT_ASSERT(device != nullptr, "Expected concrete Device but got different IDevice subclass");
        for (int cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
            auto& cq = device->command_queue(cq_id);
            cq.terminate();
        }
    }
}

void DispatchKernelInitializer::wait_for_dispatch_cores() const {
    for (auto* dev : devices_) {
        if (!dev->is_mmio_capable()) {
            continue;
        }

        auto dispatch_cores = get_virtual_dispatch_cores(dev->id());
        // Wrap in try-catch so that device close continues even if dispatch cores fail or timeout.
        // This allows the device handles to be properly released, enabling subsequent
        // device opens and tt-smi resets to succeed.
        try {
            tt::llrt::internal_::wait_until_cores_done(dev->id(), dev_msgs::RUN_MSG_GO, dispatch_cores, 0);
        } catch (const std::exception& e) {
            log_warning(
                LogMetal,
                "Device {}: Exception waiting for dispatch cores to finish during teardown. "
                "Continuing with cleanup. Error: {}",
                dev->id(),
                e.what());
        }
    }
}

void DispatchKernelInitializer::process_termination_signals() const {
    for (auto* dev : devices_) {
        const auto& info = get_registered_termination_cores(dev->id());
        for (const auto& core_to_terminate : info) {
            std::vector<uint32_t> val{core_to_terminate.val};
            detail::WriteToDeviceL1(
                dev, core_to_terminate.logical_core, core_to_terminate.address, val, core_to_terminate.core_type);
        }
        cluster_.l1_barrier(dev->id());
    }
}

bool DispatchKernelInitializer::using_fast_dispatch() const { return rtoptions_.get_fast_dispatch(); }

}  // namespace tt::tt_metal
