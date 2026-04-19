// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_kernel_initializer.hpp"

#include <thread>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <tt_metal.hpp>
#include "common/executor.hpp"
#include "device/firmware/firmware_initializer.hpp"
#include "impl/context/context_descriptor.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "device/device_impl.hpp"

#include "dispatch/topology.hpp"

#include "llrt/hal/generated/dev_msgs.hpp"

namespace tt::llrt::internal_ {
void wait_until_cores_done(
    ChipId device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms, bool skip_dispatch_alert);
}  // namespace tt::llrt::internal_

namespace tt::tt_metal {

DispatchKernelInitializer::DispatchKernelInitializer(
    std::shared_ptr<const ContextDescriptor> descriptor,
    dispatch_core_manager& dispatch_core_manager,
    DeviceManager* device_manager,
    const GetControlPlaneFn& get_control_plane,
    const GetDispatchQueryManagerFn& get_dispatch_query_manager,
    const GetMaxNumEthCoresFn& get_max_num_eth_cores,
    const GetReadsDispatchCoresFn& get_reads_dispatch_cores) :
    FirmwareInitializer(std::move(descriptor)),
    dispatch_core_manager_(dispatch_core_manager),
    device_manager_(device_manager),
    get_control_plane_(get_control_plane),
    get_dispatch_query_manager_(get_dispatch_query_manager),
    get_max_num_eth_cores_(get_max_num_eth_cores),
    get_reads_dispatch_cores_(get_reads_dispatch_cores) {
    dispatch_topology_ = std::make_unique<tt::tt_metal::DispatchTopology>(
        *descriptor_,
        dispatch_core_manager_,
        device_manager_,
        get_control_plane_,
        get_dispatch_query_manager_,
        get_max_num_eth_cores_,
        get_reads_dispatch_cores_);
}

void DispatchKernelInitializer::populate_fd_kernels_only(const std::vector<Device*>& devices) {
    if (!using_fast_dispatch() || devices.empty()) {
        return;
    }
    dispatch_topology_->populate_fd_kernels(devices, descriptor_->num_cqs());
}

void DispatchKernelInitializer::init(
    const std::vector<Device*>& devices, [[maybe_unused]] const std::unordered_set<InitializerKey>& init_done) {
    if (!using_fast_dispatch()) {
        return;
    }

    devices_ = devices;

    bool is_galaxy_cluster = descriptor_->cluster().is_galaxy_cluster();
    dispatch_mem_map_[enchantum::to_underlying(CoreType::WORKER)] = std::make_unique<tt::tt_metal::DispatchMemMap>(
        CoreType::WORKER,
        descriptor_->num_cqs(),
        descriptor_->hal(),
        is_galaxy_cluster,
        descriptor_->rtoptions().get_dram_backed_cq());
    dispatch_mem_map_[enchantum::to_underlying(CoreType::ETH)] = std::make_unique<tt::tt_metal::DispatchMemMap>(
        CoreType::ETH,
        descriptor_->num_cqs(),
        descriptor_->hal(),
        is_galaxy_cluster,
        descriptor_->rtoptions().get_dram_backed_cq());

    // Skip firmware initialization for mock devices
    if (descriptor_->is_mock_device()) {
        log_info(tt::LogMetal, "Skipping dispatch firmware initialization for mock devices");
        return;
    }

    // Dispatch requires Fabric, Profiler, and Command Queue
    TT_ASSERT(
        init_done.contains(InitializerKey::Fabric), "Fabric firmware must be initialized before dispatch firmware");
    TT_ASSERT(
        init_done.contains(InitializerKey::Profiler), "Profiler firmware must be initialized before dispatch firmware");
    TT_ASSERT(
        init_done.contains(InitializerKey::CommandQueue),
        "Host-side command queue firmware must be initialized before dispatch firmware");
    compile_dispatch_kernels();
}

void DispatchKernelInitializer::configure() {
    if (!using_fast_dispatch()) {
        return;
    }

    init_device_command_queues();
    initialized_ = true;
}

void DispatchKernelInitializer::teardown(std::unordered_set<InitializerKey>& init_done) {
    // Dispatch is torn down first; no prior teardown order to assert.
    if (!using_fast_dispatch()) {
        init_done.erase(key);
        return;
    }

    // Mock devices don't have sysmem_manager, skip FD teardown
    if (descriptor_->is_mock_device()) {
        init_done.erase(key);
        return;
    }

    log_info(tt::LogMetal, "[dispatch_teardown] terminate_command_queues() start");
    terminate_command_queues();
    log_info(tt::LogMetal, "[dispatch_teardown] terminate_command_queues() returned, calling wait_for_dispatch_cores()");
    wait_for_dispatch_cores();
    log_info(tt::LogMetal, "[dispatch_teardown] wait_for_dispatch_cores() returned, calling process_termination_signals()");
    process_termination_signals();
    log_info(tt::LogMetal, "[dispatch_teardown] process_termination_signals() returned");

    devices_.clear();
    initialized_ = false;
    init_done.erase(key);
}

bool DispatchKernelInitializer::is_initialized() const { return initialized_; }

void DispatchKernelInitializer::compile_dispatch_kernels() {
    if (descriptor_->is_mock_device()) {
        return;
    }

    // Generate static args
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }

        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        dispatch_topology_->populate_cq_static_args(dev);
        for (const auto& tunnel : tunnels_from_mmio) {
            for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                uint32_t mmio_controlled_device_id = tunnel[ts];
                auto it = std::find_if(
                    devices_.begin(), devices_.end(), [&](IDevice* d) { return d->id() == mmio_controlled_device_id; });
                if (it != devices_.end()) {
                    dispatch_topology_->populate_cq_static_args(*it);
                }
            }
        }
    }

    // Create command queue programs
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }

        dispatch_topology_->create_cq_program(dev);
        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        for (const auto& tunnel : tunnels_from_mmio) {
            for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                uint32_t mmio_controlled_device_id = tunnel[ts];
                auto it = std::find_if(
                    devices_.begin(), devices_.end(), [&](IDevice* d) { return d->id() == mmio_controlled_device_id; });
                if (it != devices_.end()) {
                    dispatch_topology_->create_cq_program(*it);
                }
            }
        }
    }

    // Compile all programs
    dispatch_topology_->compile_cq_programs();
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
            dev->init_command_queue_device_with_topology(dispatch_topology_.get());
            log_debug(tt::LogMetal, "Command Queue initialized on Device {}", dev->id());
            for (const auto& tunnel : tunnels_from_mmio) {
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    auto it = std::find_if(devices_.begin(), devices_.end(), [&](IDevice* d) {
                        return d->id() == mmio_controlled_device_id;
                    });
                    if (it != devices_.end()) {
                        (*it)->init_command_queue_device_with_topology(dispatch_topology_.get());
                        log_debug(tt::LogMetal, "Command Queue initialized on Device {}", (*it)->id());
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

const std::unordered_set<CoreCoord>& DispatchKernelInitializer::get_virtual_dispatch_cores(ChipId dev_id) const {
    return dispatch_topology_->get_virtual_dispatch_cores(dev_id);
}

const std::unordered_set<CoreCoord>& DispatchKernelInitializer::get_virtual_dispatch_routing_cores(
    ChipId dev_id) const {
    return dispatch_topology_->get_virtual_dispatch_routing_cores(dev_id);
}

void DispatchKernelInitializer::wait_for_dispatch_cores() const {
    for (auto* dev : devices_) {
        auto dispatch_cores = get_virtual_dispatch_cores(dev->id());
        log_info(tt::LogMetal, "[dispatch_teardown] wait_for_dispatch_cores device={} num_cores={}", dev->id(), dispatch_cores.size());
        // Wrap in try-catch so that device close continues even if dispatch cores fail or timeout.
        // This allows the device handles to be properly released, enabling subsequent
        // device opens and tt-smi resets to succeed.
        // skip_dispatch_alert=true: do NOT invoke on_dispatch_timeout_detected() (which runs tt-triage,
        // taking ~27s) when teardown times out.  During teardown, dispatch cores can legitimately
        // fail to finish (e.g. FABRIC_2D: close_finish() spins waiting for an ERISC ack that never
        // arrives because the fabric was already torn down).  The exception is caught below and
        // teardown continues; running triage here adds 27s per device and causes the test suite to
        // exceed the 700s predecessor timeout.
        // Use 200ms explicit timeout instead of 0 (which inherits TT_METAL_OPERATION_TIMEOUT_SECONDS=5)
        // to avoid adding 5s per-device overhead to every test teardown — this is purely waste since
        // the exception is caught and teardown continues regardless.
        try {
            tt::llrt::internal_::wait_until_cores_done(dev->id(), dev_msgs::RUN_MSG_GO, dispatch_cores, 200, true);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "Device {}: Exception waiting for dispatch cores to finish during teardown. "
                "Attempting rescue of stuck dispatch cores. Error: {}",
                dev->id(),
                e.what());
            rescue_stuck_dispatch_cores(dev);
        }
        log_info(tt::LogMetal, "[dispatch_teardown] wait_for_dispatch_cores device={} done", dev->id());
    }
}

void DispatchKernelInitializer::rescue_stuck_dispatch_cores(IDevice* device) const {
    // During teardown, dispatch cores (DISPATCH_S / DISPATCH_D) can get stuck spinning on
    // STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX, waiting for a worker completion count
    // that will never arrive because fabric was already torn down.
    //
    // To unblock: write a large value to STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX on each
    // dispatch stream. Writing to SIZE *sets* (not increments) the SPACE_AVAILABLE register,
    // satisfying any pending stream_wrap_gt/stream_wrap_ge comparison in the firmware wait loop.
    //
    // After unblocking, the termination signal path (process_termination_signals) will cleanly
    // shut down the cores.

    const auto& hal = descriptor_->hal();
    CoreType dispatch_core_type = dispatch_core_manager_.get_dispatch_core_type();
    const auto& mem_map = *dispatch_mem_map_[enchantum::to_underlying(dispatch_core_type)];

    const uint32_t overlay_start = hal.get_noc_overlay_start_addr();
    const uint32_t stream_reg_space = hal.get_noc_stream_reg_space_size();
    const uint32_t buf_size_reg_idx = hal.get_noc_stream_remote_dest_buf_size_reg_index();
    const uint32_t num_streams = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;

    // 0xFFFF is the maximum 16-bit value.  The firmware uses 17-bit wrapping arithmetic
    // (stream_wrap_gt / stream_wrap_ge with MEM_WORD_ADDR_WIDTH shift), so any count <= 0xFFFF
    // will be satisfied when SPACE_AVAILABLE is set to 0xFFFF.
    const uint32_t rescue_count = 0xFFFF;

    const auto& termination_cores = dispatch_topology_->get_registered_termination_cores(device->id());
    for (const auto& info : termination_cores) {
        for (uint32_t i = 0; i < num_streams; i++) {
            uint32_t stream_id = mem_map.get_dispatch_stream_index(i);
            uint32_t reg_addr = overlay_start + (stream_id * stream_reg_space) + (buf_size_reg_idx << 2);
            std::vector<uint32_t> val{rescue_count};
            try {
                detail::WriteToDeviceL1(device, info.logical_core, reg_addr, val, info.core_type);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "rescue_stuck_dispatch_cores: Device {} core ({},{}) stream={} write failed: {}",
                    device->id(),
                    info.logical_core.x,
                    info.logical_core.y,
                    stream_id,
                    e.what());
            }
        }
        log_warning(
            tt::LogMetal,
            "rescue_stuck_dispatch_cores: Device {} core ({},{}) injected count={:#x} on {} streams",
            device->id(),
            info.logical_core.x,
            info.logical_core.y,
            rescue_count,
            num_streams);
    }

    // Brief pause to let firmware observe the unblocked stream registers and advance
    // past the wait loop before we send the termination signal.
    if (!termination_cores.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // GAP 3: Post-rescue verification — re-attempt wait_until_cores_done with a short
        // timeout to confirm the firmware actually advanced past the stuck wait loop.
        // If it still hasn't advanced, log a warning but don't TT_THROW — the termination
        // signal path (process_termination_signals) may still succeed.
        auto dispatch_cores = get_virtual_dispatch_cores(device->id());
        try {
            tt::llrt::internal_::wait_until_cores_done(device->id(), dev_msgs::RUN_MSG_GO, dispatch_cores, 100, true);
            log_info(
                tt::LogMetal,
                "rescue_stuck_dispatch_cores: Device {} dispatch cores successfully unblocked after rescue injection",
                device->id());
        } catch (...) {
            log_warning(
                tt::LogMetal,
                "rescue_stuck_dispatch_cores: Device {} dispatch cores may still be stuck after rescue injection "
                "— proceeding to process_termination_signals anyway",
                device->id());
        }
    }
}

void DispatchKernelInitializer::process_termination_signals() const {
    for (auto* dev : devices_) {
        const auto& info = dispatch_topology_->get_registered_termination_cores(dev->id());
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
