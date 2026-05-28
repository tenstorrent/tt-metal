// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_kernel_initializer.hpp"

#include <cstdio>
#include <ctime>
#include <string>
#include <unistd.h>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <tt_metal.hpp>
#include <cstdint>
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
    ChipId device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms);
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
        CoreType::WORKER, descriptor_->num_cqs(), descriptor_->hal(), is_galaxy_cluster, descriptor_->rtoptions());
    dispatch_mem_map_[enchantum::to_underlying(CoreType::ETH)] = std::make_unique<tt::tt_metal::DispatchMemMap>(
        CoreType::ETH, descriptor_->num_cqs(), descriptor_->hal(), is_galaxy_cluster, descriptor_->rtoptions());

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

    terminate_command_queues();
    wait_for_dispatch_cores();

    process_termination_signals();

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
            for (std::uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                std::uint32_t mmio_controlled_device_id = tunnel[ts];
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
            for (std::uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                std::uint32_t mmio_controlled_device_id = tunnel[ts];
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
    // #region agent log
    {
        std::FILE* f = std::fopen("/data/rsong/tt-metal2/.cursor/debug-ae7d0a.log", "a");
        if (f) {
            std::string visible_ids;
            for (auto* d : devices_) {
                visible_ids += std::to_string(d->id()) + ",";
            }
            const char* env_visible = std::getenv("TT_VISIBLE_DEVICES");
            const char* env_mesh_id = std::getenv("TT_MESH_ID");
            const char* env_mesh_rank = std::getenv("TT_MESH_HOST_RANK");
            std::fprintf(
                f,
                "{\"sessionId\":\"ae7d0a\",\"hypothesisId\":\"H_DUAL_INIT_CONFLICT\","
                "\"location\":\"dispatch_kernel_initializer.cpp:init_device_command_queues\","
                "\"message\":\"INIT_DCQ_START\",\"data\":{\"pid\":%d,\"visible_devices_actual\":\"%s\","
                "\"TT_VISIBLE_DEVICES_env\":\"%s\",\"TT_MESH_ID\":\"%s\",\"TT_MESH_HOST_RANK\":\"%s\"},\"timestamp\":%"
                "ld}\n",
                (int)getpid(),
                visible_ids.c_str(),
                env_visible ? env_visible : "(unset)",
                env_mesh_id ? env_mesh_id : "(unset)",
                env_mesh_rank ? env_mesh_rank : "(unset)",
                (long)std::time(nullptr));
            std::fclose(f);
        }
    }
    // #endregion

    std::vector<std::shared_future<void>> events;
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }
        ChipId mmio_device_id = dev->id();
        events.emplace_back(detail::async([this, dev, mmio_device_id]() {
            auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(mmio_device_id);
            dev->init_command_queue_device_with_topology(dispatch_topology_.get());
            // #region agent log
            {
                std::FILE* f = std::fopen("/data/rsong/tt-metal2/.cursor/debug-ae7d0a.log", "a");
                if (f) {
                    std::fprintf(
                        f,
                        "{\"sessionId\":\"ae7d0a\",\"hypothesisId\":\"H_DISPATCH_NOT_LAUNCHED_TUNNEL\","
                        "\"location\":\"dispatch_kernel_initializer.cpp:init_device_command_queues\","
                        "\"message\":\"INIT_DCQ_MMIO\",\"data\":{\"pid\":%d,\"device_id\":%u,\"is_tunnel\":0},"
                        "\"timestamp\":%ld}\n",
                        (int)getpid(),
                        (unsigned)dev->id(),
                        (long)std::time(nullptr));
                    std::fclose(f);
                }
            }
            // #endregion
            log_debug(tt::LogMetal, "Command Queue initialized on Device {}", dev->id());
            for (const auto& tunnel : tunnels_from_mmio) {
                for (std::uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    std::uint32_t mmio_controlled_device_id = tunnel[ts];
                    auto it = std::find_if(devices_.begin(), devices_.end(), [&](IDevice* d) {
                        return d->id() == mmio_controlled_device_id;
                    });
                    // #region agent log
                    {
                        std::FILE* f = std::fopen("/data/rsong/tt-metal2/.cursor/debug-ae7d0a.log", "a");
                        if (f) {
                            std::fprintf(
                                f,
                                "{\"sessionId\":\"ae7d0a\",\"hypothesisId\":\"H_DISPATCH_NOT_LAUNCHED_TUNNEL\","
                                "\"location\":\"dispatch_kernel_initializer.cpp:init_device_command_queues\","
                                "\"message\":\"TUNNEL_CHECK\",\"data\":{\"pid\":%d,\"mmio_device_id\":%u,"
                                "\"tunnel_chip\":%u,\"found_in_devices_\":%d},\"timestamp\":%ld}\n",
                                (int)getpid(),
                                (unsigned)mmio_device_id,
                                (unsigned)mmio_controlled_device_id,
                                (int)(it != devices_.end()),
                                (long)std::time(nullptr));
                            std::fclose(f);
                        }
                    }
                    // #endregion
                    if (it != devices_.end()) {
                        (*it)->init_command_queue_device_with_topology(dispatch_topology_.get());
                        // #region agent log
                        {
                            std::FILE* f = std::fopen("/data/rsong/tt-metal2/.cursor/debug-ae7d0a.log", "a");
                            if (f) {
                                std::fprintf(
                                    f,
                                    "{\"sessionId\":\"ae7d0a\",\"hypothesisId\":\"H_DISPATCH_NOT_LAUNCHED_TUNNEL\","
                                    "\"location\":\"dispatch_kernel_initializer.cpp:init_device_command_queues\","
                                    "\"message\":\"INIT_DCQ_TUNNEL\",\"data\":{\"pid\":%d,\"device_id\":%u,\"mmio_"
                                    "device_id\":%u,\"is_tunnel\":1},\"timestamp\":%ld}\n",
                                    (int)getpid(),
                                    (unsigned)mmio_controlled_device_id,
                                    (unsigned)mmio_device_id,
                                    (long)std::time(nullptr));
                                std::fclose(f);
                            }
                        }
                        // #endregion
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
        const auto& info = dispatch_topology_->get_registered_termination_cores(dev->id());
        for (const auto& core_to_terminate : info) {
            std::vector<std::uint32_t> val{core_to_terminate.val};
            detail::WriteToDeviceL1(
                dev, core_to_terminate.logical_core, core_to_terminate.address, val, core_to_terminate.core_type);
        }
        cluster_.l1_barrier(dev->id());
    }
}

bool DispatchKernelInitializer::using_fast_dispatch() const { return rtoptions_.get_fast_dispatch(); }

}  // namespace tt::tt_metal
