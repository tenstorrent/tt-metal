// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch/dispatch_mem_map.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch/topology.hpp"
#include "firmware_initializer.hpp"

namespace tt::tt_metal {

class DispatchKernelInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key = InitializerKey::Dispatch;

    DispatchKernelInitializer(
        std::shared_ptr<const ContextDescriptor> descriptor,
        dispatch_core_manager& dispatch_core_manager,
        DeviceManager* device_manager,
        const GetControlPlaneFn& get_control_plane = {},
        const GetDispatchQueryManagerFn& get_dispatch_query_manager = {},
        const GetMaxNumEthCoresFn& get_max_num_eth_cores = {},
        const GetReadsDispatchCoresFn& get_reads_dispatch_cores = {});

    void init(const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& init_done) override;
    void configure() override;
    void teardown(std::unordered_set<InitializerKey>& init_done) override;
    // Returns true if fast dispatch is enabled and has been configured
    bool is_initialized() const override;
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(ChipId dev_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(ChipId dev_id) const;

    // Populate the FD kernels only. This will cause dispatch cores to be allocated in dispatch_core_manager
    void populate_fd_kernels_only(const std::vector<Device*>& devices);

private:
    void compile_dispatch_kernels();

    void init_device_command_queues();

    void terminate_command_queues();

    void wait_for_dispatch_cores() const;

    void process_termination_signals() const;

    bool using_fast_dispatch() const;

    std::vector<Device*> devices_;
    bool initialized_ = false;
    std::unique_ptr<tt::tt_metal::DispatchTopology> dispatch_topology_;
    std::array<std::unique_ptr<DispatchMemMap>, static_cast<size_t>(CoreType::COUNT)> dispatch_mem_map_;
    dispatch_core_manager& dispatch_core_manager_;
    DeviceManager* device_manager_ = nullptr;
    GetControlPlaneFn get_control_plane_;
    GetDispatchQueryManagerFn get_dispatch_query_manager_;
    GetMaxNumEthCoresFn get_max_num_eth_cores_;
    GetReadsDispatchCoresFn get_reads_dispatch_cores_;
};

}  // namespace tt::tt_metal
