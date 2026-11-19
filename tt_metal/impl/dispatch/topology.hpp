// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <device.hpp>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "device/device_impl.hpp"
#include "tt-metalium/program.hpp"
#include "tt_metal/impl/dispatch/kernel_config/fd_kernel.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_common.hpp"
#include "tt_metal/impl/context/context_descriptor.hpp"

namespace tt::tt_fabric {
class ControlPlane;
}

namespace tt {
class Cluster;
}

namespace tt::tt_metal::detail {
class ProgramCompileGroup;
}

namespace tt::tt_metal::experimental {
class KernelConfigBufferImpl;
}  // namespace tt::tt_metal::experimental

namespace tt::tt_metal {

class IDevice;
class DeviceManager;
class dispatch_core_manager;
enum DispatchWorkerType : uint32_t;

// NOC ID used by dispatch kernels to communicate with downstream cores. This parameter
// is required when setting up Command Queue objects on host.
constexpr NOC k_dispatch_downstream_noc = NOC::NOC_0;

struct DispatchKernelNode {
    int id;
    ChipId device_id;                // Device that this kernel is located on
    ChipId servicing_device_id;      // Remote device that this kernel services, used for kernels on MMIO
    uint8_t cq_id;                   // CQ this kernel implements
    DispatchWorkerType kernel_type;  // Type of dispatch kernel this is
    std::vector<int> upstream_ids;   // Upstream dispatch kernels
    std::vector<int> downstream_ids;  // Downstream dispatch kernels
    noc_selection_t noc_selection;    // NOC selection
    int tunnel_index{-1};             // Tunnel index
};

class DispatchTopology {
public:
    explicit DispatchTopology(
        const ContextDescriptor& descriptor,
        dispatch_core_manager& dispatch_core_manager,
        DeviceManager* device_manager,
        const GetControlPlaneFn& get_control_plane = {},
        const GetDispatchQueryManagerFn& get_dispatch_query_manager = {},
        const GetReadsDispatchCoresFn& get_reads_dispatch_cores = {});
    ~DispatchTopology();

    void populate_fd_kernels(const std::vector<Device*>& devices, uint32_t num_hw_cqs);
    void populate_fd_kernels(const std::set<ChipId>& device_ids, uint32_t num_hw_cqs);
    void populate_fd_kernels(const std::vector<DispatchKernelNode>& nodes);

    void populate_cq_static_args(Device* device);
    void create_cq_program(Device* device);
    void compile_cq_programs();
    std::unique_ptr<Program> get_compiled_cq_program(Device* device);
    // The kernel config buffer the command queue programs run from, or null when they run from the KERNEL_CONFIG
    // window. compile_cq_programs allocates it when a program's Tensix block does not fit the window.
    const experimental::KernelConfigBufferImpl* kernel_config_buffer() const { return kernel_config_buffer_.get(); }
    // Frees that buffer and the dispatch kernels' L1 reserved for the session. Only once they have exited.
    void release_dispatch_l1();
    void configure_dispatch_cores(Device* device);

    const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(ChipId dev_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(ChipId dev_id) const;
    // Termination signals of the device's kernels of `kernel_type` (DISPATCH or ROUTING).
    const std::unordered_set<TerminationInfo>& get_registered_termination_cores(
        ChipId dev_id, FDKernelType kernel_type) const;

    void reset();

private:
    std::vector<DispatchKernelNode> generate_nodes(const std::set<ChipId>& device_ids, uint32_t num_hw_cqs) const;

    const ContextDescriptor& descriptor_;
    dispatch_core_manager& dispatch_core_manager_;
    DeviceManager* device_manager_;
    GetControlPlaneFn get_control_plane_;
    GetDispatchQueryManagerFn get_dispatch_query_manager_;
    GetReadsDispatchCoresFn get_reads_dispatch_cores_;
    std::vector<FDKernel*> node_id_to_kernel_;
    std::unique_ptr<detail::ProgramCompileGroup> command_queue_compile_group_;
    std::shared_ptr<experimental::KernelConfigBufferImpl> kernel_config_buffer_;
    // The dispatch kernels' own L1, booked on each dispatch core's per-core allocator for the session.
    struct DispatchL1Reservation {
        IDevice* device;
        CoreCoord core;
        uint32_t address;
    };
    std::vector<DispatchL1Reservation> dispatch_l1_reservations_;
    // Reserves the dispatch kernels' L1 for the session and, when `programs` outgrow the KERNEL_CONFIG window,
    // allocates the kernel config buffer they run from.
    void reserve_dispatch_l1(const std::vector<std::pair<IDevice*, Program*>>& programs);
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> dispatch_cores_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> routing_cores_;
    std::unordered_map<ChipId, std::unordered_set<TerminationInfo>> dispatch_termination_info_;
    std::unordered_map<ChipId, std::unordered_set<TerminationInfo>> routing_termination_info_;
    const std::unordered_set<CoreCoord> empty_container_;
    const std::unordered_set<TerminationInfo> empty_termination_info_container_;
};

}  // namespace tt::tt_metal
