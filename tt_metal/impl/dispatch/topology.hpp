// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <device.hpp>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core_coord.hpp"
#include "data_types.hpp"
#include "tt-metalium/program.hpp"
#include "tt_metal/impl/dispatch/kernel_config/fd_kernel.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_common.hpp"

namespace tt {
class Cluster;
}

namespace tt::tt_metal::detail {
class ProgramCompileGroup;
}

namespace tt::tt_metal {

class IDevice;
class DeviceManager;
class DispatchMemMap;
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

// TODO: Use the correct ContextDescriptor once that PR is merged
struct ContextDescriptor {
    const tt::Cluster& cluster;
    dispatch_core_manager& dispatch_core_manager_;
    const DispatchMemMap& dispatch_mem_map;
    DeviceManager* device_manager;
};

class DispatchTopology {
public:
    explicit DispatchTopology(const ContextDescriptor& descriptor);
    ~DispatchTopology();

    void populate_fd_kernels(const std::vector<IDevice*>& devices, uint32_t num_hw_cqs);
    void populate_fd_kernels(const std::set<ChipId>& device_ids, uint32_t num_hw_cqs);
    void populate_fd_kernels(const std::vector<DispatchKernelNode>& nodes);

    void populate_cq_static_args(IDevice* device);
    void create_cq_program(IDevice* device);
    void compile_cq_programs();
    std::unique_ptr<Program> get_compiled_cq_program(IDevice* device);
    void configure_dispatch_cores(IDevice* device);

    const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(ChipId dev_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(ChipId dev_id) const;
    const std::unordered_set<TerminationInfo>& get_registered_termination_cores(ChipId dev_id);

    void reset();

private:
    std::vector<DispatchKernelNode> generate_nodes(const std::set<ChipId>& device_ids, uint32_t num_hw_cqs) const;

    ContextDescriptor context_;
    std::vector<FDKernel*> node_id_to_kernel_;
    std::unique_ptr<detail::ProgramCompileGroup> command_queue_compile_group_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> dispatch_cores_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> routing_cores_;
    mutable std::unordered_map<ChipId, std::unordered_set<CoreCoord>> empty_cores_;
    std::unordered_map<ChipId, std::unordered_set<TerminationInfo>> termination_info_;
};

}  // namespace tt::tt_metal
