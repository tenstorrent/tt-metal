// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include <optional>

#include "fd_kernel.hpp"
#include "impl/context/context_descriptor.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {

struct dispatch_s_static_config_t {
    std::optional<uint32_t> cb_base;
    std::optional<uint32_t> cb_log_page_size;
    std::optional<uint32_t> cb_size;
    std::optional<uint32_t> my_dispatch_cb_sem_id;
    std::optional<uint32_t> dispatch_d_shutdown_sem_id;
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;

    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;
    std::optional<uint32_t> first_stream_used;
    std::optional<uint32_t> completion_counter_base;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;

    // Dispatch-core-local L1 address of the realtime_profiler_msg_t block (includes the
    // program-id handoff FIFO consumed by this kernel). Assigned by DispatchMemMap via
    // CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG. Must match the value passed to the
    // co-located DispatchKernel and to the RT-profiler core kernels.
    std::optional<uint32_t> realtime_profiler_msg_addr;

    std::optional<uint32_t> dispatch_telemetry_addr;
    std::optional<bool> dispatch_telemetry_disabled;
    std::optional<uint32_t> dispatch_telemetry_control_addr;

    // Configuration for DEVICE_PRINT dispatch. Populated only when the dprint server
    // exists and dispatch_s_enabled() is true. enabled stays 0 otherwise and the kernel
    // compiles the feature out via #if DEVICE_PRINT_DISPATCH_ENABLED.
    std::optional<uint32_t> device_print_dispatch_enabled;
    std::optional<uint32_t> device_print_noc_locations_addr;
    std::optional<uint32_t> device_print_noc_locations_count;
    std::optional<uint32_t> device_print_l1_cache_addr;
    std::optional<uint32_t> device_print_l1_cache_size;
    std::optional<uint32_t> device_print_dram_x;
    std::optional<uint32_t> device_print_dram_y;
    std::optional<uint64_t> device_print_dram_rw_ptrs;
    std::optional<uint64_t> device_print_dram_buf_addr;
    std::optional<uint32_t> device_print_dram_buf_size;
    std::optional<uint64_t> device_print_cycles_for_stall;
    std::optional<uint64_t> device_print_cycles_for_full;
};

struct dispatch_s_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;     // Dependent
    std::optional<tt_cxy_pair> downstream_logical_core;   // Dependent
    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependent
};

class DispatchSKernel : public FDKernel {
public:
    DispatchSKernel(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        const ContextDescriptor& descriptor,
        dispatch_core_manager& dispatch_core_manager,
        const GetControlPlaneFn& get_control_plane = {},
        const GetDispatchQueryManagerFn& get_dispatch_query_manager = {},
        const GetMaxNumEthCoresFn& get_max_num_eth_cores = {},
        const GetReadsDispatchCoresFn& get_reads_dispatch_cores = {});

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    const dispatch_s_static_config_t& GetStaticConfig() { return static_config_; }

private:
    dispatch_s_static_config_t static_config_;
    dispatch_s_dependent_config_t dependent_config_;
};

}  // namespace tt::tt_metal
