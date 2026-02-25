// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

#include "core_coord.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "fd_kernel.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "impl/context/context_descriptor.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {

struct dispatch_static_config_t {
    std::optional<uint32_t> dispatch_cb_base;  // 0
    std::optional<uint32_t> dispatch_cb_log_page_size;
    std::optional<uint32_t> dispatch_cb_pages;
    std::optional<uint32_t> my_dispatch_cb_sem_id;

    std::optional<uint32_t> dispatch_cb_blocks;  // 5
    std::optional<uint32_t> command_queue_base_addr;
    std::optional<uint32_t> completion_queue_base_addr;
    std::optional<uint32_t> completion_queue_size;

    std::optional<uint32_t> my_downstream_cb_sem_id;

    std::optional<uint32_t> split_dispatch_page_preamble_size;  // 14
    std::optional<uint32_t> prefetch_h_max_credits;             // Used if split_prefetch is true

    std::optional<uint32_t> packed_write_max_unicast_sub_cmds;  // 19
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;
    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;
    std::optional<uint32_t> first_stream_used;

    std::optional<uint32_t> host_completion_q_wr_ptr;  // 26
    std::optional<uint32_t> dev_completion_q_wr_ptr;
    std::optional<uint32_t> dev_completion_q_rd_ptr;
    std::optional<uint32_t> dev_dispatch_progress_ptr;

    std::optional<uint32_t> fabric_header_rb_base;
    std::optional<uint32_t> fabric_header_rb_entries;
    std::optional<uint32_t> my_fabric_sync_status_addr;
    std::optional<bool> is_2d_fabric;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;

    // Offsets of runtime args
    std::optional<uint32_t> offsetof_my_dev_id;
    std::optional<uint32_t> offsetof_to_dev_id;
    std::optional<uint32_t> offsetof_router_direction;
};

struct dispatch_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;      // Dependent
    std::optional<tt_cxy_pair> downstream_logical_core;    // Dependent
    std::optional<tt_cxy_pair> downstream_s_logical_core;  // Dependent

    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependent

    std::optional<uint32_t> upstream_sync_sem;  // Dependent

    std::optional<uint32_t> downstream_cb_base;    // 10, dependent
    std::optional<uint32_t> downstream_cb_size;    // Dependent
    std::optional<uint32_t> downstream_cb_sem_id;  // Dependent

    std::optional<uint32_t> split_prefetch;                        // If upstream is NOT a prefetch_HD
    std::optional<uint32_t> prefetch_h_noc_xy;                     // Dependent. Used if split_prefetch is true
    std::optional<uint32_t> prefetch_h_local_downstream_sem_addr;  // Dependent. Used if split_prefetch is true

    std::optional<uint32_t> num_hops;

    tt::tt_metal::relay_mux_client_config fabric_mux_client_config;

    std::optional<uint32_t> my_dev_id;
    std::optional<uint32_t> ew_dim;
    std::optional<uint32_t> to_mesh_id;
    std::optional<uint32_t> to_dev_id;
    std::optional<uint32_t> router_direction;
};

class DispatchKernel : public FDKernel {
public:
    DispatchKernel(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool h_variant,
        bool d_variant,
        const ContextDescriptor& descriptor,
        dispatch_core_manager& dispatch_core_manager,
        const GetControlPlaneFn& get_control_plane = {},
        const GetDispatchQueryManagerFn& get_dispatch_query_manager = {},
        const GetMaxNumEthCoresFn& get_max_num_eth_cores = {},
        const GetReadsDispatchCoresFn& get_reads_dispatch_cores = {});

    void CreateKernel() override;

    void GenerateStaticConfigs() override;

    void GenerateDependentConfigs() override;

    void InitializeRuntimeArgsValues() override;

    void ConfigureCore() override;

    uint32_t GetDispatchBufferSize() const {
        return (1 << static_config_.dispatch_cb_log_page_size.value()) * static_config_.dispatch_cb_pages.value();
    }
    const dispatch_static_config_t& GetStaticConfig() { return static_config_; }

private:
    dispatch_static_config_t static_config_;
    dispatch_dependent_config_t dependent_config_;
    FDKernelEdmConnectionAttributes edm_connection_attributes_;

    bool is_hd() const { return static_config_.is_h_variant.value() && static_config_.is_d_variant.value(); }
};

}  // namespace tt::tt_metal
