// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fd_kernel.hpp"

typedef struct dispatch_config {
    std::optional<tt_cxy_pair> upstream_logical_core;      // Dependant
    std::optional<tt_cxy_pair> downstream_logical_core;    // Dependant
    std::optional<tt_cxy_pair> downstream_s_logical_core;  // Dependant

    std::optional<uint32_t> dispatch_cb_base;  // 0
    std::optional<uint32_t> dispatch_cb_log_page_size;
    std::optional<uint32_t> dispatch_cb_pages;
    std::optional<uint32_t> my_dispatch_cb_sem_id;
    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependant

    std::optional<uint32_t> dispatch_cb_blocks;  // 5
    std::optional<uint32_t> upstream_sync_sem;   // Dependant
    std::optional<uint32_t> command_queue_base_addr;
    std::optional<uint32_t> completion_queue_base_addr;
    std::optional<uint32_t> completion_queue_size;

    std::optional<uint32_t> downstream_cb_base;  // 10, dependent
    std::optional<uint32_t> downstream_cb_size;  // Dependent
    std::optional<uint32_t> my_downstream_cb_sem_id;
    std::optional<uint32_t> downstream_cb_sem_id;  // Dependant

    std::optional<uint32_t> split_dispatch_page_preamble_size;  // 14
    std::optional<uint32_t> split_prefetch;
    std::optional<uint32_t> prefetch_h_noc_xy;                     // Dependent
    std::optional<uint32_t> prefetch_h_local_downstream_sem_addr;  // Dependent
    std::optional<uint32_t> prefetch_h_max_credits;

    std::optional<uint32_t> packed_write_max_unicast_sub_cmds;  // 19
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;
    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;

    std::optional<uint32_t> host_completion_q_wr_ptr;  // 26
    std::optional<uint32_t> dev_completion_q_wr_ptr;
    std::optional<uint32_t> dev_completion_q_rd_ptr;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;
} dispatch_config_t;

class DispatchKernel : public FDKernel {
public:
    DispatchKernel(
        int node_id,
        chip_id_t device_id,
        chip_id_t servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool h_variant,
        bool d_variant) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {
        config.is_h_variant = h_variant;
        config.is_d_variant = d_variant;
    }
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    const dispatch_config_t& GetConfig() { return this->config; }

private:
    dispatch_config_t config;
};
