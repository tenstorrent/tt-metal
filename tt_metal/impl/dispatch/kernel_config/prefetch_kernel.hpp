// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fd_kernel.hpp"

typedef struct prefetch_config {
    std::optional<tt_cxy_pair> upstream_logical_core;      // Dependant
    std::optional<tt_cxy_pair> downstream_logical_core;    // Dependant
    std::optional<tt_cxy_pair> downstream_s_logical_core;  // Dependant

    std::optional<uint32_t> downstream_cb_base;  // Dependent
    std::optional<uint32_t> downstream_cb_log_page_size;
    std::optional<uint32_t> downstream_cb_pages;
    std::optional<uint32_t> my_downstream_cb_sem_id;
    std::optional<uint32_t> downstream_cb_sem_id;  // Dependant

    std::optional<uint32_t> pcie_base;
    std::optional<uint32_t> pcie_size;
    std::optional<uint32_t> prefetch_q_base;
    std::optional<uint32_t> prefetch_q_size;
    std::optional<uint32_t> prefetch_q_rd_ptr_addr;
    std::optional<uint32_t> prefetch_q_pcie_rd_ptr_addr;

    std::optional<uint32_t> cmddat_q_base;
    std::optional<uint32_t> cmddat_q_size;

    // Used for prefetch_h
    std::optional<uint32_t> scratch_db_base;
    std::optional<uint32_t> scratch_db_size;
    std::optional<uint32_t> downstream_sync_sem_id;

    // Used for prefetch_d
    std::optional<uint32_t> cmddat_q_pages;
    std::optional<uint32_t> my_upstream_cb_sem_id;
    std::optional<uint32_t> upstream_cb_sem_id;  // Dependant
    std::optional<uint32_t> cmddat_q_log_page_size;
    std::optional<uint32_t> cmddat_q_blocks;

    // Used for prefetch_d <--> dispatch_s data path
    std::optional<uint32_t> dispatch_s_buffer_base;
    std::optional<uint32_t> my_dispatch_s_cb_sem_id;
    std::optional<uint32_t> downstream_dispatch_s_cb_sem_id;  // Dependant
    std::optional<uint32_t> dispatch_s_buffer_size;
    std::optional<uint32_t> dispatch_s_cb_log_page_size;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;
} prefetch_config_t;

class PrefetchKernel : public FDKernel {
public:
    PrefetchKernel(
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
    const prefetch_config_t& GetConfig() { return this->config; }

private:
    prefetch_config_t config;
};
