// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include <optional>

#include "fd_kernel.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {

struct dispatch_s_static_config_t {
    std::optional<uint32_t> cb_base;
    std::optional<uint32_t> cb_log_page_size;
    std::optional<uint32_t> cb_size;
    std::optional<uint32_t> my_dispatch_cb_sem_id;
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;

    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;
    std::optional<uint32_t> first_stream_used;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;

    // D2H socket configuration for realtime dispatch telemetry
    std::optional<uint32_t> d2h_pcie_xy_enc;
    std::optional<uint32_t> d2h_data_addr_lo;
    std::optional<uint32_t> d2h_data_addr_hi;
    std::optional<uint32_t> d2h_bytes_sent_addr_lo;
    std::optional<uint32_t> d2h_bytes_sent_addr_hi;
    std::optional<uint32_t> d2h_bytes_acked_addr_lo;
    std::optional<uint32_t> d2h_bytes_acked_addr_hi;
    std::optional<uint32_t> d2h_fifo_size;
    std::optional<uint32_t> d2h_enabled;
};

struct dispatch_s_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;     // Dependent
    std::optional<tt_cxy_pair> downstream_logical_core;   // Dependent
    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependent
};

class DispatchSKernel : public FDKernel {
public:
    DispatchSKernel(
        int node_id, ChipId device_id, ChipId servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection);

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
