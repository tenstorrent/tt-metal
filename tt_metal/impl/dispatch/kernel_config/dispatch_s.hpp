// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include <optional>

#include "fd_kernel.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_xy_pair.h>

namespace tt {
namespace tt_metal {

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
};

struct dispatch_s_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;     // Dependant
    std::optional<tt_cxy_pair> downstream_logical_core;   // Dependant
    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependent
};

class DispatchSKernel : public FDKernel {
public:
    DispatchSKernel(
        int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);
        this->logical_core_ = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().dispatcher_s_core(
            device_id, channel, cq_id_);
        this->kernel_type_ = FDKernelType::DISPATCH;
    }
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    const dispatch_s_static_config_t& GetStaticConfig() { return static_config_; }

private:
    dispatch_s_static_config_t static_config_;
    dispatch_s_dependent_config_t dependent_config_;
};

}  // namespace tt_metal
}  // namespace tt
