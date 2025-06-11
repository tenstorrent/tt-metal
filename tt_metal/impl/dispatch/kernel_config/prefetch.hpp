// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

#include "core_coord.hpp"
#include "fd_kernel.hpp"
#include "mesh_graph.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include "dispatch/kernel_config/relay_mux.hpp"

namespace tt {
namespace tt_metal {

struct prefetch_static_config_t {
    std::optional<uint32_t> my_downstream_cb_sem_id;

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
    std::optional<uint32_t> ringbuffer_size;

    // Used for prefetch_d
    std::optional<uint32_t> cmddat_q_pages;
    std::optional<uint32_t> my_upstream_cb_sem_id;
    std::optional<uint32_t> cmddat_q_log_page_size;
    std::optional<uint32_t> cmddat_q_blocks;

    // Used for prefetch_d <--> dispatch_s data path
    std::optional<uint32_t> dispatch_s_buffer_base;
    std::optional<uint32_t> my_dispatch_s_cb_sem_id;
    std::optional<uint32_t> dispatch_s_buffer_size;
    std::optional<uint32_t> dispatch_s_cb_log_page_size;

    std::optional<uint32_t> fabric_header_rb_base;
    std::optional<uint32_t> fabric_header_rb_entries;
    std::optional<uint32_t> my_fabric_sync_status_addr;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;
};

struct prefetch_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;
    std::optional<tt_cxy_pair> downstream_logical_core;
    std::optional<tt_cxy_pair> downstream_s_logical_core;

    std::optional<uint32_t> downstream_cb_base;
    std::optional<uint32_t> downstream_cb_log_page_size;
    std::optional<uint32_t> downstream_cb_pages;
    std::optional<uint32_t> downstream_cb_sem_id;

    std::optional<uint32_t> upstream_cb_sem_id;

    std::optional<uint32_t> downstream_dispatch_s_cb_sem_id;

    std::optional<uint32_t> num_hops;

    tt::tt_metal::relay_mux_client_config fabric_mux_client_config;
};

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
        auto& core_manager = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager();  // Not thread safe
        static_config_.is_h_variant = h_variant;
        static_config_.is_d_variant = d_variant;
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);

        if (h_variant && d_variant) {
            this->logical_core_ = core_manager.prefetcher_core(device_id, channel, cq_id);
        } else if (h_variant) {
            channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(
                servicing_device_id);
            this->logical_core_ = core_manager.prefetcher_core(servicing_device_id, channel, cq_id);
        } else if (d_variant) {
            this->logical_core_ = core_manager.prefetcher_d_core(device_id, channel, cq_id);
        }
        this->kernel_type_ = FDKernelType::DISPATCH;
    }

    void CreateKernel() override;

    void GenerateStaticConfigs() override;

    void GenerateDependentConfigs() override;

    void ConfigureCore() override;

    const prefetch_static_config_t& GetStaticConfig() { return static_config_; }

private:
    prefetch_static_config_t static_config_;
    prefetch_dependent_config_t dependent_config_;
    FDKernelEdmConnectionAttributes edm_connection_attributes_;
};

}  // namespace tt_metal
}  // namespace tt
