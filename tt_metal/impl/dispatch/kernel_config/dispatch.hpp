// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

#include "assert.hpp"
#include "core_coord.hpp"
#include "impl/context/metal_context.hpp"
#include "fd_kernel.hpp"
#include "mesh_graph.hpp"
#include "system_memory_manager.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include <umd/device/tt_xy_pair.h>

typedef struct dispatch_static_config {
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

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;

    // Populated if fabric is being used to talk to downstream
    std::optional<uint32_t> client_interface_addr;
} dispatch_static_config_t;

typedef struct dispatch_dependent_config {
    std::optional<tt_cxy_pair> upstream_logical_core;      // Dependant
    std::optional<tt_cxy_pair> downstream_logical_core;    // Dependant
    std::optional<tt_cxy_pair> downstream_s_logical_core;  // Dependant

    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependant

    std::optional<uint32_t> upstream_sync_sem;  // Dependant

    std::optional<uint32_t> downstream_cb_base;    // 10, dependent
    std::optional<uint32_t> downstream_cb_size;    // Dependent
    std::optional<uint32_t> downstream_cb_sem_id;  // Dependant

    std::optional<uint32_t> split_prefetch;                        // If upstream is NOT a prefetch_HD
    std::optional<uint32_t> prefetch_h_noc_xy;                     // Dependent. Used if split_prefetch is true
    std::optional<uint32_t> prefetch_h_local_downstream_sem_addr;  // Dependent. Used if split_prefetch is true

    // Populated if fabric is being used to talk to downstream
    std::optional<uint32_t> fabric_router_noc_xy;
    std::optional<uint32_t> upstream_mesh_id;
    std::optional<uint32_t> upstream_chip_id;
    std::optional<uint32_t> downstream_mesh_id;
    std::optional<uint32_t> downstream_chip_id;
} dispatch_dependent_config_t;

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
        auto& core_manager = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager();  // Not thread safe
        TT_FATAL(
            noc_selection.downstream_noc == tt::tt_metal::k_dispatch_downstream_noc,
            "Invalid downstream NOC specified for Dispatcher kernel");
        TT_FATAL(
            noc_selection.upstream_noc != noc_selection.downstream_noc,
            "Dispatcher kernel cannot have identical upstream and downstream NOCs.");
        static_config_.is_h_variant = h_variant;
        static_config_.is_d_variant = d_variant;
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);
        if (h_variant && d_variant) {
            this->logical_core_ = core_manager.dispatcher_core(device_id, channel, cq_id);
        } else if (h_variant) {
            channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(
                servicing_device_id);
            this->logical_core_ = core_manager.dispatcher_core(servicing_device_id, channel, cq_id);
        } else if (d_variant) {
            this->logical_core_ = core_manager.dispatcher_d_core(device_id, channel, cq_id);
        }
    }

    void CreateKernel() override;

    void GenerateStaticConfigs() override;

    void GenerateDependentConfigs() override;

    void ConfigureCore() override;

    void UpdateArgsForFabric(
        const CoreCoord& fabric_router,
        tt::tt_fabric::mesh_id_t src_mesh_id,
        chip_id_t src_chip_id,
        tt::tt_fabric::mesh_id_t dst_mesh_id,
        chip_id_t dst_chip_id) override;

    uint32_t GetDispatchBufferSize() const {
        return (1 << static_config_.dispatch_cb_log_page_size.value()) * static_config_.dispatch_cb_pages.value();
    }
    const dispatch_static_config_t& GetStaticConfig() { return static_config_; }

private:
    dispatch_static_config_t static_config_;
    dispatch_dependent_config_t dependent_config_;
};
