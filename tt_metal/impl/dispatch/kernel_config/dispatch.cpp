// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>
#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "assert.hpp"
#include "dispatch/command_queue_common.hpp"
#include "device.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch_s.hpp"
#include "hal_types.hpp"
#include "prefetch.hpp"
#include "impl/context/metal_context.hpp"
#include "rtoptions.hpp"
#include <umd/device/types/xy_pair.h>
#include "dispatch/system_memory_manager.hpp"

#include "tt_metal/api/tt-metalium/device_pool.hpp"

using namespace tt::tt_metal;

void DispatchKernel::GenerateStaticConfigs() {
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());
    uint8_t cq_id_ = this->cq_id_;
    auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());

    // May be zero if not using dispatch on fabric
    static_config_.fabric_header_rb_base =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::FABRIC_HEADER_RB);
    static_config_.fabric_header_rb_entries = tt::tt_metal::DispatchSettings::FABRIC_HEADER_RB_ENTRIES;
    static_config_.my_fabric_sync_status_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS);

    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device_->sysmem_manager().get_completion_queue_size(cq_id_);

        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        static_config_.dispatch_cb_log_page_size = DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = command_queue_start_addr;
        static_config_.completion_queue_base_addr = completion_queue_start_addr;
        static_config_.completion_queue_size = completion_queue_size;

        static_config_.my_downstream_cb_sem_id = 0;  // unused

        static_config_.split_dispatch_page_preamble_size = 0;        // unused
        static_config_.prefetch_h_max_credits = 0;                   // unused prefetch_downstream_buffer_pages

        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
        static_config_.max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
        static_config_.max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        static_config_.mcast_go_signal_addr =
            MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        static_config_.unicast_go_signal_addr =
            (MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? MetalContext::instance().hal().get_dev_addr(
                      HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        static_config_.distributed_dispatcher =
            MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher();
        static_config_.first_stream_used = my_dispatch_constants.get_dispatch_stream_index(0);

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else if (static_config_.is_h_variant.value()) {
        // DISPATCH_H services a remote chip, and so has a different channel
        channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(servicing_device_id_);
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device_->sysmem_manager().get_completion_queue_size(cq_id_);

        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        static_config_.dispatch_cb_log_page_size = DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = command_queue_start_addr;
        static_config_.completion_queue_base_addr = completion_queue_start_addr;
        static_config_.completion_queue_size = completion_queue_size;

        static_config_.my_downstream_cb_sem_id = 0;  // Unused

        static_config_.split_dispatch_page_preamble_size = 0;
        static_config_.prefetch_h_max_credits = my_dispatch_constants.prefetch_d_buffer_pages();
        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr = 0;       // Unused
        static_config_.max_num_worker_sems = 1;                 // Used for array sizing, set to 1 even if unused
        static_config_.max_num_go_signal_noc_data_entries = 1;  // Used for array sizing, sset to 1 even if unused
        static_config_.mcast_go_signal_addr = 0;                // Unused
        static_config_.unicast_go_signal_addr = 0;              // Unused
        static_config_.distributed_dispatcher = 0;              // Unused
        static_config_.first_stream_used = 0;                   // Unused

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else if (static_config_.is_d_variant.value()) {
        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        static_config_.dispatch_cb_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = 0;  // These are unused for DISPATCH_D
        static_config_.completion_queue_base_addr = 0;
        static_config_.completion_queue_size = 0;

        static_config_.split_dispatch_page_preamble_size = 0;
        static_config_.prefetch_h_max_credits = my_dispatch_constants.prefetch_d_buffer_pages();
        static_config_.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program_, logical_core_, my_dispatch_constants.prefetch_d_buffer_pages(), GetCoreType());

        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
        static_config_.max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
        static_config_.max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        static_config_.mcast_go_signal_addr =
            MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        static_config_.unicast_go_signal_addr =
            (MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? MetalContext::instance().hal().get_dev_addr(
                      HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        static_config_.distributed_dispatcher =
            MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher();
        static_config_.first_stream_used = my_dispatch_constants.get_dispatch_stream_index(0);

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }

    if (!is_hd()) {
        create_edm_connection_sems(edm_connection_attributes_);
        static_config_.is_2d_fabric =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().get_fabric_topology() ==
            tt_fabric::Topology::Mesh;
        static_config_.is_2d_fabric_dynamic =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().get_fabric_config() ==
            tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
    } else {
        static_config_.is_2d_fabric = false;
        static_config_.is_2d_fabric_dynamic = false;
    }
}

void DispatchKernel::GenerateDependentConfigs() {
    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        // Upstream
        TT_ASSERT(upstream_kernels_.size() == 1);
        auto prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0]);
        TT_ASSERT(prefetch_kernel);
        dependent_config_.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        dependent_config_.upstream_dispatch_cb_sem_id = prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id;
        dependent_config_.upstream_sync_sem = prefetch_kernel->GetStaticConfig().downstream_sync_sem_id;

        if (prefetch_kernel->GetStaticConfig().is_h_variant.value() &&
            prefetch_kernel->GetStaticConfig().is_d_variant.value()) {
            dependent_config_.split_prefetch = false;
            dependent_config_.prefetch_h_noc_xy = 0;
            dependent_config_.prefetch_h_local_downstream_sem_addr = 0;
        } else {
            dependent_config_.split_prefetch = true;
            dependent_config_.prefetch_h_noc_xy = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(
                prefetch_kernel->GetVirtualCore().x, prefetch_kernel->GetVirtualCore().y);
            dependent_config_.prefetch_h_local_downstream_sem_addr =
                prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id.value();
        }

        // Downstream
        if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            TT_ASSERT(downstream_kernels_.size() == 1);
            auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(downstream_kernels_[0]);
            TT_ASSERT(dispatch_s_kernel);
            dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
        } else {
            // If no dispatch_s, no downstream
            TT_ASSERT(downstream_kernels_.size() == 0);
            dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        }
        dependent_config_.downstream_logical_core = UNUSED_LOGICAL_CORE;  // Unused
        dependent_config_.downstream_cb_base = 0;                         // Unused
        dependent_config_.downstream_cb_size = 0;                         // Unused
        dependent_config_.downstream_cb_sem_id = UNUSED_SEM_ID;           // Unused
        dependent_config_.num_hops = 0;
    } else if (static_config_.is_h_variant.value()) {
        // Upstream, expect connection to DISPATCH_D

        // May be overwritten below
        dependent_config_.num_hops = 0;
        TT_ASSERT(upstream_kernels_.size() == 1);
        if (auto dispatch_d = dynamic_cast<DispatchKernel*>(upstream_kernels_[0])) {
            dependent_config_.upstream_logical_core = dispatch_d->GetLogicalCore();
            dependent_config_.upstream_dispatch_cb_sem_id =
                dispatch_d->GetStaticConfig().my_downstream_cb_sem_id.value();
            dependent_config_.upstream_sync_sem = 0;  // Unused
            dependent_config_.num_hops = tt::tt_metal::get_num_hops(device_id_, dispatch_d->GetDeviceId());
            assemble_2d_fabric_packet_header_args(this->dependent_config_, GetDeviceId(), dispatch_d->GetDeviceId());
        } else {
            TT_FATAL(false, "Unimplemented path");
        }

        // Downstream
        // PREFETCH_H || FABRIC_MUX
        // Downstream, no official downstream core but use the field to connect is to the PREFETCH_H that we need to
        // write to when resuming sending of commands post exec_buf stall.
        bool found_prefetch_h = false;
        bool found_relay_mux = false;
        for (FDKernel* ds_kernel : downstream_kernels_) {
            if (auto prefetch_h_kernel = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
                TT_ASSERT(prefetch_h_kernel && prefetch_h_kernel->GetStaticConfig().is_h_variant.value());
                TT_ASSERT(!found_prefetch_h, "DISPATCH_H has multiple downstream PREFETCH_H kernels.");
                found_prefetch_h = true;
                dependent_config_.prefetch_h_noc_xy = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(
                    prefetch_h_kernel->GetVirtualCore().x, prefetch_h_kernel->GetVirtualCore().y);
                dependent_config_.prefetch_h_local_downstream_sem_addr =
                    prefetch_h_kernel->GetStaticConfig().my_downstream_cb_sem_id;
            } else if (auto relay_mux = dynamic_cast<tt::tt_metal::RelayMux*>(ds_kernel)) {
                TT_ASSERT(!found_relay_mux, "DISPATCH_H has multiple downstream RELAY_MUX kernels.");
                found_relay_mux = true;

                constexpr tt::tt_fabric::FabricMuxChannelType ch_type =
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL;
                tt::tt_metal::assemble_fabric_mux_client_config_args(
                    node_id_, ch_type, relay_mux, dependent_config_.fabric_mux_client_config);
            } else {
                TT_FATAL(false, "DISPATCH_H Downstream - Unimplemented path");
            }
        }

        TT_ASSERT(found_prefetch_h, "DISPATCH_H expects a PREFETCH_H downstream");

        dependent_config_.downstream_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.split_prefetch = true;
        dependent_config_.downstream_cb_base = 0;    // Unused
        dependent_config_.downstream_cb_size = 0;    // Unused
        dependent_config_.downstream_cb_sem_id = 0;  // Unused
    } else if (static_config_.is_d_variant.value()) {
        // Upstream, expect a PREFETCH_D
        TT_ASSERT(upstream_kernels_.size() == 1);
        auto prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0]);
        TT_ASSERT(prefetch_kernel);
        dependent_config_.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        dependent_config_.upstream_dispatch_cb_sem_id = prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id;
        dependent_config_.upstream_sync_sem = prefetch_kernel->GetStaticConfig().downstream_sync_sem_id;
        // May be overwritten below
        dependent_config_.num_hops = 0;

        if (prefetch_kernel->GetStaticConfig().is_h_variant.value() &&
            prefetch_kernel->GetStaticConfig().is_d_variant.value()) {
            dependent_config_.split_prefetch = false;
            dependent_config_.prefetch_h_noc_xy = 0;
            dependent_config_.prefetch_h_local_downstream_sem_addr = 0;
        } else {
            dependent_config_.split_prefetch = true;
            dependent_config_.prefetch_h_noc_xy = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(
                prefetch_kernel->GetVirtualCore().x, prefetch_kernel->GetVirtualCore().y);
            dependent_config_.prefetch_h_local_downstream_sem_addr =
                prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id.value();
        }

        // Downstream, expect a MUX_D
        // Or direct connection to DISPATCH_H if using fabric
        //
        // + A Dispatch_s if enabled

        bool found_dispatch_s = false;
        bool found_dispatch_h = false;
        bool found_relay_mux = false;  // fabric mux
        for (auto ds_kernel : downstream_kernels_) {
            if (auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(ds_kernel)) {
                TT_ASSERT(!found_dispatch_s, "DISPATCH_D has multiple downstream DISPATCH_S kernels.");
                dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
                found_dispatch_s = true;
            } else if (auto dispatch_h_kernel = dynamic_cast<DispatchKernel*>(ds_kernel)) {
                TT_ASSERT(!found_dispatch_h, "DISPATCH_D has multiple downstream DISPATCH_H kernels.");
                dependent_config_.downstream_logical_core = dispatch_h_kernel->GetLogicalCore();
                dependent_config_.downstream_cb_size = dispatch_h_kernel->GetDispatchBufferSize();
                dependent_config_.downstream_cb_base = dispatch_h_kernel->GetStaticConfig().dispatch_cb_base.value();
                dependent_config_.downstream_cb_sem_id =
                    dispatch_h_kernel->GetStaticConfig().my_dispatch_cb_sem_id.value();
                dependent_config_.num_hops = tt::tt_metal::get_num_hops(dispatch_h_kernel->GetDeviceId(), device_id_);
                assemble_2d_fabric_packet_header_args(
                    this->dependent_config_, GetDeviceId(), dispatch_h_kernel->GetDeviceId());
                found_dispatch_h = true;
            } else if (auto relay_mux = dynamic_cast<tt::tt_metal::RelayMux*>(ds_kernel)) {
                TT_ASSERT(!found_relay_mux, "DISPATCH_D has multiple downstream RELAY_MUX kernels.");
                found_relay_mux = true;

                constexpr tt::tt_fabric::FabricMuxChannelType ch_type =
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
                tt::tt_metal::assemble_fabric_mux_client_config_args(
                    node_id_, ch_type, relay_mux, dependent_config_.fabric_mux_client_config);
            } else {
                TT_FATAL(false, "Unexpected downstream kernel for dispatch_d");
            }
        }

        TT_FATAL(
            !MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled() || found_dispatch_s,
            "dispatch_d is missing dispatch_s downstream");
        TT_FATAL(found_dispatch_h, "Path not implemented for dispatch_d. dispatch_h in downstream is required");

        if (!found_dispatch_s) {
            dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        }
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }
}

void DispatchKernel::CreateKernel() {
    // Issue #19729: Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
    // Num num_virtual_active_eth_cores is set if the user application requested virtualizing the
    // number of ethernet cores across devices (to essentially fake uniformity). This value is the
    // max number of ethernet cores acorss all chip in the cluster.
    // num_physical_ethernet_cores is the number of actual available ethernet cores on the current device.
    // virtualize_num_eth_cores is set if the number of virtual cores is greater than the number of actual
    // ethernet cores in the chip.
    uint32_t num_virtual_active_eth_cores = tt::DevicePool::instance().get_max_num_eth_cores_across_all_devices();
    uint32_t num_physical_active_eth_cores =
        MetalContext::instance()
            .get_control_plane()
            .get_active_ethernet_cores(device_->id(), /*skip_reserved_tunnel_cores*/ true)
            .size();
    bool virtualize_num_eth_cores = num_virtual_active_eth_cores > num_physical_active_eth_cores;

    std::vector<uint32_t> compile_args = {};

    auto my_virtual_core = get_virtual_core_coord(logical_core_, GetCoreType());
    auto upstream_virtual_core = get_virtual_core_coord(dependent_config_.upstream_logical_core.value(), GetCoreType());
    auto downstream_virtual_core =
        get_virtual_core_coord(dependent_config_.downstream_logical_core.value(), GetCoreType());
    auto downstream_s_virtual_core =
        get_virtual_core_coord(dependent_config_.downstream_s_logical_core.value(), GetCoreType());

    auto my_virtual_noc_coords = device_->virtual_noc0_coordinate(noc_selection_.non_dispatch_noc, my_virtual_core);
    auto upstream_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.upstream_noc, upstream_virtual_core);
    auto downstream_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_virtual_core);
    auto downstream_s_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_s_virtual_core);

    std::map<std::string, std::string> defines = {
        {"MY_NOC_X", std::to_string(my_virtual_noc_coords.x)},
        {"MY_NOC_Y", std::to_string(my_virtual_noc_coords.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(upstream_virtual_noc_coords.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_virtual_noc_coords.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},

        // Add all the dispatch-specific defines
        {"DISPATCH_CB_BASE", std::to_string(static_config_.dispatch_cb_base.value())},
        {"DISPATCH_CB_LOG_PAGE_SIZE", std::to_string(static_config_.dispatch_cb_log_page_size.value())},
        {"DISPATCH_CB_PAGES", std::to_string(static_config_.dispatch_cb_pages.value())},
        {"MY_DISPATCH_CB_SEM_ID", std::to_string(static_config_.my_dispatch_cb_sem_id.value())},
        {"UPSTREAM_DISPATCH_CB_SEM_ID", std::to_string(dependent_config_.upstream_dispatch_cb_sem_id.value())},
        {"DISPATCH_CB_BLOCKS", std::to_string(static_config_.dispatch_cb_blocks.value())},
        {"UPSTREAM_SYNC_SEM", std::to_string(dependent_config_.upstream_sync_sem.value())},
        {"COMMAND_QUEUE_BASE_ADDR", std::to_string(static_config_.command_queue_base_addr.value())},
        {"COMPLETION_QUEUE_BASE_ADDR", std::to_string(static_config_.completion_queue_base_addr.value())},
        {"COMPLETION_QUEUE_SIZE", std::to_string(static_config_.completion_queue_size.value())},
        {"DOWNSTREAM_CB_BASE", std::to_string(dependent_config_.downstream_cb_base.value())},
        {"DOWNSTREAM_CB_SIZE", std::to_string(dependent_config_.downstream_cb_size.value())},
        {"MY_DOWNSTREAM_CB_SEM_ID", std::to_string(static_config_.my_downstream_cb_sem_id.value())},
        {"DOWNSTREAM_CB_SEM_ID", std::to_string(dependent_config_.downstream_cb_sem_id.value())},
        {"SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE", std::to_string(static_config_.split_dispatch_page_preamble_size.value())},
        {"SPLIT_PREFETCH", std::to_string(dependent_config_.split_prefetch.value())},
        {"PREFETCH_H_NOC_XY", std::to_string(dependent_config_.prefetch_h_noc_xy.value())},
        {"PREFETCH_H_LOCAL_DOWNSTREAM_SEM_ADDR",
         std::to_string(dependent_config_.prefetch_h_local_downstream_sem_addr.value())},
        {"PREFETCH_H_MAX_CREDITS", std::to_string(static_config_.prefetch_h_max_credits.value())},
        {"PACKED_WRITE_MAX_UNICAST_SUB_CMDS", std::to_string(static_config_.packed_write_max_unicast_sub_cmds.value())},
        {"DISPATCH_S_SYNC_SEM_BASE_ADDR", std::to_string(static_config_.dispatch_s_sync_sem_base_addr.value())},
        {"MAX_NUM_WORKER_SEMS", std::to_string(static_config_.max_num_worker_sems.value())},
        {"MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES",
         std::to_string(static_config_.max_num_go_signal_noc_data_entries.value())},
        {"MCAST_GO_SIGNAL_ADDR", std::to_string(static_config_.mcast_go_signal_addr.value())},
        {"UNICAST_GO_SIGNAL_ADDR", std::to_string(static_config_.unicast_go_signal_addr.value())},
        {"DISTRIBUTED_DISPATCHER", std::to_string(static_config_.distributed_dispatcher.value())},
        {"HOST_COMPLETION_Q_WR_PTR", std::to_string(static_config_.host_completion_q_wr_ptr.value())},
        {"DEV_COMPLETION_Q_WR_PTR", std::to_string(static_config_.dev_completion_q_wr_ptr.value())},
        {"DEV_COMPLETION_Q_RD_PTR", std::to_string(static_config_.dev_completion_q_rd_ptr.value())},
        {"FIRST_STREAM_USED", std::to_string(static_config_.first_stream_used.value())},
        {"VIRTUALIZE_UNICAST_CORES", std::to_string(virtualize_num_eth_cores)},
        {"NUM_VIRTUAL_UNICAST_CORES", std::to_string(num_virtual_active_eth_cores)},
        {"NUM_PHYSICAL_UNICAST_CORES", std::to_string(num_physical_active_eth_cores)},
        {"FABRIC_HEADER_RB_BASE", std::to_string(static_config_.fabric_header_rb_base.value())},
        {"FABRIC_HEADER_RB_ENTRIES", std::to_string(static_config_.fabric_header_rb_entries.value())},
        {"MY_FABRIC_SYNC_STATUS_ADDR", std::to_string(static_config_.my_fabric_sync_status_addr.value())},
        {"FABRIC_MUX_X", std::to_string(dependent_config_.fabric_mux_client_config.virtual_x.value_or(0))},
        {"FABRIC_MUX_Y", std::to_string(dependent_config_.fabric_mux_client_config.virtual_y.value_or(0))},
        {"FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL",
         std::to_string(dependent_config_.fabric_mux_client_config.num_buffers_per_channel.value_or(0))},
        {"FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES",
         std::to_string(dependent_config_.fabric_mux_client_config.channel_buffer_size_bytes.value_or(0))},
        {"FABRIC_MUX_CHANNEL_BASE_ADDRESS",
         std::to_string(dependent_config_.fabric_mux_client_config.channel_base_address.value_or(0))},
        {"FABRIC_MUX_CONNECTION_INFO_ADDRESS",
         std::to_string(dependent_config_.fabric_mux_client_config.connection_info_address.value_or(0))},
        {"FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS",
         std::to_string(dependent_config_.fabric_mux_client_config.connection_handshake_address.value_or(0))},
        {"FABRIC_MUX_FLOW_CONTROL_ADDRESS",
         std::to_string(dependent_config_.fabric_mux_client_config.flow_control_address.value_or(0))},
        {"FABRIC_MUX_BUFFER_INDEX_ADDRESS",
         std::to_string(dependent_config_.fabric_mux_client_config.buffer_index_address.value_or(0))},
        {"FABRIC_MUX_STATUS_ADDRESS",
         std::to_string(dependent_config_.fabric_mux_client_config.status_address.value_or(0))},
        {"FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS",
         std::to_string(dependent_config_.fabric_mux_client_config.termination_signal_address.value_or(0))},
        {"WORKER_CREDITS_STREAM_ID",
         std::to_string(dependent_config_.fabric_mux_client_config.worker_credits_stream_id.value_or(0))},
        {"FABRIC_WORKER_FLOW_CONTROL_SEM", std::to_string(edm_connection_attributes_.worker_flow_control_sem)},
        {"FABRIC_WORKER_TEARDOWN_SEM", std::to_string(edm_connection_attributes_.worker_teardown_sem)},
        {"FABRIC_WORKER_BUFFER_INDEX_SEM", std::to_string(edm_connection_attributes_.worker_buffer_index_sem)},
        {"NUM_HOPS", std::to_string(dependent_config_.num_hops.value())},
        {"MY_DEV_ID", std::to_string(dependent_config_.my_dev_id.value_or(0))},
        {"EW_DIM", std::to_string(dependent_config_.ew_dim.value_or(0))},
        {"TO_MESH_ID", std::to_string(dependent_config_.to_mesh_id.value_or(0))},
        {"TO_DEV_ID", std::to_string(dependent_config_.to_dev_id.value_or(0))},
        {"ROUTER_DIRECTION", std::to_string(dependent_config_.router_direction.value_or(0))},
        {"IS_D_VARIANT", std::to_string(static_config_.is_d_variant.value())},
        {"IS_H_VARIANT", std::to_string(static_config_.is_h_variant.value())},
    };
    if (!is_hd()) {
        defines["FABRIC_RELAY"] = "1";
        if (static_config_.is_2d_fabric.value_or(false)) {
            defines["FABRIC_2D"] = "1";
        }
        if (static_config_.is_2d_fabric_dynamic.value_or(false)) {
            defines["FABRIC_2D_DYNAMIC"] = "1";
        }
    }
    // Compile at Os on IERISC to fit in code region.
    auto optimization_level = (GetCoreType() == CoreType::WORKER) ? KernelBuildOptLevel::O2 : KernelBuildOptLevel::Os;
    configure_kernel_variant(
        dispatch_kernel_file_names[DISPATCH], compile_args, defines, false, true, false, optimization_level);
}

void DispatchKernel::ConfigureCore() {
    // For all dispatchers, need to clear the dispatch message
    std::vector<uint32_t> zero = {0x0};
    auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());
    uint32_t dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
    for (uint32_t i = 0; i < DispatchSettings::DISPATCH_MESSAGE_ENTRIES; i++) {
        uint32_t dispatch_s_sync_sem_addr = dispatch_s_sync_sem_base_addr + my_dispatch_constants.get_sync_offset(i);
        detail::WriteToDeviceL1(device_, logical_core_, dispatch_s_sync_sem_addr, zero, GetCoreType());
    }

    // For DISPATCH_D, need to clear completion q events
    if (!static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        uint32_t completion_q0_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
        uint32_t completion_q1_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
        detail::WriteToDeviceL1(device_, logical_core_, completion_q0_last_event_ptr, zero, GetCoreType());
        detail::WriteToDeviceL1(device_, logical_core_, completion_q1_last_event_ptr, zero, GetCoreType());
    }
}
