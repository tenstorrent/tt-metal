// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "dispatch.hpp"
#include "prefetch.hpp"
#include "dispatch_s.hpp"
#include "demux.hpp"
#include "mux.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>

using namespace tt::tt_metal;

void DispatchKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_->id());
    uint8_t cq_id_ = this->cq_id_;
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());

    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device_->sysmem_manager().get_completion_queue_size(cq_id_);

        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        static_config_.dispatch_cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = command_queue_start_addr;
        static_config_.completion_queue_base_addr = completion_queue_start_addr;
        static_config_.completion_queue_size = completion_queue_size;

        static_config_.my_downstream_cb_sem_id = 0;  // unused

        static_config_.split_dispatch_page_preamble_size = 0;        // unused
        static_config_.split_prefetch = false;                       // split_prefetcher
        dependent_config_.prefetch_h_noc_xy = 0;                     // unused prefetch noc_xy
        dependent_config_.prefetch_h_local_downstream_sem_addr = 0;  // unused prefetch_local_downstream_sem_addr
        static_config_.prefetch_h_max_credits = 0;                   // unused prefetch_downstream_buffer_pages

        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
        static_config_.max_num_worker_sems = dispatch_constants::DISPATCH_MESSAGE_ENTRIES;
        static_config_.max_num_go_signal_noc_data_entries = dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        static_config_.mcast_go_signal_addr =
            hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        static_config_.unicast_go_signal_addr =
            (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        static_config_.distributed_dispatcher = (GetCoreType() == CoreType::ETH);

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else if (static_config_.is_h_variant.value()) {
        // DISPATCH_H services a remote chip, and so has a different channel
        channel = tt::Cluster::instance().get_assigned_channel_for_device(servicing_device_id_);
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device_->sysmem_manager().get_completion_queue_size(cq_id_);

        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        static_config_.dispatch_cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = command_queue_start_addr;
        static_config_.completion_queue_base_addr = completion_queue_start_addr;
        static_config_.completion_queue_size = completion_queue_size;

        static_config_.my_downstream_cb_sem_id = 0;  // Unused

        static_config_.split_dispatch_page_preamble_size = 0;
        static_config_.split_prefetch = true;
        // TODO: why is this hard-coded to 1 CQ on Galaxy?
        if (tt::Cluster::instance().is_galaxy_cluster()) {
            static_config_.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(1);
        } else {
            static_config_.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(device_->num_hw_cqs());
        }

        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr = 0;       // Unused
        static_config_.max_num_worker_sems = 1;                 // Used for array sizing, set to 1 even if unused
        static_config_.max_num_go_signal_noc_data_entries = 1;  // Used for array sizing, sset to 1 even if unused
        static_config_.mcast_go_signal_addr = 0;                // Unused
        static_config_.unicast_go_signal_addr = 0;              // Unused
        static_config_.distributed_dispatcher = 0;              // Unused

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else if (static_config_.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device_->sysmem_manager().get_completion_queue_size(cq_id_);

        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        static_config_.dispatch_cb_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = 0;  // These are unused for DISPATCH_D
        static_config_.completion_queue_base_addr = 0;
        static_config_.completion_queue_size = 0;

        static_config_.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program_,
            logical_core_,
            my_dispatch_constants.mux_buffer_pages(device_->num_hw_cqs()),
            GetCoreType());  // Apparently unused

        static_config_.split_dispatch_page_preamble_size = sizeof(dispatch_packet_header_t);
        static_config_.split_prefetch = true;
        dependent_config_.prefetch_h_noc_xy = 0;
        dependent_config_.prefetch_h_local_downstream_sem_addr = 1;
        static_config_.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(device_->num_hw_cqs());

        // To match with previous implementation, need to use grid size from mmio device. TODO: that doesn't seem
        // correct though?
        auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id_);
        const auto& dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config(mmio_device_id);
        CoreCoord remote_grid_size =
            tt::get_compute_grid_size(mmio_device_id, device_->num_hw_cqs(), dispatch_core_config);
        static_config_.packed_write_max_unicast_sub_cmds = remote_grid_size.x * remote_grid_size.y;
        static_config_.dispatch_s_sync_sem_base_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
        static_config_.max_num_worker_sems = dispatch_constants::DISPATCH_MESSAGE_ENTRIES;
        static_config_.max_num_go_signal_noc_data_entries = dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        static_config_.mcast_go_signal_addr =
            hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        static_config_.unicast_go_signal_addr =
            (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        static_config_.distributed_dispatcher = (GetCoreType() == CoreType::ETH);

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
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

        // Downstream
        if (device_->dispatch_s_enabled()) {
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
    } else if (static_config_.is_h_variant.value()) {
        // Upstream, expect DEMUX
        TT_ASSERT(upstream_kernels_.size() == 1);
        auto demux_kernel = dynamic_cast<DemuxKernel*>(upstream_kernels_[0]);
        TT_ASSERT(demux_kernel);
        dependent_config_.upstream_logical_core = demux_kernel->GetLogicalCore();
        int demux_idx =
            demux_kernel->GetDownstreamPort(this);  // Need to know which port this kernel connects to upstream
        dependent_config_.upstream_dispatch_cb_sem_id =
            demux_kernel->GetStaticConfig().output_depacketize_local_sem_id[demux_idx].value();
        dependent_config_.upstream_sync_sem = 0;  // Unused

        // Downstream, no official downstream core but use the field to connect is to the PREFETCH_H that we need to
        // write to when resuming sending of commands post exec_buf stall.
        TT_ASSERT(downstream_kernels_.size() == 1);
        auto prefetch_h_kernel = dynamic_cast<PrefetchKernel*>(downstream_kernels_[0]);
        TT_ASSERT(prefetch_h_kernel);
        dependent_config_.downstream_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.prefetch_h_noc_xy = tt::tt_metal::hal.noc_xy_encoding(
            prefetch_h_kernel->GetVirtualCore().x, prefetch_h_kernel->GetVirtualCore().y);
        dependent_config_.prefetch_h_local_downstream_sem_addr =
            prefetch_h_kernel->GetStaticConfig().my_downstream_cb_sem_id;
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
        // Downstream, expect a MUX_D and a DISPATCH_S if enabled
        auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(downstream_kernels_[0]);
        auto mux_kernel = dynamic_cast<MuxKernel*>(downstream_kernels_[0]);
        if (device_->dispatch_s_enabled()) {
            TT_ASSERT(downstream_kernels_.size() == 2);
            mux_kernel = dynamic_cast<MuxKernel*>(downstream_kernels_[1]);
            if (!dispatch_s_kernel) {
                dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(downstream_kernels_[1]);
                mux_kernel = dynamic_cast<MuxKernel*>(downstream_kernels_[0]);
            }
            TT_ASSERT(dispatch_s_kernel);
            dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
        } else {
            TT_ASSERT(downstream_kernels_.size() == 1);
            dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        }
        TT_ASSERT(mux_kernel);
        dependent_config_.downstream_logical_core = mux_kernel->GetLogicalCore();
        // Some configs depend on which port this kernel connects to on the downstream kernel
        int dispatch_d_idx = mux_kernel->GetUpstreamPort(this);  // Need the port that this connects to downstream
        dependent_config_.downstream_cb_size = mux_kernel->GetStaticConfig().rx_queue_size_words.value() << 4;
        // MUX queue id is "dependent_config_.downstream_cb_size.value()"
        // The address for that queue starts at "rx_queue_start_addr_words + i*rx_queue_size_words" (based on kernel
        // code)
        dependent_config_.downstream_cb_base = (mux_kernel->GetStaticConfig().rx_queue_start_addr_words.value() << 4) +
                                               dispatch_d_idx * dependent_config_.downstream_cb_size.value();
        dependent_config_.downstream_cb_sem_id = dispatch_d_idx;
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }
}

void DispatchKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        static_config_.dispatch_cb_base.value(),
        static_config_.dispatch_cb_log_page_size.value(),
        static_config_.dispatch_cb_pages.value(),
        static_config_.my_dispatch_cb_sem_id.value(),
        dependent_config_.upstream_dispatch_cb_sem_id.value(),

        static_config_.dispatch_cb_blocks.value(),
        dependent_config_.upstream_sync_sem.value(),
        static_config_.command_queue_base_addr.value(),
        static_config_.completion_queue_base_addr.value(),
        static_config_.completion_queue_size.value(),

        dependent_config_.downstream_cb_base.value(),
        dependent_config_.downstream_cb_size.value(),
        static_config_.my_downstream_cb_sem_id.value(),
        dependent_config_.downstream_cb_sem_id.value(),

        static_config_.split_dispatch_page_preamble_size.value(),
        static_config_.split_prefetch.value(),
        dependent_config_.prefetch_h_noc_xy.value(),
        dependent_config_.prefetch_h_local_downstream_sem_addr.value(),
        static_config_.prefetch_h_max_credits.value(),

        static_config_.packed_write_max_unicast_sub_cmds.value(),
        static_config_.dispatch_s_sync_sem_base_addr.value(),
        static_config_.max_num_worker_sems.value(),
        static_config_.max_num_go_signal_noc_data_entries.value(),
        static_config_.mcast_go_signal_addr.value(),
        static_config_.unicast_go_signal_addr.value(),
        static_config_.distributed_dispatcher.value(),

        static_config_.host_completion_q_wr_ptr.value(),
        static_config_.dev_completion_q_wr_ptr.value(),
        static_config_.dev_completion_q_rd_ptr.value(),

        static_config_.is_d_variant.value(),
        static_config_.is_h_variant.value(),
    };
    TT_ASSERT(compile_args.size() == 31);
    auto my_virtual_core = device_->virtual_core_from_logical_core(logical_core_, GetCoreType());
    auto upstream_virtual_core =
        device_->virtual_core_from_logical_core(dependent_config_.upstream_logical_core.value(), GetCoreType());
    auto downstream_virtual_core =
        device_->virtual_core_from_logical_core(dependent_config_.downstream_logical_core.value(), GetCoreType());
    auto downstream_s_virtual_core =
        device_->virtual_core_from_logical_core(dependent_config_.downstream_s_logical_core.value(), GetCoreType());

    auto my_virtual_noc_coords = device_->virtual_noc0_coordinate(noc_selection_.non_dispatch_noc, my_virtual_core);
    auto upstream_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.upstream_noc, upstream_virtual_core);
    auto downstream_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_virtual_core);
    auto downstream_s_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_s_virtual_core);

    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(my_virtual_noc_coords.x)},
        {"MY_NOC_Y", std::to_string(my_virtual_noc_coords.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(upstream_virtual_noc_coords.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_virtual_noc_coords.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},
    };
    configure_kernel_variant(dispatch_kernel_file_names[DISPATCH], compile_args, defines, false, false, false);
}

void DispatchKernel::ConfigureCore() {
    // For all dispatchers, need to clear the dispatch message
    std::vector<uint32_t> zero = {0x0};
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    uint32_t dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
    uint32_t dispatch_message_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    for (uint32_t i = 0; i < dispatch_constants::DISPATCH_MESSAGE_ENTRIES; i++) {
        uint32_t dispatch_s_sync_sem_addr =
            dispatch_s_sync_sem_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        detail::WriteToDeviceL1(device_, logical_core_, dispatch_s_sync_sem_addr, zero, GetCoreType());
        detail::WriteToDeviceL1(device_, logical_core_, dispatch_message_addr, zero, GetCoreType());
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
