// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_kernel.hpp"
#include "prefetch_kernel.hpp"
#include "dispatch_s_kernel.hpp"
#include "demux_kernel.hpp"
#include "mux_kernel.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

void DispatchKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    uint8_t cq_id = this->cq_id;
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());

    if (this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device->sysmem_manager().get_completion_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().dispatcher_core(device->id(), channel, cq_id);
        this->config.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.dispatch_cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        this->config.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        this->config.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        this->config.command_queue_base_addr = command_queue_start_addr;
        this->config.completion_queue_base_addr = completion_queue_start_addr;
        this->config.completion_queue_size = completion_queue_size;

        this->config.my_downstream_cb_sem_id = 0;  // unused

        this->config.split_dispatch_page_preamble_size = 0;     // unused
        this->config.split_prefetch = false;                    // split_prefetcher
        this->config.prefetch_h_noc_xy = 0;                     // unused prefetch noc_xy
        this->config.prefetch_h_local_downstream_sem_addr = 0;  // unused prefetch_local_downstream_sem_addr
        this->config.prefetch_h_max_credits = 0;                // unused prefetch_downstream_buffer_pages

        this->config.packed_write_max_unicast_sub_cmds =
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
        this->config.dispatch_s_sync_sem_base_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
        this->config.max_num_worker_sems = dispatch_constants::DISPATCH_MESSAGE_ENTRIES;
        this->config.max_num_go_signal_noc_data_entries = dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        this->config.mcast_go_signal_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        this->config.unicast_go_signal_addr =
            (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        this->config.distributed_dispatcher = (GetCoreType() == CoreType::ETH);

        this->config.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else if (this->config.is_h_variant.value()) {
        // DISPATCH_H services a remote chip, and so has a different channel
        channel = tt::Cluster::instance().get_assigned_channel_for_device(servicing_device_id);
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device->sysmem_manager().get_completion_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().dispatcher_core(servicing_device_id, channel, cq_id);
        this->config.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.dispatch_cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        this->config.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        this->config.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        this->config.command_queue_base_addr = command_queue_start_addr;
        this->config.completion_queue_base_addr = completion_queue_start_addr;
        this->config.completion_queue_size = completion_queue_size;

        this->config.my_downstream_cb_sem_id = 0;  // Unused

        this->config.split_dispatch_page_preamble_size = 0;
        this->config.split_prefetch = true;
        // TODO: why is this hard-coded to 1 CQ on Galaxy?
        if (tt::Cluster::instance().is_galaxy_cluster()) {
            this->config.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(1);
        } else {
            this->config.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());
        }

        this->config.packed_write_max_unicast_sub_cmds =
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
        this->config.dispatch_s_sync_sem_base_addr = 0;       // Unused
        this->config.max_num_worker_sems = 1;                 // Used for array sizing, set to 1 even if unused
        this->config.max_num_go_signal_noc_data_entries = 1;  // Used for array sizing, sset to 1 even if unused
        this->config.mcast_go_signal_addr = 0;                // Unused
        this->config.unicast_go_signal_addr = 0;              // Unused
        this->config.distributed_dispatcher = 0;              // Unused

        this->config.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else if (this->config.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device->sysmem_manager().get_completion_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().dispatcher_d_core(device->id(), channel, cq_id);
        this->config.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.dispatch_cb_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        this->config.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        this->config.my_dispatch_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        this->config.command_queue_base_addr = 0;  // These are unused for DISPATCH_D
        this->config.completion_queue_base_addr = 0;
        this->config.completion_queue_size = 0;

        this->config.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program,
            this->logical_core,
            my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs()),
            GetCoreType());  // Apparently unused

        this->config.split_dispatch_page_preamble_size = sizeof(dispatch_packet_header_t);
        this->config.split_prefetch = true;
        this->config.prefetch_h_noc_xy = 0;
        this->config.prefetch_h_local_downstream_sem_addr = 1;
        this->config.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());

        // To match with previous implementation, need to use grid size from mmio device. TODO: that doesn't seem
        // correct though?
        auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        const auto& dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config(mmio_device_id);
        CoreCoord remote_grid_size =
            tt::get_compute_grid_size(mmio_device_id, device->num_hw_cqs(), dispatch_core_config);
        this->config.packed_write_max_unicast_sub_cmds = remote_grid_size.x * remote_grid_size.y;
        this->config.dispatch_s_sync_sem_base_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
        this->config.max_num_worker_sems = dispatch_constants::DISPATCH_MESSAGE_ENTRIES;
        this->config.max_num_go_signal_noc_data_entries = dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        this->config.mcast_go_signal_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        this->config.unicast_go_signal_addr =
            (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        this->config.distributed_dispatcher = (GetCoreType() == CoreType::ETH);

        this->config.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }
}

void DispatchKernel::GenerateDependentConfigs() {
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    if (this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        // Upstream
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto prefetch_kernel = dynamic_cast<PrefetchKernel*>(this->upstream_kernels[0]);
        TT_ASSERT(prefetch_kernel);
        this->config.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        this->config.upstream_dispatch_cb_sem_id = prefetch_kernel->GetConfig().my_downstream_cb_sem_id;
        this->config.upstream_sync_sem = prefetch_kernel->GetConfig().downstream_sync_sem_id;

        // Downstream
        if (device->dispatch_s_enabled()) {
            TT_ASSERT(this->downstream_kernels.size() == 1);
            auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(this->downstream_kernels[0]);
            TT_ASSERT(dispatch_s_kernel);
            this->config.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
        } else {
            // If no dispatch_s, no downstream
            TT_ASSERT(this->downstream_kernels.size() == 0);
            this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        }
        this->config.downstream_logical_core = UNUSED_LOGICAL_CORE;
        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.downstream_cb_size =
            (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * my_dispatch_constants.dispatch_buffer_pages();
        this->config.downstream_cb_sem_id = UNUSED_SEM_ID;
    } else if (this->config.is_h_variant.value()) {
        // Upstream, expect DEMUX
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto demux_kernel = dynamic_cast<DemuxKernel*>(this->upstream_kernels[0]);
        TT_ASSERT(demux_kernel);
        this->config.upstream_logical_core = demux_kernel->GetLogicalCore();
        int demux_idx =
            demux_kernel->GetDownstreamPort(this);  // Need to know which port this kernel connects to upstream
        this->config.upstream_dispatch_cb_sem_id =
            demux_kernel->GetConfig().output_depacketize_local_sem_id[demux_idx].value();
        this->config.upstream_sync_sem = 0;  // Unused

        // Downstream, no official downstream core but use the field to connect is to the PREFETCH_H that we need to
        // write to when resuming sending of commands post exec_buf stall.
        TT_ASSERT(this->downstream_kernels.size() == 1);
        auto prefetch_h_kernel = dynamic_cast<PrefetchKernel*>(this->downstream_kernels[0]);
        TT_ASSERT(prefetch_h_kernel);
        this->config.downstream_logical_core = UNUSED_LOGICAL_CORE_ADJUSTED;
        this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE_ADJUSTED;
        this->config.prefetch_h_noc_xy = tt::tt_metal::hal.noc_xy_encoding(
            prefetch_h_kernel->GetVirtualCore().x, prefetch_h_kernel->GetVirtualCore().y);
        this->config.prefetch_h_local_downstream_sem_addr = prefetch_h_kernel->GetConfig().my_downstream_cb_sem_id;
        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base();  // Unused
        this->config.downstream_cb_size = (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                          my_dispatch_constants.dispatch_buffer_pages();  // Unused
        this->config.downstream_cb_sem_id = 0;                                            // Unused
    } else if (this->config.is_d_variant.value()) {
        // Upstream, expect a PREFETCH_D
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto prefetch_kernel = dynamic_cast<PrefetchKernel*>(this->upstream_kernels[0]);
        TT_ASSERT(prefetch_kernel);
        this->config.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        this->config.upstream_dispatch_cb_sem_id = prefetch_kernel->GetConfig().my_downstream_cb_sem_id;
        this->config.upstream_sync_sem = prefetch_kernel->GetConfig().downstream_sync_sem_id;
        // Downstream, expect a MUX_D and a DISPATCH_S if enabled
        auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(this->downstream_kernels[0]);
        auto mux_kernel = dynamic_cast<MuxKernel*>(this->downstream_kernels[0]);
        if (device->dispatch_s_enabled()) {
            TT_ASSERT(this->downstream_kernels.size() == 2);
            mux_kernel = dynamic_cast<MuxKernel*>(this->downstream_kernels[1]);
            if (!dispatch_s_kernel) {
                dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(this->downstream_kernels[1]);
                mux_kernel = dynamic_cast<MuxKernel*>(this->downstream_kernels[0]);
            }
            TT_ASSERT(dispatch_s_kernel);
            this->config.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
        } else {
            TT_ASSERT(this->downstream_kernels.size() == 1);
            this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        }
        TT_ASSERT(mux_kernel);
        this->config.downstream_logical_core = mux_kernel->GetLogicalCore();
        // Some configs depend on which port this kernel connects to on the downstream kernel
        int dispatch_d_idx = mux_kernel->GetUpstreamPort(this);  // Need the port that this connects to downstream
        this->config.downstream_cb_size = (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                          my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());
        this->config.downstream_cb_base =
            my_dispatch_constants.dispatch_buffer_base() + this->config.downstream_cb_size.value() * dispatch_d_idx;
        this->config.downstream_cb_sem_id = dispatch_d_idx;
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }
}

void DispatchKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.dispatch_cb_base.value(),
        config.dispatch_cb_log_page_size.value(),
        config.dispatch_cb_pages.value(),
        config.my_dispatch_cb_sem_id.value(),
        config.upstream_dispatch_cb_sem_id.value(),

        config.dispatch_cb_blocks.value(),
        config.upstream_sync_sem.value(),
        config.command_queue_base_addr.value(),
        config.completion_queue_base_addr.value(),
        config.completion_queue_size.value(),

        config.downstream_cb_base.value(),
        config.downstream_cb_size.value(),
        config.my_downstream_cb_sem_id.value(),
        config.downstream_cb_sem_id.value(),

        config.split_dispatch_page_preamble_size.value(),
        config.split_prefetch.value(),
        config.prefetch_h_noc_xy.value(),
        config.prefetch_h_local_downstream_sem_addr.value(),
        config.prefetch_h_max_credits.value(),

        config.packed_write_max_unicast_sub_cmds.value(),
        config.dispatch_s_sync_sem_base_addr.value(),
        config.max_num_worker_sems.value(),
        config.max_num_go_signal_noc_data_entries.value(),
        config.mcast_go_signal_addr.value(),
        config.unicast_go_signal_addr.value(),
        config.distributed_dispatcher.value(),

        config.host_completion_q_wr_ptr.value(),
        config.dev_completion_q_wr_ptr.value(),
        config.dev_completion_q_rd_ptr.value(),

        config.is_d_variant.value(),
        config.is_h_variant.value(),
    };
    TT_ASSERT(compile_args.size() == 31);
    auto my_virtual_core = device->virtual_core_from_logical_core(this->logical_core, GetCoreType());
    auto upstream_virtual_core =
        device->virtual_core_from_logical_core(config.upstream_logical_core.value(), GetCoreType());
    auto downstream_virtual_core =
        device->virtual_core_from_logical_core(config.downstream_logical_core.value(), GetCoreType());
    auto downstream_s_virtual_core =
        device->virtual_core_from_logical_core(config.downstream_s_logical_core.value(), GetCoreType());

    auto my_virtual_noc_coords = device->virtual_noc0_coordinate(noc_selection.non_dispatch_noc, my_virtual_core);
    auto upstream_virtual_noc_coords =
        device->virtual_noc0_coordinate(noc_selection.upstream_noc, upstream_virtual_core);
    auto downstream_virtual_noc_coords =
        device->virtual_noc0_coordinate(noc_selection.downstream_noc, downstream_virtual_core);
    auto downstream_s_virtual_noc_coords =
        device->virtual_noc0_coordinate(noc_selection.downstream_noc, downstream_s_virtual_core);

    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(my_virtual_noc_coords.x)},
        {"MY_NOC_Y", std::to_string(my_virtual_noc_coords.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},
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
    tt::log_warning("Configure Dispatch (device {} core {})", device->id(), logical_core.str());
    for (uint32_t i = 0; i < dispatch_constants::DISPATCH_MESSAGE_ENTRIES; i++) {
        uint32_t dispatch_s_sync_sem_addr =
            dispatch_s_sync_sem_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        detail::WriteToDeviceL1(device, this->logical_core, dispatch_s_sync_sem_addr, zero, GetCoreType());
        detail::WriteToDeviceL1(device, this->logical_core, dispatch_message_addr, zero, GetCoreType());
    }

    // For DISPATCH_D, need to clear completion q events
    if (!this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        tt::log_warning("Configure Dispatch D Counters (device {} core {})", device->id(), logical_core.str());
        uint32_t completion_q0_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
        uint32_t completion_q1_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
        detail::WriteToDeviceL1(device, logical_core, completion_q0_last_event_ptr, zero, GetCoreType());
        detail::WriteToDeviceL1(device, logical_core, completion_q1_last_event_ptr, zero, GetCoreType());
    }
}
