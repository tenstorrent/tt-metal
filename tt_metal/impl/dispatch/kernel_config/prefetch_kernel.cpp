// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "prefetch_kernel.hpp"
#include "dispatch_kernel.hpp"
#include "dispatch_s_kernel.hpp"
#include "eth_router_kernel.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

void PrefetchKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    uint8_t cq_id = this->cq_id;
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());

    if (this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().prefetcher_core(device->id(), channel, cq_id);

        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.downstream_cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        this->config.downstream_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        this->config.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program, this->logical_core, my_dispatch_constants.dispatch_buffer_pages(), GetCoreType());

        this->config.pcie_base = issue_queue_start_addr;
        this->config.pcie_size = issue_queue_size;
        this->config.prefetch_q_base =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        this->config.prefetch_q_size = my_dispatch_constants.prefetch_q_size();
        this->config.prefetch_q_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        this->config.prefetch_q_pcie_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);

        this->config.cmddat_q_base = my_dispatch_constants.cmddat_q_base();
        this->config.cmddat_q_size = my_dispatch_constants.cmddat_q_size();

        this->config.scratch_db_base = my_dispatch_constants.scratch_db_base();
        this->config.scratch_db_size = my_dispatch_constants.scratch_db_size();
        this->config.downstream_sync_sem_id =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        // prefetch_d only
        this->config.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        this->config.my_upstream_cb_sem_id = 0;
        this->config.upstream_cb_sem_id = 0;
        this->config.cmddat_q_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        this->config.cmddat_q_blocks = dispatch_constants::PREFETCH_D_BUFFER_BLOCKS;

        uint32_t dispatch_s_buffer_base = 0xff;
        if (device->dispatch_s_enabled()) {
            uint32_t dispatch_buffer_base = my_dispatch_constants.dispatch_buffer_base();
            if (GetCoreType() == CoreType::WORKER) {
                // dispatch_s is on the same Tensix core as dispatch_d. Shared resources. Offset CB start idx.
                dispatch_s_buffer_base =
                    dispatch_buffer_base + (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                               my_dispatch_constants.dispatch_buffer_pages();
            } else {
                // dispatch_d and dispatch_s are on different cores. No shared resources: dispatch_s CB starts at base.
                dispatch_s_buffer_base = dispatch_buffer_base;
            }
        }
        this->config.dispatch_s_buffer_base = dispatch_s_buffer_base;
        this->config.my_dispatch_s_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program, this->logical_core, my_dispatch_constants.dispatch_s_buffer_pages(), GetCoreType());
        this->config.dispatch_s_buffer_size = my_dispatch_constants.dispatch_s_buffer_size();
        this->config.dispatch_s_cb_log_page_size = dispatch_constants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE;
    } else if (this->config.is_h_variant.value()) {
        // PREFETCH_H services a remote chip, and so has a different channel
        channel = tt::Cluster::instance().get_assigned_channel_for_device(servicing_device_id);
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().prefetcher_core(servicing_device_id, channel, cq_id);

        this->config.downstream_cb_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        if (tt::Cluster::instance().is_galaxy_cluster()) {  // TODO: whys is this hard-coded for galaxy?
            this->config.downstream_cb_pages = my_dispatch_constants.mux_buffer_pages(1);
        } else {
            this->config.downstream_cb_pages = my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());
        }

        this->config.pcie_base = issue_queue_start_addr;
        this->config.pcie_size = issue_queue_size;
        this->config.prefetch_q_base =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        this->config.prefetch_q_size = my_dispatch_constants.prefetch_q_size();
        this->config.prefetch_q_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        this->config.prefetch_q_pcie_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);

        this->config.cmddat_q_base = my_dispatch_constants.cmddat_q_base();
        this->config.cmddat_q_size = my_dispatch_constants.cmddat_q_size();

        this->config.scratch_db_base = my_dispatch_constants.scratch_db_base();
        this->config.scratch_db_size = my_dispatch_constants.scratch_db_size();
        this->config.downstream_sync_sem_id = 0;  // Unused for prefetch_h

        this->config.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        this->config.my_upstream_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        this->config.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program, this->logical_core, this->config.downstream_cb_pages.value(), GetCoreType());
        tt::tt_metal::CreateSemaphore(
            *program, this->logical_core, 0, GetCoreType());  // TODO: what is this third semaphore for?
        this->config.cmddat_q_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        this->config.cmddat_q_blocks = dispatch_constants::PREFETCH_D_BUFFER_BLOCKS;

        // PREFETCH_H has no DISPATCH_S
        this->config.dispatch_s_buffer_base = 0;
        this->config.my_dispatch_s_cb_sem_id = 0;
        this->config.dispatch_s_buffer_size = 0;
        this->config.dispatch_s_cb_log_page_size = 0;
    } else if (this->config.is_d_variant.value()) {
        this->logical_core = dispatch_core_manager::instance().prefetcher_d_core(device->id(), channel, cq_id);

        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.downstream_cb_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        this->config.downstream_cb_pages = my_dispatch_constants.dispatch_buffer_pages();

        this->config.pcie_base = 0;
        this->config.pcie_size = 0;
        this->config.prefetch_q_base = 0;
        this->config.prefetch_q_size = my_dispatch_constants.prefetch_q_size();
        this->config.prefetch_q_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        this->config.prefetch_q_pcie_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);

        this->config.cmddat_q_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.cmddat_q_size = my_dispatch_constants.prefetch_d_buffer_size();

        uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        this->config.scratch_db_base = (my_dispatch_constants.dispatch_buffer_base() +
                                        my_dispatch_constants.prefetch_d_buffer_size() + pcie_alignment - 1) &
                                       (~(pcie_alignment - 1));
        this->config.scratch_db_size = my_dispatch_constants.scratch_db_size();
        this->config.downstream_sync_sem_id =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        this->config.my_upstream_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        this->config.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program,
            this->logical_core,
            my_dispatch_constants.dispatch_buffer_pages(),
            GetCoreType());  // TODO: this is out of order to match previous implementation
        this->config.cmddat_q_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        this->config.cmddat_q_blocks = dispatch_constants::PREFETCH_D_BUFFER_BLOCKS;

        uint32_t dispatch_s_buffer_base = 0xff;
        if (device->dispatch_s_enabled() || true) {  // Just to make it match previous implementation
            uint32_t dispatch_buffer_base = my_dispatch_constants.dispatch_buffer_base();
            if (GetCoreType() == CoreType::WORKER) {
                // dispatch_s is on the same Tensix core as dispatch_d. Shared resources. Offset CB start idx.
                dispatch_s_buffer_base =
                    dispatch_buffer_base + (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                               my_dispatch_constants.dispatch_buffer_pages();
            } else {
                // dispatch_d and dispatch_s are on different cores. No shared resources: dispatch_s CB starts at base.
                dispatch_s_buffer_base = dispatch_buffer_base;
            }
        }
        this->config.dispatch_s_buffer_base = dispatch_s_buffer_base;
        this->config.my_dispatch_s_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program, this->logical_core, my_dispatch_constants.dispatch_s_buffer_pages(), GetCoreType());
        this->config.dispatch_s_buffer_size = my_dispatch_constants.dispatch_s_buffer_size();
        this->config.dispatch_s_cb_log_page_size = device->dispatch_s_enabled()
                                                       ? dispatch_constants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE
                                                       : dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
    } else {
        TT_FATAL(false, "PrefetchKernel must be one of (or both) H and D variants");
    }
}

void PrefetchKernel::GenerateDependentConfigs() {
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    if (this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        // Upstream
        TT_ASSERT(this->upstream_kernels.size() == 0);
        this->config.upstream_logical_core = UNUSED_LOGICAL_CORE;
        this->config.upstream_cb_sem_id = 0;  // Used in prefetch_d only

        // Downstream
        if (device->dispatch_s_enabled()) {
            TT_ASSERT(this->downstream_kernels.size() == 2);
        } else {
            TT_ASSERT(this->downstream_kernels.size() == 1);
        }
        bool found_dispatch = false;
        bool found_dispatch_s = false;
        for (FDKernel* k : this->downstream_kernels) {
            if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                this->config.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                this->config.downstream_cb_sem_id = dispatch_kernel->GetConfig().my_dispatch_cb_sem_id;
            } else if (auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(k)) {
                TT_ASSERT(!found_dispatch_s, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch_s = true;

                this->config.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
                this->config.downstream_dispatch_s_cb_sem_id = dispatch_s_kernel->GetConfig().my_dispatch_cb_sem_id;
            } else {
                TT_FATAL(false, "Unrecognized downstream kernel.");
            }
        }
        if (device->dispatch_s_enabled()) {
            // Should have found dispatch_s in the downstream kernels
            TT_ASSERT(found_dispatch && found_dispatch_s);
        } else {
            // No dispatch_s, just write 0s to the configs dependent on it
            TT_ASSERT(found_dispatch && ~found_dispatch_s);
            this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
            this->config.downstream_dispatch_s_cb_sem_id = UNUSED_SEM_ID;
        }
    } else if (this->config.is_h_variant.value()) {
        // Upstream, just host so no dispatch core
        TT_ASSERT(this->upstream_kernels.size() == 0);
        this->config.upstream_logical_core = UNUSED_LOGICAL_CORE_ADJUSTED;
        this->config.upstream_cb_sem_id = 0;  // Used in prefetch_d only

        // Downstream, expect just one ROUTER
        TT_ASSERT(this->downstream_kernels.size() == 1);
        auto router_kernel = dynamic_cast<EthRouterKernel*>(this->downstream_kernels[0]);
        TT_ASSERT(router_kernel);
        this->config.downstream_logical_core = router_kernel->GetLogicalCore();
        this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE_ADJUSTED;
        uint32_t router_idx = router_kernel->GetUpstreamPort(this);  // Need the port that this connects to downstream
        this->config.downstream_cb_base = (router_kernel->GetConfig().rx_queue_start_addr_words.value() << 4) +
                                          (router_kernel->GetConfig().rx_queue_size_words.value() << 4) * router_idx;
        this->config.downstream_cb_sem_id = router_kernel->GetConfig().input_packetize_local_sem[router_idx];
        this->config.downstream_dispatch_s_cb_sem_id = 0;  // No downstream DISPATCH_S in this case
    } else if (this->config.is_d_variant.value()) {
        // Upstream, expect just one ROUTER
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto router_kernel = dynamic_cast<EthRouterKernel*>(this->upstream_kernels[0]);
        TT_ASSERT(router_kernel);
        this->config.upstream_logical_core = router_kernel->GetLogicalCore();
        int router_idx = router_kernel->GetDownstreamPort(this);
        this->config.upstream_cb_sem_id = router_kernel->GetConfig().output_depacketize_local_sem[router_idx];

        // Downstream, expect a DISPATCH_D and s DISPATCH_S
        if (device->dispatch_s_enabled()) {
            TT_ASSERT(this->downstream_kernels.size() == 2);
        } else {
            TT_ASSERT(this->downstream_kernels.size() == 1);
        }
        bool found_dispatch = false;
        bool found_dispatch_s = false;
        for (FDKernel* k : this->downstream_kernels) {
            if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                this->config.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                this->config.downstream_cb_sem_id = dispatch_kernel->GetConfig().my_dispatch_cb_sem_id;
            } else if (auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(k)) {
                TT_ASSERT(!found_dispatch_s, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch_s = true;

                this->config.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
                this->config.downstream_dispatch_s_cb_sem_id = dispatch_s_kernel->GetConfig().my_dispatch_cb_sem_id;
            } else {
                TT_FATAL(false, "Unrecognized downstream kernel.");
            }
        }
        if (device->dispatch_s_enabled()) {
            // Should have found dispatch_s in the downstream kernels
            TT_ASSERT(found_dispatch && found_dispatch_s);
        } else {
            // No dispatch_s, just write 0s to the configs dependent on it
            TT_ASSERT(found_dispatch && ~found_dispatch_s);
            this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
            this->config.downstream_dispatch_s_cb_sem_id =
                device->dispatch_s_enabled() ? UNUSED_SEM_ID : 1;  // Just to make it match previous implementation
        }
    } else {
        TT_FATAL(false, "PrefetchKernel must be one of (or both) H and D variants");
    }
}

void PrefetchKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.downstream_cb_base.value(),
        config.downstream_cb_log_page_size.value(),
        config.downstream_cb_pages.value(),
        config.my_downstream_cb_sem_id.value(),
        config.downstream_cb_sem_id.value(),
        config.pcie_base.value(),
        config.pcie_size.value(),
        config.prefetch_q_base.value(),
        config.prefetch_q_size.value(),
        config.prefetch_q_rd_ptr_addr.value(),
        config.prefetch_q_pcie_rd_ptr_addr.value(),
        config.cmddat_q_base.value(),
        config.cmddat_q_size.value(),
        config.scratch_db_base.value(),
        config.scratch_db_size.value(),
        config.downstream_sync_sem_id.value(),
        config.cmddat_q_pages.value(),
        config.my_upstream_cb_sem_id.value(),
        config.upstream_cb_sem_id.value(),
        config.cmddat_q_log_page_size.value(),
        config.cmddat_q_blocks.value(),
        config.dispatch_s_buffer_base.value(),
        config.my_dispatch_s_cb_sem_id.value(),
        config.downstream_dispatch_s_cb_sem_id.value(),
        config.dispatch_s_buffer_size.value(),
        config.dispatch_s_cb_log_page_size.value(),
        config.is_d_variant.value(),
        config.is_h_variant.value(),
    };
    TT_ASSERT(compile_args.size() == 28);
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
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},  // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(upstream_virtual_noc_coords.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_virtual_noc_coords.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},
    };
    this->configure_kernel_variant(
        dispatch_kernel_file_names[PREFETCH],
        compile_args,
        defines,
        false,
        false,
        // TEMP: Disable function inlining on Prefetcher when watcher is enabled but no_inline is not specified to
        // respect code space
        tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled() &&
            (not tt::llrt::RunTimeOptions::get_instance().get_watcher_noinline()));
}

void PrefetchKernel::ConfigureCore() {
    // Only H-type prefetchers need L1 configuration
    if (this->config.is_h_variant.value()) {
        tt::log_warning("Configure Prefetch H (device {} core {})", device->id(), logical_core.str());
        // Initialize the FetchQ
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
        auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        std::vector<uint32_t> prefetch_q(my_dispatch_constants.prefetch_q_entries(), 0);
        uint32_t prefetch_q_base =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        std::vector<uint32_t> prefetch_q_rd_ptr_addr_data = {
            (uint32_t)(prefetch_q_base + my_dispatch_constants.prefetch_q_size())};
        uint32_t prefetch_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        uint32_t prefetch_q_pcie_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);
        uint32_t completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        uint32_t completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
        uint32_t dispatch_message_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
        uint32_t completion_q0_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
        uint32_t completion_q1_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
        std::vector<uint32_t> prefetch_q_pcie_rd_ptr_addr_data = {
            get_absolute_cq_offset(channel, cq_id, cq_size) + cq_start};
        detail::WriteToDeviceL1(
            device, this->logical_core, prefetch_q_rd_ptr, prefetch_q_rd_ptr_addr_data, GetCoreType());
        detail::WriteToDeviceL1(
            device, this->logical_core, prefetch_q_pcie_rd_ptr, prefetch_q_pcie_rd_ptr_addr_data, GetCoreType());
        detail::WriteToDeviceL1(device, this->logical_core, prefetch_q_base, prefetch_q, GetCoreType());
    }
}
