// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_kernels.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"

#define UNUSED_LOGICAL_CORE tt_cxy_pair(this->device->id(), 0, 0)
// TODO: Just to make match with previous implementation, remove later
#define UNUSED_LOGICAL_CORE_ADJUSTED tt_cxy_pair(this->device->id() + tt::Cluster::instance().number_of_pci_devices(), 0, 0)
#define UNUSED_SEM_ID 0

static std::vector<string> dispatch_kernel_file_names = {
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp", // PREFETCH
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp", // PREFETCH_HD
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp", // PREFETCH_H
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp", // PREFETCH_D
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp", // DISPATCH
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp", // DISPATCH_HD
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp", // DISPATCH_H
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp", // DISPATCH_D
    "tt_metal/impl/dispatch/kernels/cq_dispatch_slave.cpp", // DISPATCH_S
    "",                                                     // MUX
    "tt_metal/impl/dispatch/kernels/packet_mux.cpp",        // MUX_D
    "tt_metal/impl/dispatch/kernels/packet_demux.cpp",      // DEMUX
    "",                                                     // DEMUX_D
    "tt_metal/impl/dispatch/kernels/vc_eth_tunneler.cpp",   // US_TUNNELER_LOCAL
    "tt_metal/impl/dispatch/kernels/vc_eth_tunneler.cpp",   // US_TUNNELER_REMOTE
    "tt_metal/impl/dispatch/kernels/vc_packet_router.cpp",  // PACKET_ROUTER_MUX
    "tt_metal/impl/dispatch/kernels/vc_packet_router.cpp",  // PACKET_ROUTER_DEMUX
    "" // COUNT
};
// static_assert(dispatch_kernel_file_names.size() == DispatchWorkerType::COUNT);

FDKernel *FDKernel::Generate(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection, DispatchWorkerType type) {
    switch (type) {
        case PREFETCH_HD:
            return new PrefetchKernel(node_id, device_id, cq_id, noc_selection, true, true);
        case PREFETCH_H:
            return new PrefetchKernel(node_id, device_id, cq_id, noc_selection, true, false);
        case PREFETCH_D:
            return new PrefetchKernel(node_id, device_id, cq_id, noc_selection, false, true);
        case DISPATCH_HD:
            return new DispatchKernel(node_id, device_id, cq_id, noc_selection, true, true);
        case DISPATCH_H:
            return new DispatchKernel(node_id, device_id, cq_id, noc_selection, true, false);
        case DISPATCH_D:
            return new DispatchKernel(node_id, device_id, cq_id, noc_selection, false, true);
        case DISPATCH_S:
            return new DispatchSKernel(node_id, device_id, cq_id, noc_selection);
        case MUX_D:
            return new MuxKernel(node_id, device_id, cq_id, noc_selection);
        case DEMUX:
            return new DemuxKernel(node_id, device_id, cq_id, noc_selection);
        case US_TUNNELER_REMOTE:
            return new EthTunnelerKernel(node_id, device_id, cq_id, noc_selection, true);
        case US_TUNNELER_LOCAL:
            return new EthTunnelerKernel(node_id, device_id, cq_id, noc_selection, false);
        case PACKET_ROUTER_MUX:
            return new EthRouterKernel(node_id, device_id, cq_id, noc_selection, true);
        case PACKET_ROUTER_DEMUX:
            return new EthRouterKernel(node_id, device_id, cq_id, noc_selection, false);
        default:
            TT_FATAL(false, "Unrecognized dispatch kernel type: {}.", type);
            return nullptr;
    }
}

void FDKernel::configure_kernel_variant(
    const string &path,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines_in,
    bool is_active_eth_core,
    bool send_to_brisc,
    bool force_watcher_no_inline) {

    // TODO: just pass in the programmable index
    uint32_t programmable_core_type_index = (GetCoreType() == CoreType::WORKER) ?
        hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX) :
        is_active_eth_core ? hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) :
        hal.get_programmable_core_type_index(HalProgrammableCoreType::IDLE_ETH);

    std::map<string, string> defines = {
        {"DISPATCH_KERNEL", "1"},
        {"FD_CORE_TYPE", std::to_string(programmable_core_type_index)},
    };
    if (force_watcher_no_inline) {
        defines.insert({"WATCHER_NOINLINE", std::to_string(force_watcher_no_inline)});
    }
    if (tt::llrt::OptionsG.watcher_dispatch_disabled()) {
        defines["FORCE_WATCHER_OFF"] = "1";
    }
    if (!tt::DPrintServerReadsDispatchCores(this->device)) {
        defines["FORCE_DPRINT_OFF"] = "1";
    }
    defines.insert(defines_in.begin(), defines_in.end());

    if (GetCoreType() == CoreType::WORKER) {
        tt::tt_metal::CreateKernel(
            *program,
            path,
            this->logical_core,
            tt::tt_metal::DataMovementConfig {
                .processor = send_to_brisc ? tt::tt_metal::DataMovementProcessor::RISCV_0 : tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = this->noc_selection.non_dispatch_noc,
                .compile_args = compile_args,
                .defines = defines
            }
        );
    } else {
        tt::tt_metal::CreateKernel(
            *program,
            path,
            this->logical_core,
            tt::tt_metal::EthernetConfig{
                .eth_mode = is_active_eth_core ? Eth::SENDER : Eth::IDLE,
                .noc = this->noc_selection.non_dispatch_noc,
                .compile_args = compile_args,
                .defines = defines
            }
        );
    }
}

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
        channel++; // TODO
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().prefetcher_core(device->id() + tt::Cluster::instance().number_of_pci_devices(), channel, cq_id); // TODO

        this->config.downstream_cb_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        this->config.downstream_cb_pages = my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());

        this->config.pcie_base = issue_queue_start_addr;
        this->config.pcie_size = issue_queue_size;
        this->config.prefetch_q_base = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        this->config.prefetch_q_size = my_dispatch_constants.prefetch_q_size();
        this->config.prefetch_q_rd_ptr_addr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        this->config.prefetch_q_pcie_rd_ptr_addr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);

        this->config.cmddat_q_base = my_dispatch_constants.cmddat_q_base();
        this->config.cmddat_q_size = my_dispatch_constants.cmddat_q_size();

        this->config.scratch_db_base = my_dispatch_constants.scratch_db_base();
        this->config.scratch_db_size = my_dispatch_constants.scratch_db_size();
        this->config.downstream_sync_sem_id = 0; // Unused for prefetch_h

        this->config.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        this->config.my_upstream_cb_sem_id = tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        this->config.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program,
            this->logical_core,
            my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs()),
            GetCoreType());  // TODO: Changes for Galaxy
        tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType()); // TODO: what is this third semaphore for?
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
        this->config.downstream_sync_sem_id = tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        this->config.my_upstream_cb_sem_id = tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        this->config.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program,
            this->logical_core,
            my_dispatch_constants.dispatch_buffer_pages(),
            GetCoreType());  // TODO: this is out of order to match previous implementation
        this->config.cmddat_q_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        this->config.cmddat_q_blocks = dispatch_constants::PREFETCH_D_BUFFER_BLOCKS;

        uint32_t dispatch_s_buffer_base = 0xff;
        if (device->dispatch_s_enabled() || true) { // Just to make it match previous implementation
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
        this->config.dispatch_s_cb_log_page_size = device->dispatch_s_enabled()? dispatch_constants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE : dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
    } else {
        TT_FATAL(false, "PrefetchKernel must be one of (or both) H and D variants");
    }
}

void DispatchKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    uint8_t cq_id = this->cq_id;
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());

    if (this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants
                                .get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
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
        channel++; // TODO
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device->sysmem_manager().get_completion_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().dispatcher_core(device->id() + tt::Cluster::instance().number_of_pci_devices(), channel, cq_id); // TODO
        this->config.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.dispatch_cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        this->config.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        this->config.my_dispatch_cb_sem_id = tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        this->config.command_queue_base_addr = command_queue_start_addr;
        this->config.completion_queue_base_addr = completion_queue_start_addr;
        this->config.completion_queue_size = completion_queue_size;

        this->config.my_downstream_cb_sem_id = 0; // Unused

        this->config.split_dispatch_page_preamble_size = 0;
        this->config.split_prefetch = true;
        this->config.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());

        this->config.packed_write_max_unicast_sub_cmds =
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
        this->config.dispatch_s_sync_sem_base_addr = 0; // Unused
        this->config.max_num_worker_sems = 1; // Used for array sizing, set to 1 even if unused
        this->config.max_num_go_signal_noc_data_entries = 1; // Used for array sizing, sset to 1 even if unused
        this->config.mcast_go_signal_addr = 0; // Unused
        this->config.unicast_go_signal_addr = 0; // Unused
        this->config.distributed_dispatcher = 0; // Unused

        this->config.host_completion_q_wr_ptr  = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_wr_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_rd_ptr= my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
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
        this->config.my_dispatch_cb_sem_id = tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        this->config.command_queue_base_addr = 0; // These are unused for DISPATCH_D
        this->config.completion_queue_base_addr = 0;
        this->config.completion_queue_size = 0;

        this->config.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program,
            this->logical_core,
            my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs()),
            GetCoreType()); // Apparently unused

        this->config.split_dispatch_page_preamble_size = sizeof(dispatch_packet_header_t);
        this->config.split_prefetch = true;
        this->config.prefetch_h_noc_xy = 0;
        this->config.prefetch_h_local_downstream_sem_addr = 1;
        this->config.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());

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

        this->config.host_completion_q_wr_ptr = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_wr_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
        this->config.dev_completion_q_rd_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }
}

void DispatchSKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    uint8_t cq_id = this->cq_id;
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());

    uint32_t dispatch_s_buffer_base = 0xff;
    if (device->dispatch_s_enabled()) {
        uint32_t dispatch_buffer_base = my_dispatch_constants.dispatch_buffer_base();
        if (GetCoreType() == CoreType::WORKER) {
            // dispatch_s is on the same Tensix core as dispatch_d. Shared resources. Offset CB start idx.
            dispatch_s_buffer_base = dispatch_buffer_base + (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                                                my_dispatch_constants.dispatch_buffer_pages();
        } else {
            // dispatch_d and dispatch_s are on different cores. No shared resources: dispatch_s CB starts at base.
            dispatch_s_buffer_base = dispatch_buffer_base;
        }
    }
    this->logical_core = dispatch_core_manager::instance().dispatcher_s_core(device->id(), channel, cq_id);
    this->config.cb_base = dispatch_s_buffer_base;
    this->config.cb_log_page_size = dispatch_constants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE;
    this->config.cb_size = my_dispatch_constants.dispatch_s_buffer_size();
    // used by dispatch_s to sync with prefetch
    this->config.my_dispatch_cb_sem_id = tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
    this->config.dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
    // used by dispatch_d to signal that dispatch_s can send go signal

    this->config.mcast_go_signal_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    this->config.unicast_go_signal_addr =
        (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
            ? hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
            : 0;
    this->config.distributed_dispatcher = (GetCoreType() == CoreType::ETH);
    this->config.worker_sem_base_addr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    this->config.max_num_worker_sems = dispatch_constants::DISPATCH_MESSAGE_ENTRIES;
    this->config.max_num_go_signal_noc_data_entries = dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
}

void MuxKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    this->logical_core = dispatch_core_manager::instance().mux_d_core(this->device->id(), channel, this->cq_id);
    this->config.vc_count = upstream_kernels.size() + 1; // TODO: update for deeper tunnels?
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
    this->config.reserved = 0;
    this->config.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
    this->config.rx_queue_size_words = ((1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                        my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs())) >>
                                       4;
    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        this->config.remote_rx_queue_id[idx] = 1;
        this->config.remote_rx_network_type[idx] = DispatchRemoteNetworkType::NOC0;
    }

    this->config.tx_network_type = (uint32_t)DispatchRemoteNetworkType::NOC0;
    this->config.test_results_buf_addr_arg = 0;
    this->config.test_results_buf_size_bytes = 0;
    this->config.timeout_cycles = 0;
    this->config.output_depacketize = 0x0;
    this->config.output_depacketize_info = 0x0;

    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        this->config.input_packetize[idx] = 0x1;
        this->config.input_packetize_local_sem[idx] =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
    }
}

void DemuxKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id() + tt::Cluster::instance().number_of_pci_devices()); // TODO: this is the downstream
    this->logical_core = dispatch_core_manager::instance().demux_core(this->device->id() + tt::Cluster::instance().number_of_pci_devices(), channel, this->cq_id);
    this->config.vc_count = downstream_kernels.size() + 1; // TODO: update for deeper tunnels?
    this->config.endpoint_id_start_index = 0xD1;
    this->config.rx_queue_start_addr_words = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED) >> 4;
    this->config.rx_queue_size_words = 0x10000 >> 4;

    this->config.remote_rx_network_type = DispatchRemoteNetworkType::NOC0;

    this->config.test_results_buf_addr_arg = 0;
    this->config.test_results_buf_size_bytes = 0;
    this->config.timeout_cycles = 0;
    this->config.output_depacketize = 0;

    // TODO: Do we need an upstream sem here?
    for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
        FDKernel *k = this->downstream_kernels[idx];
        this->config.remote_tx_queue_id[idx] = 0;
        this->config.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
        this->config.output_depacketize_cb_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        this->config.output_depacketize_local_sem_id[idx] =
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        this->config.output_depacketize_remove_header[idx] = 1;
    }
}

void EthTunnelerKernel::GenerateStaticConfigs() {
    this->downstream_device_id = device->id() + tt::Cluster::instance().number_of_pci_devices(); // TODO: update for galaxy...
    if (this->IsRemote()) {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(downstream_device_id.value());
        this->logical_core = dispatch_core_manager::instance().tunneler_core(device->id(), downstream_device_id.value(), channel, cq_id);
    } else {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        this->logical_core = dispatch_core_manager::instance().us_tunneler_core_local(device->id(), channel, cq_id);
    }
    this->config.endpoint_id_start_index = 0xDACADACA;
    this->config.in_queue_start_addr_words = 0x19000 >> 4;
    this->config.in_queue_size_words = 0x4000 >> 4;
    this->config.kernel_status_buf_addr_arg = 0x39000;
    this->config.kernel_status_buf_size_bytes = 0x7000;
    this->config.timeout_cycles = 0;
}

void EthRouterKernel::GenerateStaticConfigs() {
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
    if (this->as_mux) {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id() + tt::Cluster::instance().number_of_pci_devices()); // TODO: this is the downstream
        this->logical_core = dispatch_core_manager::instance().mux_core(device->id() + tt::Cluster::instance().number_of_pci_devices(), channel, cq_id);
        this->config.vc_count = upstream_kernels.size() + 1;
        this->config.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
        this->config.rx_queue_size_words = my_dispatch_constants.mux_buffer_size(device->num_hw_cqs()) >> 4;
        this->config.router_lanes = this->upstream_kernels.size();

        this->config.kernel_status_buf_addr_arg = 0;
        this->config.kernel_status_buf_size_bytes = 0;
        this->config.timeout_cycles = 0;
        this->config.output_depacketize = {0x0};
        this->config.output_depacketize_log_page_size = {0x0};
        this->config.output_depacketize_downstream_sem = {0x0};
        this->config.output_depacketize_local_sem = {0x0};
        this->config.output_depacketize_remove_header = {0x0};

        for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
            this->config.input_packetize[idx] = 0x1;
            this->config.input_packetize_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
            this->config.input_packetize_local_sem[idx] =
                tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
            this->config.remote_rx_queue_id[idx] = 1;
        }
    } else {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        this->logical_core = dispatch_core_manager::instance().demux_d_core(device->id(), channel, cq_id);
        this->config.vc_count = downstream_kernels.size() + 1;
        this->config.rx_queue_start_addr_words =
            hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED) >> 4;
        this->config.rx_queue_size_words = 0x8000 >> 4;
        this->config.router_lanes = this->downstream_kernels.size();

        this->config.kernel_status_buf_addr_arg = 0;
        this->config.kernel_status_buf_size_bytes = 0;
        this->config.timeout_cycles = 0;
        this->config.output_depacketize = {0x0};

        this->config.input_packetize = {0x0};
        this->config.input_packetize_log_page_size = {0x0};
        this->config.input_packetize_upstream_sem = {0x0};
        this->config.input_packetize_local_sem = {0x0};
        this->config.input_packetize_src_endpoint = {0x0};
        this->config.input_packetize_dst_endpoint = {0x0};

        for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
            this->config.output_depacketize_log_page_size[idx] = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
            this->config.output_depacketize_local_sem[idx] =
                tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
            this->config.output_depacketize_remove_header[idx] = 0;
            this->config.remote_rx_queue_id[idx] = this->config.vc_count.value() + idx;
        }
    }
}

void PrefetchKernel::GenerateDependentConfigs() {
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
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
        for (FDKernel *k : this->downstream_kernels) {
            if (auto dispatch_kernel = dynamic_cast<DispatchKernel *>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                this->config.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                this->config.downstream_cb_sem_id = dispatch_kernel->GetConfig().my_dispatch_cb_sem_id;
            } else if (auto dispatch_s_kernel = dynamic_cast<DispatchSKernel *>(k)) {
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
        auto router_kernel = dynamic_cast<EthRouterKernel *>(this->downstream_kernels[0]);
        TT_ASSERT(router_kernel);
        this->config.downstream_logical_core = router_kernel->GetLogicalCore();
        this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE_ADJUSTED;
        uint32_t router_idx = router_kernel->GetUpstreamPort(this); // Need the port that this connects to downstream
        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base() + my_dispatch_constants.mux_buffer_size(device->num_hw_cqs()) * router_idx;
        this->config.downstream_cb_sem_id = router_kernel->GetConfig().input_packetize_local_sem[router_idx];
        this->config.downstream_dispatch_s_cb_sem_id = 0; // No downstream DISPATCH_S in this case
    } else if (this->config.is_d_variant.value()) {
        // Upstream, expect just one ROUTER
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto router_kernel = dynamic_cast<EthRouterKernel *>(this->upstream_kernels[0]);
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
        for (FDKernel *k : this->downstream_kernels) {
            if (auto dispatch_kernel = dynamic_cast<DispatchKernel *>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                this->config.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                this->config.downstream_cb_sem_id = dispatch_kernel->GetConfig().my_dispatch_cb_sem_id;
            } else if (auto dispatch_s_kernel = dynamic_cast<DispatchSKernel *>(k)) {
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
            this->config.downstream_dispatch_s_cb_sem_id = device->dispatch_s_enabled()? UNUSED_SEM_ID : 1; // Just to make it match previous implementation
        }
    } else {
        TT_FATAL(false, "PrefetchKernel must be one of (or both) H and D variants");
    }
}

void DispatchKernel::GenerateDependentConfigs() {
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
    if (this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        // Upstream
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto prefetch_kernel = dynamic_cast<PrefetchKernel *>(this->upstream_kernels[0]);
        TT_ASSERT(prefetch_kernel);
        this->config.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        this->config.upstream_dispatch_cb_sem_id = prefetch_kernel->GetConfig().my_downstream_cb_sem_id;
        this->config.upstream_sync_sem = prefetch_kernel->GetConfig().downstream_sync_sem_id;

        // Downstream
        if (device->dispatch_s_enabled()) {
            TT_ASSERT(this->downstream_kernels.size() == 1);
            auto dispatch_s_kernel = dynamic_cast<DispatchSKernel *>(this->downstream_kernels[0]);
            TT_ASSERT(dispatch_s_kernel);
            this->config.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
        } else {
            // If no dispatch_s, no downstream
            TT_ASSERT(this->downstream_kernels.size() == 0);
            this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        }
        this->config.downstream_logical_core = UNUSED_LOGICAL_CORE;
        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base();
        this->config.downstream_cb_size = (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                          my_dispatch_constants.dispatch_buffer_pages();
        this->config.downstream_cb_sem_id = UNUSED_SEM_ID;
    } else if (this->config.is_h_variant.value()) {
        // Upstream, expect DEMUX
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto demux_kernel = dynamic_cast<DemuxKernel *>(this->upstream_kernels[0]);
        TT_ASSERT(demux_kernel);
        this->config.upstream_logical_core = demux_kernel->GetLogicalCore();
        int demux_idx = demux_kernel->GetDownstreamPort(this); // Need to know which port this kernel connects to upstream
        this->config.upstream_dispatch_cb_sem_id = demux_kernel->GetConfig().output_depacketize_local_sem_id[demux_idx].value();
        this->config.upstream_sync_sem = 0; // Unused

        // Downstream, no official downstream core but use the field to connect is to the PREFETCH_H that we need to
        // write to when resuming sending of commands post exec_buf stall.
        TT_ASSERT(this->downstream_kernels.size() == 1);
        auto prefetch_h_kernel = dynamic_cast<PrefetchKernel *>(this->downstream_kernels[0]);
        TT_ASSERT(prefetch_h_kernel);
        this->config.downstream_logical_core = UNUSED_LOGICAL_CORE_ADJUSTED;
        this->config.downstream_s_logical_core = UNUSED_LOGICAL_CORE_ADJUSTED;
        this->config.prefetch_h_noc_xy = NOC_XY_ENCODING(prefetch_h_kernel->GetPhysicalCore().x, prefetch_h_kernel->GetPhysicalCore().y);
        this->config.prefetch_h_local_downstream_sem_addr = prefetch_h_kernel->GetConfig().my_downstream_cb_sem_id;
        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base(); // Unused
        this->config.downstream_cb_size = (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * my_dispatch_constants.dispatch_buffer_pages(); // Unused
        this->config.downstream_cb_sem_id = 0; // Unused
    } else if (this->config.is_d_variant.value()) {
        // Upstream, expect a PREFETCH_D
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto prefetch_kernel = dynamic_cast<PrefetchKernel *>(this->upstream_kernels[0]);
        TT_ASSERT(prefetch_kernel);
        this->config.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        this->config.upstream_dispatch_cb_sem_id = prefetch_kernel->GetConfig().my_downstream_cb_sem_id;
        this->config.upstream_sync_sem = prefetch_kernel->GetConfig().downstream_sync_sem_id;
        // Downstream, expect a MUX_D and a DISPATCH_S if enabled
        auto dispatch_s_kernel = dynamic_cast<DispatchSKernel *>(this->downstream_kernels[0]);
        auto mux_kernel = dynamic_cast<MuxKernel *>(this->downstream_kernels[0]);
        if (device->dispatch_s_enabled()) {
            TT_ASSERT(this->downstream_kernels.size() == 2);
            mux_kernel = dynamic_cast<MuxKernel *>(this->downstream_kernels[1]);
            if (!dispatch_s_kernel) {
                dispatch_s_kernel = dynamic_cast<DispatchSKernel *>(this->downstream_kernels[1]);
                mux_kernel = dynamic_cast<MuxKernel *>(this->downstream_kernels[0]);
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
        int dispatch_d_idx = mux_kernel->GetUpstreamPort(this); // Need the port that this connects to downstream
        this->config.downstream_cb_size = (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs());
        this->config.downstream_cb_base = my_dispatch_constants.dispatch_buffer_base() + this->config.downstream_cb_size.value() * dispatch_d_idx;
        this->config.downstream_cb_sem_id = dispatch_d_idx;
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }
}

void DispatchSKernel::GenerateDependentConfigs() {
    // Upstream
    TT_ASSERT(this->upstream_kernels.size() == 1);
    auto prefetch_kernel = dynamic_cast<PrefetchKernel *>(this->upstream_kernels[0]);
    TT_ASSERT(prefetch_kernel);
    this->config.upstream_logical_core = prefetch_kernel->GetLogicalCore();
    this->config.upstream_dispatch_cb_sem_id = prefetch_kernel->GetConfig().my_dispatch_s_cb_sem_id;

    // Downstream
    TT_ASSERT(this->downstream_kernels.size() == 1);
    auto dispatch_kernel = dynamic_cast<DispatchKernel *>(this->downstream_kernels[0]);
    TT_ASSERT(dispatch_kernel);
    this->config.downstream_logical_core = dispatch_kernel->GetLogicalCore();
}

void MuxKernel::GenerateDependentConfigs() {
    // Upstream
    TT_ASSERT(this->upstream_kernels.size() <= MAX_SWITCH_FAN_IN && this->upstream_kernels.size() > 0);
    this->config.mux_fan_in = this->upstream_kernels.size();
    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        FDKernel *k = this->upstream_kernels[idx];
        auto dispatch_kernel = dynamic_cast<DispatchKernel *>(k);
        TT_ASSERT(dispatch_kernel);
        this->config.remote_rx_x[idx] = k->GetPhysicalCore().x;
        this->config.remote_rx_y[idx] = k->GetPhysicalCore().y;
        this->config.input_packetize_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE; // Does this ever change?
        this->config.input_packetize_upstream_sem[idx] = dispatch_kernel->GetConfig().my_downstream_cb_sem_id; // TODO: upstream can be tunneler as well?
    }
    uint32_t src_id = 0xC1 + (this->tunnel_stop - 1) * this->upstream_kernels.size();
    uint32_t dest_id = 0xD1 + (this->tunnel_stop - 1) * this->upstream_kernels.size();
    this->config.input_packetize_src_endpoint = packet_switch_4B_pack(src_id, src_id + 1, src_id + 2, src_id + 3);
    this->config.input_packetize_dest_endpoint = packet_switch_4B_pack(dest_id, dest_id + 1, dest_id + 2, dest_id + 3);

    // Downstream
    TT_ASSERT(this->downstream_kernels.size() == 1);
    FDKernel *ds = this->downstream_kernels[0];
    auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(ds);
    TT_ASSERT(ds);
    this->config.remote_tx_queue_start_addr_words =
        tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
        (this->config.vc_count.value() - 1) * tunneler_kernel->GetConfig().in_queue_size_words.value();
    this->config.remote_tx_queue_size_words = tunneler_kernel->GetConfig().in_queue_size_words;
    this->config.remote_tx_x = ds->GetPhysicalCore().x;
    this->config.remote_tx_y = ds->GetPhysicalCore().y;
    this->config.remote_tx_queue_id = this->config.vc_count.value() - 1;
}

void DemuxKernel::GenerateDependentConfigs() {
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
    // Upstream, expect EthTunneler
    TT_ASSERT(this->upstream_kernels.size() == 1);
    auto us = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[0]);
    TT_ASSERT(us);

    this->config.remote_rx_x = us->GetPhysicalCore().x;
    this->config.remote_rx_y = us->GetPhysicalCore().y;
    this->config.remote_rx_queue_id = this->config.vc_count.value() * 2 - 1;
    uint32_t dest_map_array[4] = {0, 1, 2, 3};  // TODO: how to set these generically?
    uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
    this->config.dest_endpoint_output_map_hi = (uint32_t)(dest_endpoint_output_map >> 32);
    this->config.dest_endpoint_output_map_lo = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF);

    // Downstream, expect DISPATCH_H or DEMUX
    TT_ASSERT(this->downstream_kernels.size() <= MAX_SWITCH_FAN_OUT && this->downstream_kernels.size() > 0);
    this->config.demux_fan_out = this->downstream_kernels.size();
    this->config.output_depacketize = ((1 << MAX_SWITCH_FAN_OUT) - 1);
    // this->config.output_depacketize = ((1 << MAX_SWITCH_FAN_OUT) - 1) >>
    //                                   (MAX_SWITCH_FAN_OUT - this->config.demux_fan_out.value());  // 1 for each vald output.
    for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
        FDKernel *k = this->downstream_kernels[idx];
        this->config.remote_tx_x[idx] = k->GetPhysicalCore().x;
        this->config.remote_tx_y[idx] = k->GetPhysicalCore().y;
        // Expect downstream to be either a DISPATCH or another DEMUX
        if (auto dispatch_kernel = dynamic_cast<DispatchKernel *>(k)) {
            this->config.remote_tx_queue_start_addr_words[idx] = dispatch_kernel->GetConfig().dispatch_cb_base.value() >> 4;
            this->config.remote_tx_queue_size_words[idx] =
                ((1 << dispatch_kernel->GetConfig().dispatch_cb_log_page_size.value()) *
                 dispatch_kernel->GetConfig().dispatch_cb_pages.value()) >>
                4;
            this->config.output_depacketize_downstream_sem_id[idx] = dispatch_kernel->GetConfig().my_dispatch_cb_sem_id;
        } else if (auto demux_kernel = dynamic_cast<DemuxKernel *>(k)) {
            this->config.remote_tx_queue_start_addr_words[idx] =
                my_dispatch_constants.mux_buffer_size(device->num_hw_cqs());
            this->config.remote_tx_queue_size_words[idx] =
                my_dispatch_constants.dispatch_buffer_base();
            this->config.output_depacketize_downstream_sem_id[idx] = 0; // TODO: This is wrong, hard-coded on main
        } else {
            TT_FATAL(false, "Unexpected kernel type downstream of DEMUX");
        }
    }
}

void EthTunnelerKernel::GenerateDependentConfigs() {
    if (this->IsRemote()) {
        // For remote tunneler, we don't actually have the device constructed for the paired tunneler, so can't pull
        // info from it. Core coord can be computed without the device, and relevant fields match this tunneler.
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(downstream_device_id.value());
        tt_cxy_pair paired_logical_core =
            dispatch_core_manager::instance().us_tunneler_core_local(downstream_device_id.value(), channel, cq_id);
        CoreCoord paired_physical_coord = tt::get_physical_core_coordinate(paired_logical_core, CoreType::ETH);

        // Upstream, we expect a US_TUNNELER_LOCAL and a PACKET_ROUTER
        TT_ASSERT(this->upstream_kernels.size() == 2);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[0]);
        auto router_kernel = dynamic_cast<EthRouterKernel *>(this->upstream_kernels[1]);
        if (!tunneler_kernel) {
            tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[1]);
            router_kernel = dynamic_cast<EthRouterKernel *>(this->upstream_kernels[0]);
        }
        TT_ASSERT(tunneler_kernel && router_kernel);
        TT_ASSERT(!tunneler_kernel->IsRemote());
        this->config.vc_count = router_kernel->GetConfig().vc_count;
        for (int idx = 0; idx < MAX_SWITCH_FAN_OUT; idx++) {
            if (router_kernel->GetConfig().remote_tx_queue_id[idx]) {
                this->config.remote_sender_x[idx] = router_kernel->GetPhysicalCore().x;
                this->config.remote_sender_y[idx] = router_kernel->GetPhysicalCore().y;
                this->config.remote_sender_queue_id[idx] = router_kernel->GetConfig().remote_tx_queue_id[idx].value() +
                                                           router_kernel->GetConfig().vc_count.value() -
                                                           1;  // TODO: why isn't it the same?
                this->config.remote_sender_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            }
        }
        // Last upstream connection is the return path from other tunneler
        this->config.remote_sender_x[this->config.vc_count.value() - 1] = paired_physical_coord.x;
        this->config.remote_sender_y[this->config.vc_count.value() - 1] = paired_physical_coord.y;
        this->config.remote_sender_queue_id[this->config.vc_count.value() - 1] = this->config.vc_count.value() * 2 - 1;
        this->config.remote_sender_network_type[this->config.vc_count.value() - 1] = (uint32_t)DispatchRemoteNetworkType::ETH;
        this->config.inner_stop_mux_d_bypass = 0;

        // Downstream, we expect the same US_TUNNELER_LOCAL and a DEMUX (tunnel start)/MUX_D (non-tunnel start)
        TT_ASSERT(this->downstream_kernels.size() == 2);
        auto ds_tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->downstream_kernels[0]);
        auto demux_kernel = dynamic_cast<DemuxKernel *>(this->downstream_kernels[1]);
        if (!ds_tunneler_kernel) {
            ds_tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->downstream_kernels[1]);
            demux_kernel = dynamic_cast<DemuxKernel *>(this->downstream_kernels[0]);
        }
        TT_ASSERT(ds_tunneler_kernel == tunneler_kernel);
        TT_ASSERT(ds_tunneler_kernel && demux_kernel);
        for (uint32_t idx = 0; idx < this->config.vc_count.value(); idx++) {
            if (idx == this->config.vc_count.value() - 1) {
                // Last VC is the return VC
                this->config.remote_receiver_x[idx] = demux_kernel->GetPhysicalCore().x;
                this->config.remote_receiver_y[idx] = demux_kernel->GetPhysicalCore().y;
                this->config.remote_receiver_queue_id[idx] = 0;
                this->config.remote_receiver_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                this->config.remote_receiver_queue_start[idx] = demux_kernel->GetConfig().rx_queue_start_addr_words;
                this->config.remote_receiver_queue_size[idx] = demux_kernel->GetConfig().rx_queue_size_words;
            } else {
                this->config.remote_receiver_x[idx] = paired_physical_coord.x;
                this->config.remote_receiver_y[idx] = paired_physical_coord.y;
                this->config.remote_receiver_queue_id[idx] = idx;
                this->config.remote_receiver_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::ETH;
                this->config.remote_receiver_queue_start[idx] = this->config.in_queue_start_addr_words.value() +
                                                                idx * this->config.in_queue_size_words.value();
                this->config.remote_receiver_queue_size[idx] = this->config.in_queue_size_words;
            }
        }
    } else {
        // Upstream, we expect a US_TUNNELER_REMOTE and a MUX_D
        TT_ASSERT(this->upstream_kernels.size() == 2);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[0]);
        auto mux_kernel = dynamic_cast<MuxKernel *>(this->upstream_kernels[1]);
        if (!tunneler_kernel) {
            tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[1]);
            mux_kernel = dynamic_cast<MuxKernel *>(this->upstream_kernels[0]);
        }
        TT_ASSERT(tunneler_kernel && mux_kernel);
        TT_ASSERT(tunneler_kernel->IsRemote());
        this->config.vc_count = mux_kernel->GetConfig().vc_count;
        for (uint32_t idx = 0; idx < this->config.vc_count.value(); idx++) {
            if (idx == this->config.vc_count.value() - 1) {
                // Last VC is the return VC
                this->config.remote_sender_x[idx] = mux_kernel->GetPhysicalCore().x;
                this->config.remote_sender_y[idx] = mux_kernel->GetPhysicalCore().y;
                this->config.remote_sender_queue_id[idx] = idx; // TODO: update for deeper tunnels
                this->config.remote_sender_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            } else {
                this->config.remote_sender_x[idx] = tunneler_kernel->GetPhysicalCore().x;
                this->config.remote_sender_y[idx] = tunneler_kernel->GetPhysicalCore().y;
                this->config.remote_sender_queue_id[idx] = this->config.vc_count.value() + idx;
                this->config.remote_sender_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::ETH;
            }
        }

        // Downstream, we expect the same US_TUNNELER_REMOTE and a VC_PACKER_ROUTER
        TT_ASSERT(this->downstream_kernels.size() == 2);
        auto ds_tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->downstream_kernels[0]);
        auto router_kernel = dynamic_cast<EthRouterKernel *>(this->downstream_kernels[1]);
        if (!ds_tunneler_kernel) {
            ds_tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->downstream_kernels[1]);
            router_kernel = dynamic_cast<EthRouterKernel *>(this->downstream_kernels[0]);
        }
        TT_ASSERT(ds_tunneler_kernel == tunneler_kernel);
        TT_ASSERT(ds_tunneler_kernel && router_kernel);
        for (int idx = 0; idx < MAX_SWITCH_FAN_IN; idx++) {
            if (router_kernel->GetConfig().remote_rx_queue_id[idx]) {
                this->config.remote_receiver_x[idx] = router_kernel->GetPhysicalCore().x;
                this->config.remote_receiver_y[idx] = router_kernel->GetPhysicalCore().y;
                this->config.remote_receiver_queue_id[idx] =
                    router_kernel->GetConfig().remote_rx_queue_id[idx].value() -
                    this->config.vc_count.value();  // TODO: why isn't this the same?
                this->config.remote_receiver_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                this->config.remote_receiver_queue_start[idx] = router_kernel->GetConfig().rx_queue_start_addr_words.value() + idx * router_kernel->GetConfig().rx_queue_size_words.value();
                this->config.remote_receiver_queue_size[idx] = router_kernel->GetConfig().rx_queue_size_words.value();
            }
        }
        uint32_t return_vc_id = this->config.vc_count.value() - 1;
        this->config.remote_receiver_x[return_vc_id] = tunneler_kernel->GetPhysicalCore().x;
        this->config.remote_receiver_y[return_vc_id] = tunneler_kernel->GetPhysicalCore().y;
        this->config.remote_receiver_queue_id[return_vc_id] = return_vc_id;
        this->config.remote_receiver_network_type[return_vc_id] = (uint32_t)DispatchRemoteNetworkType::ETH;
        this->config.remote_receiver_queue_start[return_vc_id] =
            tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
            (return_vc_id)*tunneler_kernel->GetConfig().in_queue_size_words.value();
        this->config.remote_receiver_queue_size[return_vc_id] = tunneler_kernel->GetConfig().in_queue_size_words;
        this->config.inner_stop_mux_d_bypass = 0; // TODO: This needs to change for deeper tunnels
    }
}

void EthRouterKernel::GenerateDependentConfigs() {
    if (this->as_mux) {
        // Upstream, expect PRETETCH_Hs
        TT_ASSERT(this->upstream_kernels.size() <= MAX_SWITCH_FAN_IN && this->upstream_kernels.size() > 0);

        // Downstream, expect US_TUNNELER_REMOTE
        TT_ASSERT(this->downstream_kernels.size() == 1);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->downstream_kernels[0]);
        TT_ASSERT(tunneler_kernel);

        uint32_t router_id = 0; // TODO: This needs to change for multi-mux
        for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
            auto prefetch_kernel = dynamic_cast<PrefetchKernel *>(this->upstream_kernels[idx]);
            TT_ASSERT(prefetch_kernel);
            this->config.remote_tx_x[idx] = tunneler_kernel->GetPhysicalCore().x;
            this->config.remote_tx_y[idx] = tunneler_kernel->GetPhysicalCore().y;
            this->config.remote_tx_queue_id[idx] = idx; // TODO: needs to be fixed for galaxy
            this->config.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            this->config.remote_tx_queue_start_addr_words[idx] =
                tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
                (idx + router_id * MAX_SWITCH_FAN_IN) * tunneler_kernel->GetConfig().in_queue_size_words.value();
            this->config.remote_tx_queue_size_words[idx] = tunneler_kernel->GetConfig().in_queue_size_words.value();

            this->config.remote_rx_x[idx] = prefetch_kernel->GetPhysicalCore().x;
            this->config.remote_rx_y[idx] = prefetch_kernel->GetPhysicalCore().y;
            this->config.remote_rx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;

            this->config.input_packetize_upstream_sem[idx] = prefetch_kernel->GetConfig().my_downstream_cb_sem_id.value();
        }

        uint32_t src_id_start = 0xA1 + router_id * MAX_SWITCH_FAN_IN;
        uint32_t dst_id_start = 0xB1 + router_id * MAX_SWITCH_FAN_IN;
        this->config.input_packetize_src_endpoint = {src_id_start, src_id_start + 1, src_id_start + 2, src_id_start + 3};
        this->config.input_packetize_dst_endpoint = {dst_id_start, dst_id_start + 1, dst_id_start + 2, dst_id_start + 3};
    } else {
        // Upstream, expect US_TUNNELER_LOCAL
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[0]);
        TT_ASSERT(tunneler_kernel);

        // Downstream, expect PREFETCH_D/US_TUNNELER_REMOTE
        TT_ASSERT(this->downstream_kernels.size() <= MAX_SWITCH_FAN_OUT && this->downstream_kernels.size() > 0);

        for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
            auto prefetch_kernel = dynamic_cast<PrefetchKernel *>(this->downstream_kernels[idx]);
            TT_ASSERT(prefetch_kernel);

            this->config.remote_tx_x[idx] = prefetch_kernel->GetPhysicalCore().x;
            this->config.remote_tx_y[idx] = prefetch_kernel->GetPhysicalCore().y;
            this->config.remote_tx_queue_id[idx] = 0;
            this->config.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            this->config.remote_tx_queue_start_addr_words[idx] =
                dispatch_constants::get(GetCoreType()).dispatch_buffer_base() >> 4;
            this->config.remote_tx_queue_size_words[idx] =
                dispatch_constants::get(GetCoreType()).prefetch_d_buffer_size() >> 4;

            this->config.remote_rx_x[idx] = tunneler_kernel->GetPhysicalCore().x;
            this->config.remote_rx_y[idx] = tunneler_kernel->GetPhysicalCore().y;
            this->config.remote_rx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;

            this->config.output_depacketize[idx] = 1;
            this->config.output_depacketize_downstream_sem[idx] = prefetch_kernel->GetConfig().my_upstream_cb_sem_id;
        }
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
    const auto& grid_size = device->grid_size();
    CoreCoord my_phys_core = tt::get_physical_core_coordinate(this->logical_core, GetCoreType());
    CoreCoord upstream_phys_core = tt::get_physical_core_coordinate(config.upstream_logical_core.value(), GetCoreType());
    CoreCoord downstream_phys_core = tt::get_physical_core_coordinate(config.downstream_logical_core.value(), GetCoreType());
    CoreCoord downstream_s_phys_core =
        tt::get_physical_core_coordinate(config.downstream_s_logical_core.value(), GetCoreType());
    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, my_phys_core.x))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, my_phys_core.y))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)}, // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, upstream_phys_core.x))},
        {"UPSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, upstream_phys_core.y))},
        {"DOWNSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, downstream_phys_core.x))},
        {"DOWNSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, downstream_phys_core.y))},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, downstream_s_phys_core.x))},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, downstream_s_phys_core.y))},
    };
    this->configure_kernel_variant(
        dispatch_kernel_file_names[PREFETCH],
        compile_args,
        defines,
        false,
        false,
        // TEMP: Disable function inlining on Prefetcher when watcher is enabled but no_inline is not specified to
        // respect code space
        tt::llrt::OptionsG.get_watcher_enabled() && (not tt::llrt::OptionsG.get_watcher_noinline()));
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
    const auto& grid_size = device->grid_size();
    CoreCoord my_phys_core = tt::get_physical_core_coordinate(this->logical_core, GetCoreType());
    CoreCoord upstream_phys_core = tt::get_physical_core_coordinate(config.upstream_logical_core.value(), GetCoreType());
    CoreCoord downstream_phys_core = tt::get_physical_core_coordinate(config.downstream_logical_core.value(), GetCoreType());
    CoreCoord downstream_s_phys_core =
        tt::get_physical_core_coordinate(config.downstream_s_logical_core.value(), GetCoreType());
    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, my_phys_core.x))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, my_phys_core.y))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, upstream_phys_core.x))},
        {"UPSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, upstream_phys_core.y))},
        {"DOWNSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, downstream_phys_core.x))},
        {"DOWNSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, downstream_phys_core.y))},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, downstream_s_phys_core.x))},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, downstream_s_phys_core.y))},
    };
    configure_kernel_variant(
        dispatch_kernel_file_names[DISPATCH],
        compile_args,
        defines,
        false,
        false,
        false);
}

void DispatchSKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.cb_base.value(),
        config.cb_log_page_size.value(),
        config.cb_size.value(),
        config.my_dispatch_cb_sem_id.value(),
        config.upstream_dispatch_cb_sem_id.value(),
        config.dispatch_s_sync_sem_base_addr.value(),
        config.mcast_go_signal_addr.value(),
        config.unicast_go_signal_addr.value(),
        config.distributed_dispatcher.value(),
        config.worker_sem_base_addr.value(),
        config.max_num_worker_sems.value(),
        config.max_num_go_signal_noc_data_entries.value(),
    };
    TT_ASSERT(compile_args.size() == 12);
    const auto& grid_size = device->grid_size();
    CoreCoord my_phys_core = tt::get_physical_core_coordinate(this->logical_core, GetCoreType());
    CoreCoord upstream_phys_core = tt::get_physical_core_coordinate(config.upstream_logical_core.value(), GetCoreType());
    CoreCoord downstream_phys_core = tt::get_physical_core_coordinate(config.downstream_logical_core.value(), GetCoreType());
    CoreCoord downstream_s_phys_core =
        tt::get_physical_core_coordinate(UNUSED_LOGICAL_CORE, GetCoreType()); // Unused, remove later
    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, my_phys_core.x))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, my_phys_core.y))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)}, // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, upstream_phys_core.x))},
        {"UPSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, upstream_phys_core.y))},
        {"DOWNSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, downstream_phys_core.x))},
        {"DOWNSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, downstream_phys_core.y))},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, downstream_s_phys_core.x))}, // Unused, remove later
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, downstream_s_phys_core.y))}, // Unused, remove later
    };
    configure_kernel_variant(dispatch_kernel_file_names[DISPATCH_S], compile_args, defines, false, true, false);
}

void MuxKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.reserved.value(),
        config.rx_queue_start_addr_words.value(),
        config.rx_queue_size_words.value(),
        config.mux_fan_in.value(),
        0, 0, 0, 0, // Populate remote_rx_config after
        config.remote_tx_queue_start_addr_words.value(),
        config.remote_tx_queue_size_words.value(),
        config.remote_tx_x.value(),
        config.remote_tx_y.value(),
        config.remote_tx_queue_id.value(),
        config.tx_network_type.value(),
        config.test_results_buf_addr_arg.value(),
        config.test_results_buf_size_bytes.value(),
        config.timeout_cycles.value(),
        config.output_depacketize.value(),
        config.output_depacketize_info.value(),
        0, 0, 0, 0, // Populate input_packetize_config after
        config.input_packetize_src_endpoint.value(),
        config.input_packetize_dest_endpoint.value()
    };
    for (int idx = 0; idx < MAX_SWITCH_FAN_IN; idx++) {
        if (config.remote_rx_x[idx]) {
            compile_args[4 + idx] |= (config.remote_rx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (config.remote_rx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (config.remote_rx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (config.remote_rx_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.input_packetize[idx]) {
            compile_args[19 + idx] |= (config.input_packetize[idx].value() & 0xFF);
            compile_args[19 + idx] |= (config.input_packetize_log_page_size[idx].value() & 0xFF) << 8;
            compile_args[19 + idx] |= (config.input_packetize_upstream_sem[idx].value() & 0xFF) << 16;
            compile_args[19 + idx] |= (config.input_packetize_local_sem[idx].value() & 0xFF) << 24;
        }
    }
    TT_ASSERT(compile_args.size() == 25);
    const auto& grid_size = device->grid_size();
    std::map<string, string> defines = { // All of these unused, remove later
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, 0))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, 0))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, 0))},
        {"UPSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"SKIP_NOC_LOGGING", "1"}
    };
    configure_kernel_variant(dispatch_kernel_file_names[MUX_D], compile_args, defines, false, false, false);
}

void DemuxKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.endpoint_id_start_index.value(),
        config.rx_queue_start_addr_words.value(),
        config.rx_queue_size_words.value(),
        config.demux_fan_out.value(),
        0, 0, 0, 0, // Populate remote_tx_config after
        0, 0, 0, 0, 0, 0, 0, 0, // Populate remote_tx_queue_start_addr_words & remote_tx_queue_size_words after
        config.remote_rx_x.value(),
        config.remote_rx_y.value(),
        config.remote_rx_queue_id.value(),
        config.remote_rx_network_type.value(),
        config.dest_endpoint_output_map_hi.value(),
        config.dest_endpoint_output_map_lo.value(),
        config.test_results_buf_addr_arg.value(),
        config.test_results_buf_size_bytes.value(),
        config.timeout_cycles.value(),
        config.output_depacketize.value(),
        0, 0, 0, 0 // Populate output_depacketize_config after
    };
    for (int idx = 0; idx < MAX_SWITCH_FAN_OUT; idx++) {
        if (config.remote_tx_x[idx]) {
            compile_args[4 + idx] |= (config.remote_tx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (config.remote_tx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (config.remote_tx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (config.remote_tx_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.remote_tx_queue_start_addr_words[idx]) {
            compile_args[8 + idx * 2] = config.remote_tx_queue_start_addr_words[idx].value();
            compile_args[9 + idx * 2] = config.remote_tx_queue_size_words[idx].value();
        }
        if (config.output_depacketize_cb_log_page_size[idx]) {
            compile_args[26 + idx] |= (config.output_depacketize_cb_log_page_size[idx].value() & 0xFF);
            compile_args[26 + idx] |= (config.output_depacketize_downstream_sem_id[idx].value() & 0xFF) << 8;
            compile_args[26 + idx] |= (config.output_depacketize_local_sem_id[idx].value() & 0xFF) << 16;
            compile_args[26 + idx] |= (config.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
        }
    }
    TT_ASSERT(compile_args.size() == 30);
    const auto& grid_size = device->grid_size();
    std::map<string, string> defines = { // All of these unused, remove later
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, 0))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, 0))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, 0))},
        {"UPSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"SKIP_NOC_LOGGING", "1"}
    };
    configure_kernel_variant(dispatch_kernel_file_names[DEMUX], compile_args, defines, false, false, false);
}

void EthTunnelerKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.endpoint_id_start_index.value(),
        config.vc_count.value(), // # Tunnel lanes = VC count
        config.in_queue_start_addr_words.value(),
        config.in_queue_size_words.value(),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Populate remote_receiver_config after
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Populate remote_receiver_queue_* after
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Populate remote_sender_* after
        config.kernel_status_buf_addr_arg.value(),
        config.kernel_status_buf_size_bytes.value(),
        config.timeout_cycles.value(),
        config.inner_stop_mux_d_bypass.value()
    };
    for (int idx = 0; idx < MAX_TUNNEL_LANES; idx++) {
        if (config.remote_receiver_x[idx]) {
            compile_args[4 + idx] |= (config.remote_receiver_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (config.remote_receiver_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (config.remote_receiver_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (config.remote_receiver_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.remote_receiver_queue_start[idx]) {
            compile_args[14 + idx * 2] = config.remote_receiver_queue_start[idx].value();
            compile_args[15 + idx * 2] = config.remote_receiver_queue_size[idx].value();
        } else {
            compile_args[15 + idx * 2] = 2; // Dummy size for unused VCs
        }
        if (config.remote_sender_x[idx]) {
            compile_args[34 + idx] |= (config.remote_sender_x[idx].value() & 0xFF);
            compile_args[34 + idx] |= (config.remote_sender_y[idx].value() & 0xFF) << 8;
            compile_args[34 + idx] |= (config.remote_sender_queue_id[idx].value() & 0xFF) << 16;
            compile_args[34 + idx] |= (config.remote_sender_network_type[idx].value() & 0xFF) << 24;
        }
    }
    TT_ASSERT(compile_args.size() == 48);
    const auto& grid_size = device->grid_size();
    std::map<string, string> defines = { // All of these unused, remove later
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, 0))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, 0))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, 0))},
        {"UPSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"SKIP_NOC_LOGGING", "1"}
    };
    configure_kernel_variant(
        dispatch_kernel_file_names[this->is_remote ? US_TUNNELER_REMOTE : US_TUNNELER_LOCAL],
        compile_args,
        defines,
        true,
        false,
        false);
}

void EthRouterKernel::CreateKernel() {
    std::vector<uint32_t> compile_args {
        0, // Unused
        config.rx_queue_start_addr_words.value(),
        config.rx_queue_size_words.value(),
        config.router_lanes.value(),
        0, 0, 0, 0, // Populate remote_tx_* after
        0, 0, 0, 0, 0, 0, 0, 0, // Populate remote_tx_queue_* after
        0, 0, 0, 0, // Populate remote_rx_* after
        0, 0, // Unused
        config.kernel_status_buf_addr_arg.value(),
        config.kernel_status_buf_size_bytes.value(),
        config.timeout_cycles.value(),
        0, // Populate output_depacketize after
        0, 0, 0, 0, // Populate output_depacketize_* after
        0, 0, 0, 0, // Populate input_packetize_* afterA
        0, // input_packetize_src_endpoint
        0, // input_packetize_dst_endpoint
    };
    // Some unused values, just hardcode them to match for checking purposes...
    if (!this->as_mux) {
        compile_args[0] = 0xB1;
        compile_args[21] = 84;
    }
    for (int idx = 0; idx < MAX_SWITCH_FAN_OUT; idx++) {
        if (config.remote_tx_x[idx]) {
            compile_args[4 + idx] |= (config.remote_tx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (config.remote_tx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (config.remote_tx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (config.remote_tx_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.remote_tx_queue_start_addr_words[idx]) {
            compile_args[8 + idx * 2] = config.remote_tx_queue_start_addr_words[idx].value();
            compile_args[9 + idx * 2] = config.remote_tx_queue_size_words[idx].value();
        }
        if (config.output_depacketize[idx]) {
            compile_args[25] |= (config.output_depacketize[idx].value() & 0x1) << idx;
            compile_args[26 + idx] |= (config.output_depacketize_log_page_size[idx].value() & 0xFF);
            compile_args[26 + idx] |= (config.output_depacketize_downstream_sem[idx].value() & 0xFF) << 8;
            compile_args[26 + idx] |= (config.output_depacketize_local_sem[idx].value() & 0xFF) << 16;
            compile_args[26 + idx] |= (config.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
        }
    }
    for (int idx = 0; idx < MAX_SWITCH_FAN_IN; idx++) {
        if (config.remote_rx_x[idx]) {
            compile_args[16 + idx] |= (config.remote_rx_x[idx].value() & 0xFF);
            compile_args[16 + idx] |= (config.remote_rx_y[idx].value() & 0xFF) << 8;
            compile_args[16 + idx] |= (config.remote_rx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[16 + idx] |= (config.remote_rx_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.input_packetize[idx]) {
            compile_args[30 + idx] |= (config.input_packetize[idx].value() & 0xFF);
            compile_args[30 + idx] |= (config.input_packetize_log_page_size[idx].value() & 0xFF) << 8;
            compile_args[30 + idx] |= (config.input_packetize_upstream_sem[idx].value() & 0xFF) << 16;
            compile_args[30 + idx] |= (config.input_packetize_local_sem[idx].value() & 0xFF) << 24;
        }
        if (config.input_packetize_src_endpoint[idx])
            compile_args[34] |= (config.input_packetize_src_endpoint[idx].value() & 0xFF) << (8 * idx);
        if (config.input_packetize_dst_endpoint[idx])
            compile_args[35] |= (config.input_packetize_dst_endpoint[idx].value() & 0xFF) << (8 * idx);
    }
    TT_ASSERT(compile_args.size() == 36);
    const auto& grid_size = device->grid_size();
    std::map<string, string> defines = { // All of these unused, remove later
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, 0))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, 0))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, 0))},
        {"UPSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"SKIP_NOC_LOGGING", "1"}
    };
    configure_kernel_variant(dispatch_kernel_file_names[PACKET_ROUTER_MUX], compile_args, defines, false, false, false);
}

void PrefetchKernel::ConfigureCore() {
    // Only H-type prefetchers need L1 configuration
    if (this->config.is_h_variant.value()) {
        tt::log_warning("Configure Prefetch H (device {} core {})", device->id(), logical_core.str());
        // Initialize the FetchQ
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
        auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
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
        uint32_t dispatch_s_sync_sem_addr = dispatch_s_sync_sem_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        uint32_t dispatch_message_addr = dispatch_message_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        detail::WriteToDeviceL1(device, this->logical_core, dispatch_s_sync_sem_addr, zero, GetCoreType());
        detail::WriteToDeviceL1(device, this->logical_core, dispatch_message_addr, zero, GetCoreType());
    }

    // For DISPATCH_D, need to clear completion q events
    if (!this->config.is_h_variant.value() && this->config.is_d_variant.value()) {
        tt::log_warning("Configure Dispatch D Counters (device {} core {})", device->id(), logical_core.str());
        uint32_t completion_q0_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
        uint32_t completion_q1_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
        detail::WriteToDeviceL1(device, logical_core, completion_q0_last_event_ptr, zero, GetCoreType());
        detail::WriteToDeviceL1(device, logical_core, completion_q1_last_event_ptr, zero, GetCoreType());
    }
}

void DispatchSKernel::ConfigureCore() {
    if (!this->device->distributed_dispatcher())
        return;
    // Just need to clear the dispatch message
    tt::log_warning("Configure Dispatch S (device {} core {})", device->id(), logical_core.str());
    std::vector<uint32_t> zero = {0x0};
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    uint32_t dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
    uint32_t dispatch_message_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    for (uint32_t i = 0; i < dispatch_constants::DISPATCH_MESSAGE_ENTRIES; i++) {
        uint32_t dispatch_s_sync_sem_addr = dispatch_s_sync_sem_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        uint32_t dispatch_message_addr = dispatch_message_base_addr + my_dispatch_constants.get_dispatch_message_offset(i);
        detail::WriteToDeviceL1(device, this->logical_core, dispatch_s_sync_sem_addr, zero, GetCoreType());
        detail::WriteToDeviceL1(device, this->logical_core, dispatch_message_addr, zero, GetCoreType());
    }
}
