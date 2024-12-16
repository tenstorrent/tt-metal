// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_kernels.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"

#define UNUSED_LOGICAL_CORE tt_cxy_pair(this->device->id(), 0, 0)
// TODO: Just to make match with previous implementation, remove later
#define UNUSED_LOGICAL_CORE_ADJUSTED tt_cxy_pair(servicing_device_id, 0, 0)
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

// Helper function to get upstream device in the tunnel from current device
chip_id_t GetUpstreamDeviceId(chip_id_t device_id) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (auto tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
        for (int idx = 0; idx < tunnel.size(); idx++) {
            if (tunnel[idx] == device_id) {
                // MMIO device doesn't have an upsream, just return itself
                return (idx == 0)? device_id : tunnel[idx - 1];
            }
        }
    }
    TT_ASSERT(false, "Could not find upstream device of Device {}", device_id);
    return device_id;
}

// Same thing for downstream, is ambiuous for mmio device though if it drives more than one tunnel
chip_id_t GetDownstreamDeviceId(chip_id_t device_id) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (auto tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
        for (int idx = 0; idx < tunnel.size(); idx++) {
            if (tunnel[idx] == device_id) {
                // End of tunnel doesn't have downstream, just return itself
                return (idx == tunnel.size() - 1)? device_id : tunnel[idx + 1];
            }
        }
    }
    TT_ASSERT(false, "Could not find downstream device of Device {}", device_id);
    return device_id;
}

// Helper function to get the tunnel stop of current device
uint32_t GetTunnelStop(chip_id_t device_id) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (auto tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
        for (uint32_t idx = 0; idx < tunnel.size(); idx++) {
            if (tunnel[idx] == device_id) {
                return idx;
            }
        }
    }
    TT_ASSERT(false, "Could not find tunnel stop of Device {}", device_id);
    return 0;
}

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
        {"DEVICE_ID", std::to_string(device_id)},
    };
    if (force_watcher_no_inline) {
        defines.insert({"WATCHER_NOINLINE", std::to_string(force_watcher_no_inline)});
    }
    if (tt::llrt::RunTimeOptions::get_instance().watcher_dispatch_disabled()) {
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
        // PREFETCH_H services a remote chip, and so has a different channel
        channel = tt::Cluster::instance().get_assigned_channel_for_device(servicing_device_id);
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);

        this->logical_core = dispatch_core_manager::instance().prefetcher_core(servicing_device_id, channel, cq_id);

        this->config.downstream_cb_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        if (tt::Cluster::instance().is_galaxy_cluster()) // TODO: whys is this hard-coded for galaxy?
            this->config.downstream_cb_pages = my_dispatch_constants.mux_buffer_pages(1);
        else
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
            this->config.downstream_cb_pages.value(),
            GetCoreType());
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
        this->config.my_dispatch_cb_sem_id = tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        this->config.dispatch_cb_blocks = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
        this->config.command_queue_base_addr = command_queue_start_addr;
        this->config.completion_queue_base_addr = completion_queue_start_addr;
        this->config.completion_queue_size = completion_queue_size;

        this->config.my_downstream_cb_sem_id = 0; // Unused

        this->config.split_dispatch_page_preamble_size = 0;
        this->config.split_prefetch = true;
        // TODO: why is this hard-coded to 1 CQ on Galaxy?
        if (tt::Cluster::instance().is_galaxy_cluster())
            this->config.prefetch_h_max_credits = my_dispatch_constants.mux_buffer_pages(1);
        else
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

        // To match with previous implementation, need to use grid size from mmio device. TODO: that doesn't seem correct though?
        auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        const auto &dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config(mmio_device_id);
        CoreCoord remote_grid_size = tt::get_compute_grid_size(mmio_device_id, device->num_hw_cqs(), dispatch_core_config);
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
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
    this->config.reserved = 0;
    this->config.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
    this->config.rx_queue_size_words = ((1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                        my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs())) >>
                                       4;
    this->config.mux_fan_in = this->upstream_kernels.size();
    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        this->config.remote_rx_network_type[idx] = DispatchRemoteNetworkType::NOC0;
    }

    this->config.tx_network_type = (uint32_t)DispatchRemoteNetworkType::NOC0;
    this->config.test_results_buf_addr_arg = 0;
    this->config.test_results_buf_size_bytes = 0;
    this->config.timeout_cycles = 0;
    this->config.output_depacketize = 0x0;
    this->config.output_depacketize_info = 0x0;

    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        // Only connected dispatchers need a semaphore. TODO: can initialize anyways, but this matches previous implementation
        if (dynamic_cast<DispatchKernel *>(this->upstream_kernels[idx])) {
            this->config.input_packetize_local_sem[idx] =
                tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        }
    }
}

void DemuxKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->servicing_device_id); // TODO: this can be mmio
    this->logical_core = dispatch_core_manager::instance().demux_core(this->servicing_device_id, channel, this->placement_cq_id);
    this->config.endpoint_id_start_index = 0xD1;
    this->config.rx_queue_start_addr_words = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED) >> 4;
    this->config.rx_queue_size_words = 0x10000 >> 4;

    this->config.remote_rx_network_type = DispatchRemoteNetworkType::NOC0;

    this->config.test_results_buf_addr_arg = 0;
    this->config.test_results_buf_size_bytes = 0;
    this->config.timeout_cycles = 0;

    // TODO: Do we need an upstream sem here?
    for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
        FDKernel *k = this->downstream_kernels[idx];
        this->config.remote_tx_queue_id[idx] = 0;
        this->config.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
        this->config.output_depacketize_cb_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        // Only connected dispatchers need a semaphore. TODO: can initialize anyways, but this matches previous implementation
        if (dynamic_cast<DispatchKernel *>(k)) {
            this->config.output_depacketize_local_sem_id[idx] =
                tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        }
        this->config.output_depacketize_remove_header[idx] = 1;
    }
}

void EthTunnelerKernel::GenerateStaticConfigs() {
    chip_id_t downstream_device_id = GetDownstreamDeviceId(device_id);
    // For MMIO devices, the above function just gets one of the possible downstream devices, we've populated this specific case with servicing_device_id
    if (device->is_mmio_capable())
        downstream_device_id = servicing_device_id;
    if (this->IsRemote()) {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(downstream_device_id);
        this->logical_core = dispatch_core_manager::instance().tunneler_core(device->id(), downstream_device_id, channel, cq_id);
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
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->servicing_device_id); // TODO: can be mmio
        this->logical_core = dispatch_core_manager::instance().mux_core(this->servicing_device_id, channel, placement_cq_id);
        this->config.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
        // TODO: why is this hard-coded NUM_CQS=1 for galaxy?
        if (tt::Cluster::instance().is_galaxy_cluster())
            this->config.rx_queue_size_words = my_dispatch_constants.mux_buffer_size(1) >> 4;
        else
            this->config.rx_queue_size_words = my_dispatch_constants.mux_buffer_size(device->num_hw_cqs()) >> 4;

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
        // Mux fowrads all VCs
        this->config.fwd_vc_count = this->config.vc_count;
    } else {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        this->logical_core = dispatch_core_manager::instance().demux_d_core(device->id(), channel, placement_cq_id);
        this->config.rx_queue_start_addr_words =
            hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED) >> 4;
        this->config.rx_queue_size_words = 0x8000 >> 4;

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

        this->config.fwd_vc_count = this->config.vc_count;
        uint32_t created_semaphores = 0;
        for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
            // Forwward VCs are the ones that don't connect to a prefetch
            if (auto pk = dynamic_cast<PrefetchKernel *>(this->downstream_kernels[idx])) {
                this->config.fwd_vc_count = this->config.fwd_vc_count.value() - 1;
                this->config.output_depacketize_local_sem[idx] = // TODO: to match for now, init one per vc after
                    tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
                created_semaphores++;
            }
        }
        if (created_semaphores == 0) // Just to match previous implementation
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());

        for (int idx = 0; idx < this->config.vc_count.value(); idx++) {
            this->config.output_depacketize_log_page_size[idx] = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
            this->config.output_depacketize_remove_header[idx] = 0;
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
        this->config.downstream_cb_base =
            (router_kernel->GetConfig().rx_queue_start_addr_words.value() << 4) +
            (router_kernel->GetConfig().rx_queue_size_words.value() << 4) * router_idx;
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
        this->config.prefetch_h_noc_xy = tt::tt_metal::hal.noc_xy_encoding(prefetch_h_kernel->GetVirtualCore().x, prefetch_h_kernel->GetVirtualCore().y);
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
    // Upstream, expect DISPATCH_D or TUNNELER
    TT_ASSERT(this->upstream_kernels.size() <= MAX_SWITCH_FAN_IN && this->upstream_kernels.size() > 0);
    uint32_t num_upstream_dispatchers = 0;
    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        FDKernel *k = this->upstream_kernels[idx];
        this->config.remote_rx_x[idx] = k->GetVirtualCore().x;
        this->config.remote_rx_y[idx] = k->GetVirtualCore().y;
        this->config.input_packetize_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE; // Does this ever change?
        if (auto dispatch_kernel = dynamic_cast<DispatchKernel *>(k)) {
            this->config.input_packetize[idx] = 0x1;
            this->config.input_packetize_upstream_sem[idx] = dispatch_kernel->GetConfig().my_downstream_cb_sem_id;
            this->config.remote_rx_queue_id[idx] = 1;
            num_upstream_dispatchers++;
        } else if (auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(k)) {
            // Don't need to packetize input from tunneler
            this->config.input_packetize[idx] = 0x0;
            this->config.input_packetize_upstream_sem[idx] = 0;
            this->config.remote_rx_queue_id[idx] = tunneler_kernel->GetConfig().vc_count.value() * 2 - 1;
        } else {
            TT_FATAL(false, "Unexpected kernel type upstream of MUX");
        }
    }
    uint32_t src_id = 0xC1 + (GetTunnelStop(device_id) - 1) * num_upstream_dispatchers;
    uint32_t dest_id = 0xD1 + (GetTunnelStop(device_id) - 1) * num_upstream_dispatchers;
    this->config.input_packetize_src_endpoint = packet_switch_4B_pack(src_id, src_id + 1, src_id + 2, src_id + 3);
    this->config.input_packetize_dest_endpoint = packet_switch_4B_pack(dest_id, dest_id + 1, dest_id + 2, dest_id + 3);

    // Downstream, expect TUNNELER
    TT_ASSERT(this->downstream_kernels.size() == 1);
    FDKernel *ds = this->downstream_kernels[0];
    auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(ds);
    TT_ASSERT(ds);
    this->config.remote_tx_queue_start_addr_words =
        tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
        (tunneler_kernel->GetConfig().vc_count.value() - 1) * tunneler_kernel->GetConfig().in_queue_size_words.value();
    this->config.remote_tx_queue_size_words = tunneler_kernel->GetConfig().in_queue_size_words;
    this->config.remote_tx_x = ds->GetVirtualCore().x;
    this->config.remote_tx_y = ds->GetVirtualCore().y;
    this->config.remote_tx_queue_id = tunneler_kernel->GetConfig().vc_count.value() - 1;
}

void DemuxKernel::GenerateDependentConfigs() {
    auto &my_dispatch_constants = dispatch_constants::get(GetCoreType());
    // Upstream, expect EthTunneler or DEMUX
    TT_ASSERT(this->upstream_kernels.size() == 1);
    if (auto us = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[0])) {
        this->config.remote_rx_x = us->GetVirtualCore().x;
        this->config.remote_rx_y = us->GetVirtualCore().y;
        this->config.remote_rx_queue_id = us->GetConfig().vc_count.value() * 2 - 1;
    } else if (auto us = dynamic_cast<DemuxKernel *>(this->upstream_kernels[0])) {
        this->config.remote_rx_x = us->GetVirtualCore().x;
        this->config.remote_rx_y = us->GetVirtualCore().y;
        this->config.remote_rx_queue_id = us->GetDownstreamPort(this) + 1; // TODO: can this be cleaned up?
        // TODO: why is just this one different? Just match previous implementation for now
        if (us->GetDownstreamPort(this) == 1)
            this->config.endpoint_id_start_index = this->config.endpoint_id_start_index.value() + this->downstream_kernels.size();
    } else {
        TT_FATAL(false, "Unexpected kernel type upstream of DEMUX");
    }

    // Downstream, expect DISPATCH_H or DEMUX
    TT_ASSERT(this->downstream_kernels.size() <= MAX_SWITCH_FAN_OUT && this->downstream_kernels.size() > 0);
    this->config.demux_fan_out = this->downstream_kernels.size();
    this->config.output_depacketize = 0; // Populated per downstream kernel
    for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
        FDKernel *k = this->downstream_kernels[idx];
        this->config.remote_tx_x[idx] = k->GetVirtualCore().x;
        this->config.remote_tx_y[idx] = k->GetVirtualCore().y;
        // Expect downstream to be either a DISPATCH or another DEMUX
        if (auto dispatch_kernel = dynamic_cast<DispatchKernel *>(k)) {
            this->config.remote_tx_queue_start_addr_words[idx] = dispatch_kernel->GetConfig().dispatch_cb_base.value() >> 4;
            this->config.remote_tx_queue_size_words[idx] =
                ((1 << dispatch_kernel->GetConfig().dispatch_cb_log_page_size.value()) *
                 dispatch_kernel->GetConfig().dispatch_cb_pages.value()) >>
                4;
            this->config.output_depacketize = this->config.output_depacketize.value() | (1 << idx); // Only depacketize for dispatch downstream
            this->config.output_depacketize_downstream_sem_id[idx] = dispatch_kernel->GetConfig().my_dispatch_cb_sem_id;
            uint32_t dest_map_array[4] = {0, 1, 2, 3};  // TODO: how to set these generically? Currently just matching the hard-coded previous implementation
            uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            this->config.dest_endpoint_output_map_hi = (uint32_t)(dest_endpoint_output_map >> 32);
            this->config.dest_endpoint_output_map_lo = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF);
        } else if (auto demux_kernel = dynamic_cast<DemuxKernel *>(k)) {
            this->config.remote_tx_queue_start_addr_words[idx] = demux_kernel->GetConfig().rx_queue_start_addr_words.value();
            this->config.remote_tx_queue_size_words[idx] = 0x1000; // TODO: hard-coded on previous implementation
            // Match previous implementation where downstream demux has output_depacketize fields zeroed out. TODO: can remove this later
            this->config.output_depacketize_downstream_sem_id[idx] = 0;
            uint64_t dest_endpoint_output_map;
            if (device->num_hw_cqs() == 1) {
                uint32_t dest_map_array[4] = {0, 0, 1, 1};  // TODO: how to set these generically? Currently just matching the hard-coded previous implementation
                dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            } else {
                uint32_t dest_map_array[8] = {0, 0, 0, 0, 1, 1, 1, 1};
                dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 8);
            }
            this->config.dest_endpoint_output_map_hi = (uint32_t)(dest_endpoint_output_map >> 32);
            this->config.dest_endpoint_output_map_lo = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF);
        } else {
            TT_FATAL(false, "Unexpected kernel type downstream of DEMUX");
        }
    }
    // TODO: this is just to match the previous implementation hard-code, remove later
    if (!tt::Cluster::instance().is_galaxy_cluster())
        this->config.output_depacketize = 0x3;
}

void EthTunnelerKernel::GenerateDependentConfigs() {
    if (this->IsRemote()) {
        // For remote tunneler, we don't actually have the device constructed for the paired tunneler, so can't pull
        // info from it. Core coord can be computed without the device, and relevant fields match this tunneler.
        chip_id_t downstream_device_id = GetDownstreamDeviceId(device_id);
        uint16_t downstream_channel = tt::Cluster::instance().get_assigned_channel_for_device(downstream_device_id);
        tt_cxy_pair paired_logical_core =
            dispatch_core_manager::instance().us_tunneler_core_local(downstream_device_id, downstream_channel, cq_id);
        tt_cxy_pair paired_physical_coord = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(paired_logical_core, CoreType::ETH);

        // Upstream, we expect a US_TUNNELER_LOCAL and one or more PACKET_ROUTER
        EthTunnelerKernel *tunneler_kernel = nullptr;
        std::vector<EthRouterKernel *> router_kernels;
        for (auto k : this->upstream_kernels) {
            if (auto rk = dynamic_cast<EthRouterKernel *>(k)) {
                router_kernels.push_back(rk);
            } else if (auto tk = dynamic_cast<EthTunnelerKernel *>(k)) {
                tunneler_kernel = tk;
            } else {
                TT_FATAL(false, "Unexpected kernelt tyoe downstream of TUNNELER");
            }
        }
        TT_ASSERT(tunneler_kernel && !tunneler_kernel->IsRemote());

        // Remote sender is the upstream packet router, one queue per router output lane.
        int remote_idx = 0;
        for (auto router_kernel : router_kernels) {
            uint32_t router_vc_count = router_kernel->GetConfig().vc_count.value();
            uint32_t router_fwd_vc_count = router_kernel->GetConfig().fwd_vc_count.value();
            for (int idx = 0; idx < router_fwd_vc_count; idx++) {
                this->config.remote_sender_x[remote_idx] = router_kernel->GetVirtualCore().x;
                this->config.remote_sender_y[remote_idx] = router_kernel->GetVirtualCore().y;
                // Router output lane ids start after it's input lane ids, assume after lanes that go to on-device kernels
                this->config.remote_sender_queue_id[remote_idx] = router_vc_count + idx + router_vc_count - router_fwd_vc_count;
                this->config.remote_sender_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                remote_idx++;
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
        auto other_ds_kernel = this->downstream_kernels[1];
        if (!ds_tunneler_kernel) {
            ds_tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->downstream_kernels[1]);
            auto other_ds_kernel = this->downstream_kernels[0];
        }
        TT_ASSERT(ds_tunneler_kernel == tunneler_kernel);
        for (uint32_t idx = 0; idx < this->config.vc_count.value(); idx++) {
            if (idx == this->config.vc_count.value() - 1) {
                // Last VC is the return VC, driving a DEMUX or MUX_D
                this->config.remote_receiver_x[idx] = other_ds_kernel->GetVirtualCore().x;
                this->config.remote_receiver_y[idx] = other_ds_kernel->GetVirtualCore().y;
                this->config.remote_receiver_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                if (auto demux_kernel = dynamic_cast<DemuxKernel *>(other_ds_kernel)) {
                    this->config.remote_receiver_queue_start[idx] = demux_kernel->GetConfig().rx_queue_start_addr_words;
                    this->config.remote_receiver_queue_size[idx] = demux_kernel->GetConfig().rx_queue_size_words;
                    this->config.remote_receiver_queue_id[idx] = 0; // DEMUX input queue id always 0
                } else if (auto mux_kernel = dynamic_cast<MuxKernel *>(other_ds_kernel)) {
                    this->config.remote_receiver_queue_start[idx] = mux_kernel->GetConfig().rx_queue_start_addr_words.value() + mux_kernel->GetConfig().rx_queue_size_words.value() * (mux_kernel->GetConfig().mux_fan_in.value() - 1);
                    this->config.remote_receiver_queue_size[idx] = mux_kernel->GetConfig().rx_queue_size_words;
                    // MUX input queue id for tunneler is the last one (counting up from 0)
                    this->config.remote_receiver_queue_id[idx] = mux_kernel->GetConfig().mux_fan_in.value() - 1;
                } else {
                    TT_FATAL(false, "Unexpected kernel type downstream of ETH_TUNNELER");
                }
            } else {
                this->config.remote_receiver_x[idx] = paired_physical_coord.x;
                this->config.remote_receiver_y[idx] = paired_physical_coord.y;
                // Tunneler upstream queue ids start counting up from 0
                this->config.remote_receiver_queue_id[idx] = idx;
                this->config.remote_receiver_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::ETH;
                this->config.remote_receiver_queue_start[idx] = this->config.in_queue_start_addr_words.value() +
                                                                idx * this->config.in_queue_size_words.value();
                this->config.remote_receiver_queue_size[idx] = this->config.in_queue_size_words;
            }
        }
    } else {
        // Upstream, we expect a US_TUNNELER_REMOTE and a MUX_D. Same deal where upstream tunneler may not be populated
        // yet since its device may not be created yet.
        chip_id_t upstream_device_id = GetUpstreamDeviceId(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        tt_cxy_pair paired_logical_core = dispatch_core_manager::instance().tunneler_core(upstream_device_id, device_id, channel, cq_id);
        tt_cxy_pair paired_physical_coord = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(paired_logical_core, CoreType::ETH);

        TT_ASSERT(this->upstream_kernels.size() == 2);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[0]);
        auto mux_kernel = dynamic_cast<MuxKernel *>(this->upstream_kernels[1]);
        if (!tunneler_kernel) {
            tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[1]);
            mux_kernel = dynamic_cast<MuxKernel *>(this->upstream_kernels[0]);
        }
        TT_ASSERT(tunneler_kernel && mux_kernel);
        TT_ASSERT(tunneler_kernel->IsRemote());
        for (uint32_t idx = 0; idx < this->config.vc_count.value(); idx++) {
            if (idx == this->config.vc_count.value() - 1) {
                // Last VC is the return VC, driven by the mux
                this->config.remote_sender_x[idx] = mux_kernel->GetVirtualCore().x;
                this->config.remote_sender_y[idx] = mux_kernel->GetVirtualCore().y;
                // MUX output queue id is counted after all of it's inputs
                this->config.remote_sender_queue_id[idx] = mux_kernel->GetConfig().mux_fan_in.value();
                this->config.remote_sender_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            } else {
                this->config.remote_sender_x[idx] = paired_physical_coord.x;
                this->config.remote_sender_y[idx] = paired_physical_coord.y;
                // Tunneler downstream queue ids start counting after the upstream ones
                this->config.remote_sender_queue_id[idx] = this->config.vc_count.value() + idx;
                this->config.remote_sender_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::ETH;
            }
        }

        // Downstream, we expect the same US_TUNNELER_REMOTE and one or more VC_PACKER_ROUTER
        EthTunnelerKernel *ds_tunneler_kernel = nullptr;
        std::vector<EthRouterKernel *> router_kernels;
        for (auto k : this->downstream_kernels) {
            if (auto rk = dynamic_cast<EthRouterKernel *>(k)) {
                router_kernels.push_back(rk);
            } else if (auto tk = dynamic_cast<EthTunnelerKernel *>(k)) {
                ds_tunneler_kernel = tk;
            } else {
                TT_FATAL(false, "Unexpected kernelt tyoe downstream of TUNNELER");
            }
        }
        TT_ASSERT(ds_tunneler_kernel && ds_tunneler_kernel == tunneler_kernel);

        // Remote receiver is the downstream router, one queue per router input lane
        int remote_idx = 0;
        for (auto router_kernel : router_kernels) {
            for (int idx = 0; idx < router_kernel->GetConfig().vc_count.value(); idx++) {
                this->config.remote_receiver_x[remote_idx] = router_kernel->GetVirtualCore().x;
                this->config.remote_receiver_y[remote_idx] = router_kernel->GetVirtualCore().y;
                this->config.remote_receiver_queue_id[remote_idx] = idx; // Queue ids start counting from 0 at input
                this->config.remote_receiver_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                this->config.remote_receiver_queue_start[remote_idx] = router_kernel->GetConfig().rx_queue_start_addr_words.value() + idx * router_kernel->GetConfig().rx_queue_size_words.value();
                this->config.remote_receiver_queue_size[remote_idx] = router_kernel->GetConfig().rx_queue_size_words.value();
                remote_idx++;
            }
        }
        // Last receiver connection is the return VC, connected to the paired tunneler
        uint32_t return_vc_id = this->config.vc_count.value() - 1;
        this->config.remote_receiver_x[return_vc_id] = paired_physical_coord.x;
        this->config.remote_receiver_y[return_vc_id] = paired_physical_coord.y;
        this->config.remote_receiver_queue_id[return_vc_id] = return_vc_id;
        this->config.remote_receiver_network_type[return_vc_id] = (uint32_t)DispatchRemoteNetworkType::ETH;
        this->config.remote_receiver_queue_start[return_vc_id] =
            this->config.in_queue_start_addr_words.value() +
            (return_vc_id)*this->config.in_queue_size_words.value();
        this->config.remote_receiver_queue_size[return_vc_id] = this->config.in_queue_size_words;
        this->config.inner_stop_mux_d_bypass = 0;
        // For certain chips in a tunnel (between first stop and end of tunnel, not including), we do a bypass
        if (this->config.vc_count.value() > (device->num_hw_cqs() + 1) && this->config.vc_count.value() < (4 * device->num_hw_cqs() + 1)) {
            this->config.inner_stop_mux_d_bypass = (return_vc_id << 24) | (((tunneler_kernel->GetConfig().vc_count.value() - device->num_hw_cqs()) * 2 - 1) << 16) | (paired_physical_coord.y << 8) | (paired_physical_coord.x);
        }
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

        uint32_t router_id = tunneler_kernel->GetRouterId(this, true);
        for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
            auto prefetch_kernel = dynamic_cast<PrefetchKernel *>(this->upstream_kernels[idx]);
            TT_ASSERT(prefetch_kernel);
            this->config.remote_tx_x[idx] = tunneler_kernel->GetVirtualCore().x;
            this->config.remote_tx_y[idx] = tunneler_kernel->GetVirtualCore().y;
            this->config.remote_tx_queue_id[idx] = idx + MAX_SWITCH_FAN_IN * router_id;
            this->config.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            this->config.remote_tx_queue_start_addr_words[idx] =
                tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
                (idx + router_id * MAX_SWITCH_FAN_IN) * tunneler_kernel->GetConfig().in_queue_size_words.value();
            this->config.remote_tx_queue_size_words[idx] = tunneler_kernel->GetConfig().in_queue_size_words.value();

            this->config.remote_rx_x[idx] = prefetch_kernel->GetVirtualCore().x;
            this->config.remote_rx_y[idx] = prefetch_kernel->GetVirtualCore().y;
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
        auto us_tunneler_kernel = dynamic_cast<EthTunnelerKernel *>(this->upstream_kernels[0]);
        TT_ASSERT(us_tunneler_kernel);
        // Upstream queues connect to the upstream tunneler, as many queues as we have VCs
        for (int idx = 0; idx < config.vc_count.value(); idx++) {
            this->config.remote_rx_x[idx] = us_tunneler_kernel->GetVirtualCore().x;
            this->config.remote_rx_y[idx] = us_tunneler_kernel->GetVirtualCore().y;
            // Queue id starts counting after the input VCs
            this->config.remote_rx_queue_id[idx] = us_tunneler_kernel->GetRouterQueueIdOffset(this, false) + idx;
            this->config.remote_rx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
        }

        // Downstream, expect PREFETCH_D/US_TUNNELER_REMOTE
        TT_ASSERT(this->downstream_kernels.size() <= MAX_SWITCH_FAN_OUT && this->downstream_kernels.size() > 0);
        std::vector<PrefetchKernel *> prefetch_kernels;
        EthTunnelerKernel *ds_tunneler_kernel = nullptr;
        for (auto k : this->downstream_kernels) {
            if (auto pk = dynamic_cast<PrefetchKernel *>(k)) {
                prefetch_kernels.push_back(pk);
            } else if (auto tk = dynamic_cast<EthTunnelerKernel *>(k)) {
                ds_tunneler_kernel = tk;
            } else {
                TT_FATAL(false, "Unexpected kernel type downstream of ROUTER");
            }
        }

        // Populate remote_tx_* for prefetch kernels, assume they are connected "first"
        uint32_t remote_idx = 0;
        for (auto prefetch_kernel : prefetch_kernels) {
            this->config.remote_tx_x[remote_idx] = prefetch_kernel->GetVirtualCore().x;
            this->config.remote_tx_y[remote_idx] = prefetch_kernel->GetVirtualCore().y;
            this->config.remote_tx_queue_id[remote_idx] = 0; // Prefetch queue id always 0
            this->config.remote_tx_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            this->config.remote_tx_queue_start_addr_words[remote_idx] = prefetch_kernel->GetConfig().cmddat_q_base.value() >> 4;
            this->config.remote_tx_queue_size_words[remote_idx] = prefetch_kernel->GetConfig().cmddat_q_size.value() >> 4;
            this->config.output_depacketize[remote_idx] = 1;
            this->config.output_depacketize_downstream_sem[remote_idx] = prefetch_kernel->GetConfig().my_upstream_cb_sem_id;
            remote_idx++;
        }

        // Populate remote_tx_* for the downstream tunneler, as many queues as we have fwd VCs
        if (ds_tunneler_kernel) {
            for (int idx = 0; idx < config.fwd_vc_count.value(); idx++) {
                this->config.remote_tx_x[remote_idx] = ds_tunneler_kernel->GetVirtualCore().x;
                this->config.remote_tx_y[remote_idx] = ds_tunneler_kernel->GetVirtualCore().y;
                this->config.remote_tx_queue_id[remote_idx] = ds_tunneler_kernel->GetRouterQueueIdOffset(this, true) + idx;
                this->config.remote_tx_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                this->config.remote_tx_queue_start_addr_words[remote_idx] = ds_tunneler_kernel->GetConfig().in_queue_start_addr_words.value() + ds_tunneler_kernel->GetConfig().in_queue_size_words.value() * (this->config.remote_tx_queue_id[remote_idx].value());
                this->config.remote_tx_queue_size_words[remote_idx] = ds_tunneler_kernel->GetConfig().in_queue_size_words.value();
                // Don't depacketize when sending to tunneler
                this->config.output_depacketize[remote_idx] = 0;
                this->config.output_depacketize_downstream_sem[remote_idx] = 0;
                remote_idx++;
            }
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
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)}, // Unused, remove later
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
    auto my_virtual_core = device->virtual_core_from_logical_core(this->logical_core, GetCoreType());
    auto upstream_virtual_core =
        device->virtual_core_from_logical_core(config.upstream_logical_core.value(), GetCoreType());
    auto downstream_virtual_core =
        device->virtual_core_from_logical_core(config.downstream_logical_core.value(), GetCoreType());
    auto downstream_s_virtual_core =
        device->virtual_core_from_logical_core(UNUSED_LOGICAL_CORE, GetCoreType());

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
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)}, // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(upstream_virtual_noc_coords.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_virtual_noc_coords.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)}, // Unused, remove later
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)}, // Unused, remove later
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
            // Zero out if input packetize not set to match previous implementation. TODO: don't have to do this
            if (config.input_packetize[idx].value() != 0) {
                compile_args[19 + idx] |= (config.input_packetize[idx].value() & 0xFF);
                compile_args[19 + idx] |= (config.input_packetize_log_page_size[idx].value() & 0xFF) << 8;
                compile_args[19 + idx] |= (config.input_packetize_upstream_sem[idx].value() & 0xFF) << 16;
                compile_args[19 + idx] |= (config.input_packetize_local_sem[idx].value() & 0xFF) << 24;
            }
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
            // To match previous implementation, zero these out if output_depacketize is not set. TODO: don't have to do this
            if (config.output_depacketize.value() & (1 << idx)) {
                compile_args[26 + idx] |= (config.output_depacketize_cb_log_page_size[idx].value() & 0xFF);
                compile_args[26 + idx] |= (config.output_depacketize_downstream_sem_id[idx].value() & 0xFF) << 8;
                compile_args[26 + idx] |= (config.output_depacketize_local_sem_id[idx].value() & 0xFF) << 16;
                compile_args[26 + idx] |= (config.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
            }
        }
    }
    TT_ASSERT(compile_args.size() == 30);
    const auto& grid_size = device->grid_size();
    tt_cxy_pair my_virtual_core = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(this->logical_core, GetCoreType());
    tt::log_warning("Demux {} has logical core {}, physical core {}", node_id, logical_core.str(), my_virtual_core.str());
    tt::log_warning("Noc XY={},{}", tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, my_virtual_core.x),std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, my_virtual_core.y)));
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
        config.vc_count.value(),
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
        // compile_args[21] = 84;
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
            if (config.output_depacketize[idx].value() & 0x1) { // To match previous implementation
                compile_args[26 + idx] |= (config.output_depacketize_log_page_size[idx].value() & 0xFF);
                compile_args[26 + idx] |= (config.output_depacketize_downstream_sem[idx].value() & 0xFF) << 8;
                compile_args[26 + idx] |= (config.output_depacketize_local_sem[idx].value() & 0xFF) << 16;
                compile_args[26 + idx] |= (config.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
            }
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
