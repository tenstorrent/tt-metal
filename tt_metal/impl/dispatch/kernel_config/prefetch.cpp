// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefetch.hpp"

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
#include "dispatch.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch_s.hpp"
#include "eth_router.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/xy_pair.h>
#include "dispatch/system_memory_manager.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

void PrefetchKernel::GenerateStaticConfigs() {
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

        static_config_.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program_, logical_core_, my_dispatch_constants.dispatch_buffer_pages(), GetCoreType());

        static_config_.pcie_base = issue_queue_start_addr;
        static_config_.pcie_size = issue_queue_size;
        static_config_.prefetch_q_base =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        static_config_.prefetch_q_size = my_dispatch_constants.prefetch_q_size();
        static_config_.prefetch_q_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        static_config_.prefetch_q_pcie_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);

        static_config_.cmddat_q_base = my_dispatch_constants.cmddat_q_base();
        static_config_.cmddat_q_size = my_dispatch_constants.cmddat_q_size();

        static_config_.scratch_db_base = my_dispatch_constants.scratch_db_base();
        static_config_.scratch_db_size = my_dispatch_constants.scratch_db_size();
        static_config_.downstream_sync_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
        static_config_.ringbuffer_size = my_dispatch_constants.ringbuffer_size();

        // prefetch_d only
        static_config_.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        static_config_.my_upstream_cb_sem_id = 0;
        dependent_config_.upstream_cb_sem_id = 0;
        static_config_.cmddat_q_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        static_config_.cmddat_q_blocks = DispatchSettings::PREFETCH_D_BUFFER_BLOCKS;

        uint32_t dispatch_s_buffer_base = 0xff;
        if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            uint32_t dispatch_buffer_base = my_dispatch_constants.dispatch_buffer_base();
            if (GetCoreType() == CoreType::WORKER) {
                // dispatch_s is on the same Tensix core as dispatch_d. Shared resources. Offset CB start idx.
                dispatch_s_buffer_base =
                    dispatch_buffer_base + (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                               my_dispatch_constants.dispatch_buffer_pages();
            } else {
                // dispatch_d and dispatch_s are on different cores. No shared resources: dispatch_s CB starts at base.
                dispatch_s_buffer_base = dispatch_buffer_base;
            }
        }
        static_config_.dispatch_s_buffer_base = dispatch_s_buffer_base;
        static_config_.my_dispatch_s_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program_, logical_core_, my_dispatch_constants.dispatch_s_buffer_pages(), GetCoreType());
        static_config_.dispatch_s_buffer_size = my_dispatch_constants.dispatch_s_buffer_size();
        static_config_.dispatch_s_cb_log_page_size = DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE;
    } else if (static_config_.is_h_variant.value()) {
        // PREFETCH_H services a remote chip, and so has a different channel
        channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(servicing_device_id_);
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);

        static_config_.pcie_base = issue_queue_start_addr;
        static_config_.pcie_size = issue_queue_size;
        static_config_.prefetch_q_base =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        static_config_.prefetch_q_size = my_dispatch_constants.prefetch_q_size();
        static_config_.prefetch_q_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        static_config_.prefetch_q_pcie_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);

        static_config_.cmddat_q_base = my_dispatch_constants.cmddat_q_base();
        static_config_.cmddat_q_size = my_dispatch_constants.cmddat_q_size();

        static_config_.scratch_db_base = my_dispatch_constants.scratch_db_base();
        static_config_.scratch_db_size = my_dispatch_constants.scratch_db_size();
        static_config_.downstream_sync_sem_id = 0;  // Unused for prefetch_h
        static_config_.ringbuffer_size = my_dispatch_constants.ringbuffer_size();

        static_config_.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        static_config_.my_upstream_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        // Workaround for now. Need downstream to initialize my semaphore. Can't defer creating semaphore yet
        {
            uint32_t downstream_cb_pages;
            if (tt::tt_metal::MetalContext::instance().rtoptions().get_fd_fabric()) {
                downstream_cb_pages = my_dispatch_constants.prefetch_d_buffer_pages();
            } else if (tt::tt_metal::MetalContext::instance()
                           .get_cluster()
                           .is_galaxy_cluster()) {  // TODO: whys is this hard-coded for galaxy?
                downstream_cb_pages = my_dispatch_constants.mux_buffer_pages(1);
            } else {
                downstream_cb_pages = my_dispatch_constants.mux_buffer_pages(device_->num_hw_cqs());
            }
            static_config_.my_downstream_cb_sem_id =
                tt::tt_metal::CreateSemaphore(*program_, logical_core_, downstream_cb_pages, GetCoreType());
        }
        static_config_.cmddat_q_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        static_config_.cmddat_q_blocks = DispatchSettings::PREFETCH_D_BUFFER_BLOCKS;

        // PREFETCH_H has no DISPATCH_S
        static_config_.dispatch_s_buffer_base = 0;
        static_config_.my_dispatch_s_cb_sem_id = 0;
        static_config_.dispatch_s_buffer_size = 0;
        static_config_.dispatch_s_cb_log_page_size = 0;
    } else if (static_config_.is_d_variant.value()) {
        static_config_.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program_, logical_core_, my_dispatch_constants.dispatch_buffer_pages(), GetCoreType());

        static_config_.pcie_base = 0;
        static_config_.pcie_size = 0;
        static_config_.prefetch_q_base = 0;
        static_config_.prefetch_q_size = my_dispatch_constants.prefetch_q_size();
        static_config_.prefetch_q_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);
        static_config_.prefetch_q_pcie_rd_ptr_addr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);

        static_config_.cmddat_q_base = my_dispatch_constants.dispatch_buffer_base();
        static_config_.cmddat_q_size = my_dispatch_constants.prefetch_d_buffer_size();

        uint32_t pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
        static_config_.scratch_db_base = (my_dispatch_constants.dispatch_buffer_base() +
                                          my_dispatch_constants.prefetch_d_buffer_size() + pcie_alignment - 1) &
                                         (~(pcie_alignment - 1));
        static_config_.scratch_db_size = my_dispatch_constants.scratch_db_size();
        static_config_.downstream_sync_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
        static_config_.ringbuffer_size = my_dispatch_constants.ringbuffer_size();

        static_config_.cmddat_q_pages = my_dispatch_constants.prefetch_d_buffer_pages();
        static_config_.my_upstream_cb_sem_id =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
        static_config_.cmddat_q_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        static_config_.cmddat_q_blocks = DispatchSettings::PREFETCH_D_BUFFER_BLOCKS;

        uint32_t dispatch_s_buffer_base = 0xff;
        {  // Just to make it match previous implementation
            uint32_t dispatch_buffer_base = my_dispatch_constants.dispatch_buffer_base();
            if (GetCoreType() == CoreType::WORKER) {
                // dispatch_s is on the same Tensix core as dispatch_d. Shared resources. Offset CB start idx.
                dispatch_s_buffer_base =
                    dispatch_buffer_base + (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                               my_dispatch_constants.dispatch_buffer_pages();
            } else {
                // dispatch_d and dispatch_s are on different cores. No shared resources: dispatch_s CB starts at base.
                dispatch_s_buffer_base = dispatch_buffer_base;
            }
        }
        static_config_.dispatch_s_buffer_base = dispatch_s_buffer_base;
        static_config_.my_dispatch_s_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program_, logical_core_, my_dispatch_constants.dispatch_s_buffer_pages(), GetCoreType());
        static_config_.dispatch_s_buffer_size = my_dispatch_constants.dispatch_s_buffer_size();
        static_config_.dispatch_s_cb_log_page_size =
            MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()
                ? DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE
                : DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;
    } else {
        TT_FATAL(false, "PrefetchKernel must be one of (or both) H and D variants");
    }

    if ((static_config_.is_h_variant.value() ^ static_config_.is_d_variant.value()) &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_fd_fabric()) {
        create_edm_connection_sems(edm_connection_attributes_);
    }
}

void PrefetchKernel::GenerateDependentConfigs() {
    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        // Upstream
        TT_ASSERT(upstream_kernels_.size() == 0);
        dependent_config_.upstream_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.upstream_cb_sem_id = 0;  // Used in prefetch_d only

        // Downstream
        if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            TT_ASSERT(downstream_kernels_.size() == 2);
        } else {
            TT_ASSERT(downstream_kernels_.size() == 1);
        }
        bool found_dispatch = false;
        bool found_dispatch_s = false;
        for (FDKernel* k : downstream_kernels_) {
            if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                dependent_config_.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                dependent_config_.downstream_cb_sem_id =
                    dispatch_kernel->GetStaticConfig().my_dispatch_cb_sem_id.value();
                dependent_config_.downstream_cb_base = dispatch_kernel->GetStaticConfig().dispatch_cb_base.value();
                dependent_config_.downstream_cb_log_page_size =
                    dispatch_kernel->GetStaticConfig().dispatch_cb_log_page_size.value();
                dependent_config_.downstream_cb_pages = dispatch_kernel->GetStaticConfig().dispatch_cb_pages.value();
            } else if (auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(k)) {
                TT_ASSERT(!found_dispatch_s, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch_s = true;

                dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
                dependent_config_.downstream_dispatch_s_cb_sem_id =
                    dispatch_s_kernel->GetStaticConfig().my_dispatch_cb_sem_id;
            } else {
                TT_FATAL(false, "Unrecognized downstream kernel.");
            }
        }
        if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            // Should have found dispatch_s in the downstream kernels
            TT_ASSERT(found_dispatch && found_dispatch_s);
        } else {
            // No dispatch_s, just write 0s to the configs dependent on it
            TT_ASSERT(found_dispatch && !found_dispatch_s);
            dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
            dependent_config_.downstream_dispatch_s_cb_sem_id = UNUSED_SEM_ID;
        }
        dependent_config_.num_hops = 0;
    } else if (static_config_.is_h_variant.value()) {
        // Upstream, just host so no dispatch core
        TT_ASSERT(upstream_kernels_.size() == 0);
        dependent_config_.upstream_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.upstream_cb_sem_id = 0;  // Used in prefetch_d only
        // May be overwritten below
        dependent_config_.num_hops = 0;

        // Process downstream
        // ROUTER ||
        // PREFETCH_D ||
        // FABRIC_MUX
        for (FDKernel* ds_kernel : downstream_kernels_) {
            if (auto router_kernel = dynamic_cast<EthRouterKernel*>(ds_kernel)) {
                dependent_config_.downstream_logical_core = router_kernel->GetLogicalCore();
                dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
                uint32_t router_idx =
                    router_kernel->GetUpstreamPort(this);  // Need the port that this connects to downstream
                auto downstream_buffer_size = router_kernel->GetStaticConfig().rx_queue_size_words.value() << 4;
                dependent_config_.downstream_cb_base =
                    (router_kernel->GetStaticConfig().rx_queue_start_addr_words.value() << 4) +
                    downstream_buffer_size * router_idx;
                dependent_config_.downstream_cb_sem_id =
                    router_kernel->GetStaticConfig().input_packetize_local_sem[router_idx];
                dependent_config_.downstream_dispatch_s_cb_sem_id = 0;  // No downstream DISPATCH_S in this case

                dependent_config_.downstream_cb_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
                dependent_config_.downstream_cb_pages =
                    downstream_buffer_size / (1 << DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);
            } else if (auto prefetch_d = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
                TT_ASSERT(
                    prefetch_d->GetStaticConfig().is_d_variant.value() &&
                    !prefetch_d->GetStaticConfig().is_h_variant.value());

                dependent_config_.downstream_logical_core = prefetch_d->GetLogicalCore();
                dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
                dependent_config_.downstream_cb_base = prefetch_d->GetStaticConfig().cmddat_q_base.value();
                dependent_config_.downstream_cb_sem_id = prefetch_d->GetStaticConfig().my_upstream_cb_sem_id.value();
                dependent_config_.downstream_dispatch_s_cb_sem_id = 0;

                static_assert(
                    DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE ==
                    DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
                dependent_config_.downstream_cb_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
                dependent_config_.downstream_cb_pages = prefetch_d->GetStaticConfig().cmddat_q_pages.value();
                dependent_config_.num_hops = tt::tt_metal::get_num_hops(device_id_, prefetch_d->GetDeviceId());
            } else if (auto fabric_mux = dynamic_cast<tt::tt_metal::RelayMux*>(ds_kernel)) {
                constexpr tt::tt_fabric::FabricMuxChannelType ch_type =
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
                tt::tt_metal::assemble_fabric_mux_client_config_args(
                    node_id_, ch_type, fabric_mux, dependent_config_.fabric_mux_client_config);
            } else {
                TT_FATAL(false, "PREFETCH_H Downstream - Unimplemented path");
            }
        }
    } else if (static_config_.is_d_variant.value()) {
        // Upstream
        // One ROUTER or direct connection to PREFETCH_H if using fabric
        TT_ASSERT(upstream_kernels_.size() == 1);
        // May be overwritten below
        dependent_config_.num_hops = 0;
        if (auto router_kernel = dynamic_cast<EthRouterKernel*>(upstream_kernels_[0])) {
            dependent_config_.upstream_logical_core = router_kernel->GetLogicalCore();
            int router_idx = router_kernel->GetDownstreamPort(this);
            dependent_config_.upstream_cb_sem_id =
                router_kernel->GetStaticConfig().output_depacketize_local_sem[router_idx];
        } else if (auto prefetch_h = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0])) {
            dependent_config_.upstream_logical_core = prefetch_h->GetLogicalCore();
            dependent_config_.upstream_cb_sem_id = prefetch_h->GetStaticConfig().my_downstream_cb_sem_id.value();
            dependent_config_.num_hops = tt::tt_metal::get_num_hops(prefetch_h->GetDeviceId(), device_id_);
        } else {
            TT_FATAL(false, "Path not implemented");
        }

        // Downstream
        // DISPATCH_D || DISPATCH_S || FABRIC_MUX
        bool found_dispatch = false;
        bool found_dispatch_s = false;
        bool found_relay_mux = false;
        for (FDKernel* k : downstream_kernels_) {
            if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                dependent_config_.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                dependent_config_.downstream_cb_sem_id =
                    dispatch_kernel->GetStaticConfig().my_dispatch_cb_sem_id.value();
                dependent_config_.downstream_cb_base = dispatch_kernel->GetStaticConfig().dispatch_cb_base.value();
                dependent_config_.downstream_cb_log_page_size =
                    dispatch_kernel->GetStaticConfig().dispatch_cb_log_page_size.value();
                dependent_config_.downstream_cb_pages = dispatch_kernel->GetStaticConfig().dispatch_cb_pages.value();
            } else if (auto dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(k)) {
                TT_ASSERT(!found_dispatch_s, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch_s = true;

                dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
                dependent_config_.downstream_dispatch_s_cb_sem_id =
                    dispatch_s_kernel->GetStaticConfig().my_dispatch_cb_sem_id;
            } else if (auto relay_mux = dynamic_cast<tt::tt_metal::RelayMux*>(k)) {
                TT_ASSERT(!found_relay_mux, "PREFETCH_D kernel has multiple downstream RELAY_MUX kernels.");
                found_relay_mux = true;
                constexpr tt::tt_fabric::FabricMuxChannelType ch_type =
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL;
                tt::tt_metal::assemble_fabric_mux_client_config_args(
                    node_id_, ch_type, relay_mux, dependent_config_.fabric_mux_client_config);
            } else {
                TT_FATAL(false, "Unrecognized downstream kernel.");
            }
        }
        // No check needed for found relay mux. A direct connection to PREFETCH_H on the same core
        // is possible (used in test prefetcher) and does not need tunneling.
        if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            // Should have found dispatch_s in the downstream kernels
            TT_ASSERT(found_dispatch && found_dispatch_s);
        } else {
            // No dispatch_s, just write 0s to the configs dependent on it
            TT_ASSERT(found_dispatch && !found_dispatch_s);
            dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
            dependent_config_.downstream_dispatch_s_cb_sem_id =
                MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()
                    ? UNUSED_SEM_ID
                    : 1;  // Just to make it match previous implementation
        }
    } else {
        TT_FATAL(false, "PrefetchKernel must be one of (or both) H and D variants");
    }
}

void PrefetchKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        dependent_config_.downstream_cb_base.value(),
        dependent_config_.downstream_cb_log_page_size.value(),
        dependent_config_.downstream_cb_pages.value(),
        static_config_.my_downstream_cb_sem_id.value(),
        dependent_config_.downstream_cb_sem_id.value(),
        static_config_.pcie_base.value(),
        static_config_.pcie_size.value(),
        static_config_.prefetch_q_base.value(),
        static_config_.prefetch_q_size.value(),
        static_config_.prefetch_q_rd_ptr_addr.value(),
        static_config_.prefetch_q_pcie_rd_ptr_addr.value(),
        static_config_.cmddat_q_base.value(),
        static_config_.cmddat_q_size.value(),
        static_config_.scratch_db_base.value(),
        static_config_.scratch_db_size.value(),
        static_config_.downstream_sync_sem_id.value(),
        static_config_.cmddat_q_pages.value(),
        static_config_.my_upstream_cb_sem_id.value(),
        dependent_config_.upstream_cb_sem_id.value(),
        static_config_.cmddat_q_log_page_size.value(),
        static_config_.cmddat_q_blocks.value(),
        static_config_.dispatch_s_buffer_base.value(),
        static_config_.my_dispatch_s_cb_sem_id.value(),
        dependent_config_.downstream_dispatch_s_cb_sem_id.value(),
        static_config_.dispatch_s_buffer_size.value(),
        static_config_.dispatch_s_cb_log_page_size.value(),
        static_config_.ringbuffer_size.value(),

        static_config_.fabric_header_rb_base.value(),
        static_config_.fabric_header_rb_entries.value(),
        static_config_.my_fabric_sync_status_addr.value(),

        dependent_config_.fabric_mux_client_config.virtual_x.value_or(0),
        dependent_config_.fabric_mux_client_config.virtual_y.value_or(0),
        dependent_config_.fabric_mux_client_config.num_buffers_per_channel.value_or(0),
        dependent_config_.fabric_mux_client_config.channel_buffer_size_bytes.value_or(0),
        dependent_config_.fabric_mux_client_config.channel_base_address.value_or(0),
        dependent_config_.fabric_mux_client_config.connection_info_address.value_or(0),
        dependent_config_.fabric_mux_client_config.connection_handshake_address.value_or(0),
        dependent_config_.fabric_mux_client_config.flow_control_address.value_or(0),
        dependent_config_.fabric_mux_client_config.buffer_index_address.value_or(0),
        dependent_config_.fabric_mux_client_config.status_address.value_or(0),
        dependent_config_.fabric_mux_client_config.termination_signal_address.value_or(0),
        dependent_config_.fabric_mux_client_config.worker_credits_stream_id.value_or(0),

        edm_connection_attributes_.worker_flow_control_sem,
        edm_connection_attributes_.worker_teardown_sem,
        edm_connection_attributes_.worker_buffer_index_sem,

        dependent_config_.num_hops.value(),

        static_config_.is_d_variant.value(),
        static_config_.is_h_variant.value(),
    };

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

    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(my_virtual_noc_coords.x)},
        {"MY_NOC_Y", std::to_string(my_virtual_noc_coords.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},  // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(upstream_virtual_noc_coords.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_virtual_noc_coords.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},
    };
    // Compile at Os on IERISC to fit in code region.
    auto optimization_level = (GetCoreType() == CoreType::WORKER) ? KernelBuildOptLevel::O2 : KernelBuildOptLevel::Os;
    configure_kernel_variant(
        dispatch_kernel_file_names[PREFETCH],
        compile_args,
        defines,
        false,
        true,
        // TEMP: Disable function inlining on Prefetcher when watcher is enabled but no_inline is not specified to
        // respect code space
        tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled() &&
            (not tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_noinline()),
        optimization_level);
}

void PrefetchKernel::ConfigureCore() {
    // Only H-type prefetchers need L1 configuration
    if (static_config_.is_h_variant.value()) {
        // Initialize the FetchQ
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());
        auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
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
        uint32_t completion_q0_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
        uint32_t completion_q1_last_event_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
        std::vector<uint32_t> prefetch_q_pcie_rd_ptr_addr_data = {
            get_absolute_cq_offset(channel, cq_id_, cq_size) + cq_start};
        detail::WriteToDeviceL1(device_, logical_core_, prefetch_q_rd_ptr, prefetch_q_rd_ptr_addr_data, GetCoreType());
        detail::WriteToDeviceL1(
            device_, logical_core_, prefetch_q_pcie_rd_ptr, prefetch_q_pcie_rd_ptr_addr_data, GetCoreType());
        detail::WriteToDeviceL1(device_, logical_core_, prefetch_q_base, prefetch_q, GetCoreType());
    }
}
