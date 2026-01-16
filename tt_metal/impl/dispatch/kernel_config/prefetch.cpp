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

#include <tt_stl/assert.hpp>
#include "dispatch/command_queue_common.hpp"
#include "device.hpp"
#include "dispatch.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch_s.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/debug/inspector/inspector.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

using namespace tt::tt_metal;

PrefetchKernel::PrefetchKernel(
    int node_id,
    ChipId device_id,
    ChipId servicing_device_id,
    uint8_t cq_id,
    noc_selection_t noc_selection,
    bool h_variant,
    bool d_variant) :
    FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {
    auto& core_manager = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager();  // Not thread safe
    static_config_.is_h_variant = h_variant;
    static_config_.is_d_variant = d_variant;
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);

    DispatchWorkerType type = PREFETCH;
    if (h_variant && d_variant) {
        this->logical_core_ = core_manager.prefetcher_core(device_id, channel, cq_id);
        type = PREFETCH_HD;
    } else if (h_variant) {
        channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(servicing_device_id);
        this->logical_core_ = core_manager.prefetcher_core(servicing_device_id, channel, cq_id);
        type = PREFETCH_H;
    } else if (d_variant) {
        this->logical_core_ = core_manager.prefetcher_d_core(device_id, channel, cq_id);
        type = PREFETCH_D;
    }
    this->kernel_type_ = FDKernelType::DISPATCH;
    // Log prefetcher core info based on virtual core to inspector
    auto virtual_core = this->GetVirtualCore();
    tt::tt_metal::Inspector::set_prefetcher_core_info(virtual_core, type, cq_id, device_id, servicing_device_id);
}

void PrefetchKernel::GenerateStaticConfigs() {
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());
    uint8_t cq_id_ = this->cq_id_;
    const auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());
    auto l1_size = my_dispatch_constants.get_prefetcher_l1_size();
    // May be zero if not using dispatch on fabric
    static_config_.fabric_header_rb_base =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::FABRIC_HEADER_RB);
    static_config_.fabric_header_rb_entries = tt::tt_metal::DispatchSettings::FABRIC_HEADER_RB_ENTRIES;
    static_config_.my_fabric_sync_status_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS);

    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        bool is_mock =
            tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = is_mock ? 0x10000 : device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = is_mock ? 0x10000 : device_->sysmem_manager().get_issue_queue_size(cq_id_);

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
        static_config_.my_downstream_cb_sem_id = tt::tt_metal::CreateSemaphore(
            *program_, logical_core_, my_dispatch_constants.prefetch_d_buffer_pages(), GetCoreType());
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
    auto scratch_db_base = static_config_.scratch_db_base.value_or(0);
    auto ringbuffer_size = static_config_.ringbuffer_size.value_or(0);
    TT_ASSERT(
        scratch_db_base + ringbuffer_size <= l1_size,
        "Prefetcher allocations exceed L1 size: scratch_db_base: 0x{:X}, ringbuffer_size: 0x{:X} B, L1 size: 0x{:X} B",
        scratch_db_base,
        ringbuffer_size,
        l1_size);

    if (!is_hd()) {
        create_edm_connection_sems(edm_connection_attributes_);
        const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        static_config_.is_2d_fabric = fabric_context.is_2D_routing_enabled();
    } else {
        static_config_.is_2d_fabric = false;
    }
}

void PrefetchKernel::GenerateDependentConfigs() {
    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        // Upstream
        TT_ASSERT(upstream_kernels_.empty());
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
            if (auto* dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                dependent_config_.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                dependent_config_.downstream_cb_sem_id = dispatch_kernel->GetStaticConfig().my_dispatch_cb_sem_id;
                dependent_config_.downstream_cb_base = dispatch_kernel->GetStaticConfig().dispatch_cb_base;
                dependent_config_.downstream_cb_log_page_size =
                    dispatch_kernel->GetStaticConfig().dispatch_cb_log_page_size;
                dependent_config_.downstream_cb_pages = dispatch_kernel->GetStaticConfig().dispatch_cb_pages;
            } else if (auto* dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(k)) {
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
        TT_ASSERT(upstream_kernels_.empty());
        dependent_config_.upstream_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.upstream_cb_sem_id = 0;  // Used in prefetch_d only
        // May be overwritten below
        dependent_config_.num_hops = 0;

        // Process downstream
        // PREFETCH_D ||
        // FABRIC_MUX
        for (FDKernel* ds_kernel : downstream_kernels_) {
            if (auto* prefetch_d = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
                TT_ASSERT(
                    prefetch_d->GetStaticConfig().is_d_variant.value() &&
                    !prefetch_d->GetStaticConfig().is_h_variant.value());

                dependent_config_.downstream_logical_core = prefetch_d->GetLogicalCore();
                dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
                dependent_config_.downstream_cb_base = prefetch_d->GetStaticConfig().cmddat_q_base;
                dependent_config_.downstream_cb_sem_id = prefetch_d->GetStaticConfig().my_upstream_cb_sem_id;
                dependent_config_.downstream_dispatch_s_cb_sem_id = 0;

                static_assert(
                    DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE ==
                    DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
                dependent_config_.downstream_cb_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
                dependent_config_.downstream_cb_pages = prefetch_d->GetStaticConfig().cmddat_q_pages;
                dependent_config_.num_hops = tt::tt_metal::get_num_hops(device_id_, prefetch_d->GetDeviceId());
                assemble_2d_fabric_packet_header_args(
                    this->dependent_config_, GetDeviceId(), prefetch_d->GetDeviceId());
            } else if (auto* fabric_mux = dynamic_cast<tt::tt_metal::RelayMux*>(ds_kernel)) {
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
        if (auto* prefetch_h = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0])) {
            dependent_config_.upstream_logical_core = prefetch_h->GetLogicalCore();
            dependent_config_.upstream_cb_sem_id = prefetch_h->GetStaticConfig().my_downstream_cb_sem_id;
            dependent_config_.num_hops = tt::tt_metal::get_num_hops(prefetch_h->GetDeviceId(), device_id_);
            assemble_2d_fabric_packet_header_args(this->dependent_config_, GetDeviceId(), prefetch_h->GetDeviceId());
        } else {
            TT_FATAL(false, "Path not implemented");
        }

        // Downstream
        // DISPATCH_D || DISPATCH_S || FABRIC_MUX
        bool found_dispatch = false;
        bool found_dispatch_s = false;
        bool found_relay_mux = false;
        for (FDKernel* k : downstream_kernels_) {
            if (auto* dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
                TT_ASSERT(!found_dispatch, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch = true;

                dependent_config_.downstream_logical_core = dispatch_kernel->GetLogicalCore();
                dependent_config_.downstream_cb_sem_id = dispatch_kernel->GetStaticConfig().my_dispatch_cb_sem_id;
                dependent_config_.downstream_cb_base = dispatch_kernel->GetStaticConfig().dispatch_cb_base;
                dependent_config_.downstream_cb_log_page_size =
                    dispatch_kernel->GetStaticConfig().dispatch_cb_log_page_size;
                dependent_config_.downstream_cb_pages = dispatch_kernel->GetStaticConfig().dispatch_cb_pages;
            } else if (auto* dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(k)) {
                TT_ASSERT(!found_dispatch_s, "PREFETCH kernel has multiple downstream DISPATCH kernels.");
                found_dispatch_s = true;

                dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
                dependent_config_.downstream_dispatch_s_cb_sem_id =
                    dispatch_s_kernel->GetStaticConfig().my_dispatch_cb_sem_id;
            } else if (auto* relay_mux = dynamic_cast<tt::tt_metal::RelayMux*>(k)) {
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

void PrefetchKernel::InitializeRuntimeArgsValues() {
    // Initialize runtime args offsets
    int current_offset = 0;
    static_config_.offsetof_my_dev_id = current_offset++;
    static_config_.offsetof_to_dev_id = current_offset++;
    static_config_.offsetof_router_direction = current_offset++;
    // Initialize runtime args
    runtime_args_.resize(current_offset);
    runtime_args_[static_config_.offsetof_my_dev_id.value()] = dependent_config_.my_dev_id.value_or(0);
    runtime_args_[static_config_.offsetof_to_dev_id.value()] = dependent_config_.to_dev_id.value_or(0);
    runtime_args_[static_config_.offsetof_router_direction.value()] = dependent_config_.router_direction.value_or(0);
}

void PrefetchKernel::CreateKernel() {
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
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},  // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(upstream_virtual_noc_coords.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_virtual_noc_coords.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},

        // Direct configuration values
        {"DOWNSTREAM_CB_BASE", std::to_string(dependent_config_.downstream_cb_base.value())},
        {"DOWNSTREAM_CB_LOG_PAGE_SIZE", std::to_string(dependent_config_.downstream_cb_log_page_size.value())},
        {"DOWNSTREAM_CB_PAGES", std::to_string(dependent_config_.downstream_cb_pages.value())},
        {"MY_DOWNSTREAM_CB_SEM_ID", std::to_string(static_config_.my_downstream_cb_sem_id.value())},
        {"DOWNSTREAM_CB_SEM_ID", std::to_string(dependent_config_.downstream_cb_sem_id.value())},
        {"PCIE_BASE", std::to_string(static_config_.pcie_base.value())},
        {"PCIE_SIZE", std::to_string(static_config_.pcie_size.value())},
        {"PREFETCH_Q_BASE", std::to_string(static_config_.prefetch_q_base.value())},
        {"PREFETCH_Q_SIZE", std::to_string(static_config_.prefetch_q_size.value())},
        {"PREFETCH_Q_RD_PTR_ADDR", std::to_string(static_config_.prefetch_q_rd_ptr_addr.value())},
        {"PREFETCH_Q_PCIE_RD_PTR_ADDR", std::to_string(static_config_.prefetch_q_pcie_rd_ptr_addr.value())},
        {"CMDDAT_Q_BASE", std::to_string(static_config_.cmddat_q_base.value())},
        {"CMDDAT_Q_SIZE", std::to_string(static_config_.cmddat_q_size.value())},
        {"SCRATCH_DB_BASE", std::to_string(static_config_.scratch_db_base.value())},
        {"SCRATCH_DB_SIZE", std::to_string(static_config_.scratch_db_size.value())},
        {"DOWNSTREAM_SYNC_SEM_ID", std::to_string(static_config_.downstream_sync_sem_id.value())},
        {"CMDDAT_Q_PAGES", std::to_string(static_config_.cmddat_q_pages.value())},
        {"MY_UPSTREAM_CB_SEM_ID", std::to_string(static_config_.my_upstream_cb_sem_id.value())},
        {"UPSTREAM_CB_SEM_ID", std::to_string(dependent_config_.upstream_cb_sem_id.value())},
        {"CMDDAT_Q_LOG_PAGE_SIZE", std::to_string(static_config_.cmddat_q_log_page_size.value())},
        {"CMDDAT_Q_BLOCKS", std::to_string(static_config_.cmddat_q_blocks.value())},
        {"DISPATCH_S_BUFFER_BASE", std::to_string(static_config_.dispatch_s_buffer_base.value())},
        {"MY_DISPATCH_S_CB_SEM_ID", std::to_string(static_config_.my_dispatch_s_cb_sem_id.value())},
        {"DOWNSTREAM_DISPATCH_S_CB_SEM_ID", std::to_string(dependent_config_.downstream_dispatch_s_cb_sem_id.value())},
        {"DISPATCH_S_BUFFER_SIZE", std::to_string(static_config_.dispatch_s_buffer_size.value())},
        {"DISPATCH_S_CB_LOG_PAGE_SIZE", std::to_string(static_config_.dispatch_s_cb_log_page_size.value())},
        {"RINGBUFFER_SIZE", std::to_string(static_config_.ringbuffer_size.value())},
        // Fabric configuration
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

        {"EW_DIM", std::to_string(dependent_config_.ew_dim.value_or(0))},
        {"TO_MESH_ID", std::to_string(dependent_config_.to_mesh_id.value_or(0))},
        {"IS_D_VARIANT", std::to_string(static_config_.is_d_variant.value())},
        {"IS_H_VARIANT", std::to_string(static_config_.is_h_variant.value())},
    };

    if (!is_hd()) {
        defines["FABRIC_RELAY"] = "1";
        if (static_config_.is_2d_fabric.value_or(false)) {
            defines["FABRIC_2D"] = "1";
        }
    }

    // Runtime args offsets
    defines["OFFSETOF_MY_DEV_ID"] = std::to_string(static_config_.offsetof_my_dev_id.value_or(0));
    defines["OFFSETOF_TO_DEV_ID"] = std::to_string(static_config_.offsetof_to_dev_id.value_or(0));
    defines["OFFSETOF_ROUTER_DIRECTION"] = std::to_string(static_config_.offsetof_router_direction.value_or(0));

    // Compile at Os on IERISC to fit in code region.
    auto optimization_level = (GetCoreType() == CoreType::WORKER) ? KernelBuildOptLevel::O2 : KernelBuildOptLevel::Os;
    configure_kernel_variant(
        dispatch_kernel_file_names[PREFETCH],
        {},
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
        const auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());
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
        std::vector<uint32_t> prefetch_q_pcie_rd_ptr_addr_data = {
            get_absolute_cq_offset(channel, cq_id_, cq_size) + cq_start};
        detail::WriteToDeviceL1(device_, logical_core_, prefetch_q_rd_ptr, prefetch_q_rd_ptr_addr_data, GetCoreType());
        detail::WriteToDeviceL1(
            device_, logical_core_, prefetch_q_pcie_rd_ptr, prefetch_q_pcie_rd_ptr_addr_data, GetCoreType());
        detail::WriteToDeviceL1(device_, logical_core_, prefetch_q_base, prefetch_q, GetCoreType());
    }
}
