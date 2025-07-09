// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_s.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "assert.hpp"
#include "dispatch/command_queue_common.hpp"
#include "device.hpp"
#include "dispatch.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal_types.hpp"
#include "prefetch.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/xy_pair.h>

#include "tt_metal/api/tt-metalium/device_pool.hpp"

using namespace tt::tt_metal;

void DispatchSKernel::GenerateStaticConfigs() {
    auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());

    uint32_t dispatch_s_buffer_base = 0xff;
    if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        uint32_t dispatch_buffer_base = my_dispatch_constants.dispatch_buffer_base();
        if (GetCoreType() == CoreType::WORKER) {
            // dispatch_s is on the same Tensix core as dispatch_d. Shared resources. Offset CB start idx.
            dispatch_s_buffer_base = dispatch_buffer_base + (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                                                my_dispatch_constants.dispatch_buffer_pages();
        } else {
            // dispatch_d and dispatch_s are on different cores. No shared resources: dispatch_s CB starts at base.
            dispatch_s_buffer_base = dispatch_buffer_base;
        }
    }

    static_config_.cb_base = dispatch_s_buffer_base;
    static_config_.cb_log_page_size = DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE;
    static_config_.cb_size = my_dispatch_constants.dispatch_s_buffer_size();
    // used by dispatch_s to sync with prefetch
    static_config_.my_dispatch_cb_sem_id = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
    static_config_.dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
    // used by dispatch_d to signal that dispatch_s can send go signal

    static_config_.mcast_go_signal_addr =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    static_config_.unicast_go_signal_addr =
        (MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
            ? MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
            : 0;
    static_config_.distributed_dispatcher =
        MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher();
    static_config_.first_stream_used = my_dispatch_constants.get_dispatch_stream_index(0);
    static_config_.max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
    static_config_.max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
}

void DispatchSKernel::GenerateDependentConfigs() {
    // Upstream
    TT_ASSERT(upstream_kernels_.size() == 1);
    auto prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0]);
    TT_ASSERT(prefetch_kernel);
    dependent_config_.upstream_logical_core = prefetch_kernel->GetLogicalCore();
    dependent_config_.upstream_dispatch_cb_sem_id = prefetch_kernel->GetStaticConfig().my_dispatch_s_cb_sem_id;

    // Downstream
    TT_ASSERT(downstream_kernels_.size() == 1);
    auto dispatch_kernel = dynamic_cast<DispatchKernel*>(downstream_kernels_[0]);
    TT_ASSERT(dispatch_kernel);
    dependent_config_.downstream_logical_core = dispatch_kernel->GetLogicalCore();
}

void DispatchSKernel::CreateKernel() {
    // Issue #19729: Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
    // num_virtual_active_eth_cores is set if the user application requested virtualizing the
    // number of ethernet cores across devices (to essentially fake uniformity). This value is the
    // max number of ethernet cores across all chips in the opened cluster.
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

    auto my_virtual_core = device_->virtual_core_from_logical_core(logical_core_, GetCoreType());
    auto upstream_virtual_core =
        device_->virtual_core_from_logical_core(dependent_config_.upstream_logical_core.value(), GetCoreType());
    auto downstream_virtual_core =
        device_->virtual_core_from_logical_core(dependent_config_.downstream_logical_core.value(), GetCoreType());
    auto downstream_s_virtual_core = device_->virtual_core_from_logical_core(UNUSED_LOGICAL_CORE, GetCoreType());

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
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},  // Unused, remove later
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},  // Unused, remove later
        {"CB_BASE", std::to_string(static_config_.cb_base.value())},
        {"CB_LOG_PAGE_SIZE", std::to_string(static_config_.cb_log_page_size.value())},
        {"CB_SIZE", std::to_string(static_config_.cb_size.value())},
        {"MY_DISPATCH_CB_SEM_ID", std::to_string(static_config_.my_dispatch_cb_sem_id.value())},
        {"UPSTREAM_DISPATCH_CB_SEM_ID", std::to_string(dependent_config_.upstream_dispatch_cb_sem_id.value())},
        {"DISPATCH_S_SYNC_SEM_BASE_ADDR", std::to_string(static_config_.dispatch_s_sync_sem_base_addr.value())},
        {"MCAST_GO_SIGNAL_ADDR", std::to_string(static_config_.mcast_go_signal_addr.value())},
        {"UNICAST_GO_SIGNAL_ADDR", std::to_string(static_config_.unicast_go_signal_addr.value())},
        {"DISTRIBUTED_DISPATCHER", std::to_string(static_config_.distributed_dispatcher.value())},
        {"FIRST_STREAM_USED", std::to_string(static_config_.first_stream_used.value())},
        {"MAX_NUM_WORKER_SEMS", std::to_string(static_config_.max_num_worker_sems.value())},
        {"MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES",
         std::to_string(static_config_.max_num_go_signal_noc_data_entries.value())},
        {"VIRTUALIZE_UNICAST_CORES", std::to_string(virtualize_num_eth_cores)},
        {"NUM_VIRTUAL_UNICAST_CORES", std::to_string(num_virtual_active_eth_cores)},
        {"NUM_PHYSICAL_UNICAST_CORES", std::to_string(num_physical_active_eth_cores)},
    };
    configure_kernel_variant(dispatch_kernel_file_names[DISPATCH_S], {}, defines, false, false, false);
}

void DispatchSKernel::ConfigureCore() {
    if (!MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher()) {
        return;
    }
    // Just need to clear the dispatch message
    std::vector<uint32_t> zero = {0x0};
    auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());
    uint32_t dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
    for (uint32_t i = 0; i < DispatchSettings::DISPATCH_MESSAGE_ENTRIES; i++) {
        uint32_t dispatch_s_sync_sem_addr = dispatch_s_sync_sem_base_addr + my_dispatch_constants.get_sync_offset(i);
        detail::WriteToDeviceL1(device_, logical_core_, dispatch_s_sync_sem_addr, zero, GetCoreType());
    }
}
