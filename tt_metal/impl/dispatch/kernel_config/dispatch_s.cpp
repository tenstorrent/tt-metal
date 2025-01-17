// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_s.hpp"
#include "dispatch.hpp"
#include "prefetch.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>

using namespace tt::tt_metal;

void DispatchSKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_->id());
    uint8_t cq_id_ = this->cq_id_;
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());

    uint32_t dispatch_s_buffer_base = 0xff;
    if (device_->dispatch_s_enabled()) {
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

    static_config_.cb_base = dispatch_s_buffer_base;
    static_config_.cb_log_page_size = dispatch_constants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE;
    static_config_.cb_size = my_dispatch_constants.dispatch_s_buffer_size();
    // used by dispatch_s to sync with prefetch
    static_config_.my_dispatch_cb_sem_id = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
    static_config_.dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM);
    // used by dispatch_d to signal that dispatch_s can send go signal

    static_config_.mcast_go_signal_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    static_config_.unicast_go_signal_addr =
        (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
            ? hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
            : 0;
    static_config_.distributed_dispatcher = (GetCoreType() == CoreType::ETH);
    static_config_.worker_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    static_config_.max_num_worker_sems = dispatch_constants::DISPATCH_MESSAGE_ENTRIES;
    static_config_.max_num_go_signal_noc_data_entries = dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
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
    std::vector<uint32_t> compile_args = {
        static_config_.cb_base.value(),
        static_config_.cb_log_page_size.value(),
        static_config_.cb_size.value(),
        static_config_.my_dispatch_cb_sem_id.value(),
        dependent_config_.upstream_dispatch_cb_sem_id.value(),
        static_config_.dispatch_s_sync_sem_base_addr.value(),
        static_config_.mcast_go_signal_addr.value(),
        static_config_.unicast_go_signal_addr.value(),
        static_config_.distributed_dispatcher.value(),
        static_config_.worker_sem_base_addr.value(),
        static_config_.max_num_worker_sems.value(),
        static_config_.max_num_go_signal_noc_data_entries.value(),
    };
    TT_ASSERT(compile_args.size() == 12);
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

    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(my_virtual_noc_coords.x)},
        {"MY_NOC_Y", std::to_string(my_virtual_noc_coords.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},  // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(upstream_virtual_noc_coords.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_virtual_noc_coords.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_virtual_noc_coords.y)},
        {"DOWNSTREAM_SLAVE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},  // Unused, remove later
        {"DOWNSTREAM_SLAVE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},  // Unused, remove later
    };
    configure_kernel_variant(dispatch_kernel_file_names[DISPATCH_S], compile_args, defines, false, true, false);
}

void DispatchSKernel::ConfigureCore() {
    if (!device_->distributed_dispatcher()) {
        return;
    }
    // Just need to clear the dispatch message
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
}
