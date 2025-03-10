// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dev_msgs.h>

#include "tt_metal/impl/sub_device/dispatch.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"

namespace tt::tt_metal {
namespace subdevice_dispatch {

void set_num_worker_sems_on_dispatch(
    IDevice* device, SystemMemoryManager& manager, uint8_t cq_id, uint32_t num_worker_sems) {
    // Not needed for regular dispatch kernel
    if (!DispatchQueryManager::instance().dispatch_s_enabled()) {
        return;
    }
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);
    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    command_sequence.add_dispatch_set_num_worker_sems(num_worker_sems, DispatcherSelect::DISPATCH_SLAVE);
    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);
    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void set_go_signal_noc_data_on_dispatch(
    IDevice* device,
    const vector_memcpy_aligned<uint32_t>& go_signal_noc_data,
    SystemMemoryManager& manager,
    uint8_t cq_id) {
    uint32_t pci_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(
        sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + go_signal_noc_data.size() * sizeof(uint32_t), pci_alignment);
    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    DispatcherSelect dispatcher_for_go_signal = DispatchQueryManager::instance().dispatch_s_enabled()
                                                    ? DispatcherSelect::DISPATCH_SLAVE
                                                    : DispatcherSelect::DISPATCH_MASTER;
    command_sequence.add_dispatch_set_go_signal_noc_data(go_signal_noc_data, dispatcher_for_go_signal);
    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);
    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void reset_worker_dispatch_state_on_device(
    IDevice* device,
    SystemMemoryManager& manager,
    uint8_t cq_id,
    CoreCoord dispatch_core,
    const std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
    bool reset_launch_msg_state) {
    auto num_sub_devices = device->num_sub_devices();
    uint32_t go_signals_cmd_size = 0;
    if (reset_launch_msg_state) {
        uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        go_signals_cmd_size = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * num_sub_devices;
    }
    uint32_t cmd_sequence_sizeB =
        reset_launch_msg_state * DispatchQueryManager::instance().dispatch_s_enabled() *
            hal.get_alignment(
                HalMemType::HOST) +  // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +        // go signal cmd
        (hal.get_alignment(
             HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
                                  // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s.
                                  // dispatch_s is responsible for resetting worker count and giving dispatch_d the
                                  // latest worker state. This is encapsulated in the dispatch_s wait command (only to
                                  // be sent when dispatch is distributed on 2 cores)
         DispatchQueryManager::instance().distributed_dispatcher() * hal.get_alignment(HalMemType::HOST)) *
            num_sub_devices;
    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    bool clear_count = true;
    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    const auto& dispatch_core_config = DispatchQueryManager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();
    uint32_t dispatch_message_base_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    if (reset_launch_msg_state) {
        if (DispatchQueryManager::instance().dispatch_s_enabled()) {
            uint16_t index_bitmask = 0;
            for (uint32_t i = 0; i < num_sub_devices; ++i) {
                index_bitmask |= 1 << i;
            }
            command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
            dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
        }
        go_msg_t reset_launch_message_read_ptr_go_signal;
        reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
        reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)dispatch_core.x;
        reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)dispatch_core.y;
        for (uint32_t i = 0; i < num_sub_devices; ++i) {
            reset_launch_message_read_ptr_go_signal.dispatch_message_offset =
                (uint8_t)DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(i);
            uint32_t dispatch_message_addr =
                dispatch_message_base_addr + DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(i);
            // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
            SubDeviceId sub_device_id(static_cast<uint8_t>(i));
            command_sequence.add_dispatch_go_signal_mcast(
                expected_num_workers_completed[i],
                *reinterpret_cast<uint32_t*>(&reset_launch_message_read_ptr_go_signal),
                dispatch_message_addr,
                device->num_noc_mcast_txns(sub_device_id),
                device->num_noc_unicast_txns(sub_device_id),
                device->noc_data_start_index(sub_device_id),
                dispatcher_for_go_signal);
        }
    }
    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed
    // this step, before sending kernel config data to workers or notifying dispatch_s that its safe to send the
    // go_signal. Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr + DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(i);
        SubDeviceId sub_device_id(static_cast<uint8_t>(i));
        uint32_t expected_num_workers = expected_num_workers_completed[i] +
                                        device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id) +
                                        device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        if (DispatchQueryManager::instance().distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, expected_num_workers, clear_count, false, true, 1);
        }
        command_sequence.add_dispatch_wait(false, dispatch_message_addr, expected_num_workers, clear_count);
    }
    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);
    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

}  // namespace subdevice_dispatch

}  // namespace tt::tt_metal
