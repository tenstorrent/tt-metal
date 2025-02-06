// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <command_queue.hpp>

#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"

namespace tt::tt_metal::distributed {

// Use this function to send go signals to a device not running a program.
// In the MeshWorkload context, a go signal must be sent to each device when
// a workload is dispatched, in order to maintain consistent global state.
void write_go_signal(
    uint8_t cq_id,
    IDevice* device,
    SubDeviceId sub_device_id,
    SystemMemoryManager& sysmem_manager,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    int num_unicast_txns) {
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB =
        align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) + hal.get_alignment(HalMemType::HOST);

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    auto dispatch_core_config = DispatchQueryManager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();
    auto sub_device_index = sub_device_id.to_index();

    HugepageDeviceCommand go_signal_cmd_sequence(cmd_region, cmd_sequence_sizeB);
    go_msg_t run_program_go_signal;
    run_program_go_signal.signal = RUN_MSG_GO;
    run_program_go_signal.master_x = dispatch_core.x;
    run_program_go_signal.master_y = dispatch_core.y;
    run_program_go_signal.dispatch_message_offset =
        (uint8_t)DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(sub_device_index);

    uint32_t dispatch_message_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE) +
        DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(sub_device_index);

    // When running with dispatch_s enabled:
    //   - dispatch_d must notify dispatch_s that a go signal can be sent
    //   - dispatch_s then mcasts the go signal to all workers.
    // When running without dispatch_s:
    //   - dispatch_d handles sending the go signal to all workers
    // There is no need for dispatch_d to barrier before sending the dispatch_s notification or go signal,
    // since this go signal is not preceeded by NOC txns for program config data
    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    if (DispatchQueryManager::instance().dispatch_s_enabled()) {
        uint16_t index_bitmask = 1 << sub_device_index;
        go_signal_cmd_sequence.add_notify_dispatch_s_go_signal_cmd(
            0,                                   /* wait */
            index_bitmask /* index_bitmask */);  // When running on sub devices, we must account for this
        dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
    }
    go_signal_cmd_sequence.add_dispatch_go_signal_mcast(
        expected_num_workers_completed,
        *reinterpret_cast<uint32_t*>(&run_program_go_signal),
        dispatch_message_addr,
        send_mcast ? device->num_noc_mcast_txns(sub_device_id) : 0,
        send_unicasts ? ((num_unicast_txns > 0) ? num_unicast_txns : device->num_noc_unicast_txns(sub_device_id)) : 0,
        device->noc_data_start_index(sub_device_id, send_mcast, send_unicasts), /* noc_data_start_idx */
        dispatcher_for_go_signal);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);

    sysmem_manager.fetch_queue_reserve_back(cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

bool is_row_major_intersection(const LogicalDeviceRange& parent, const LogicalDeviceRange& intersection) {
    return intersection.grid_size().x == parent.grid_size().x;
}

LogicalDeviceRange convex_relative_complement(
    const LogicalDeviceRange& parent, const LogicalDeviceRange& intersection) {
    TT_FATAL(parent.contains(intersection), "Parent must contain intersection");
    auto intersection_grid_size = intersection.grid_size();
    auto parent_grid_size = parent.grid_size();
    TT_FATAL(
        intersection_grid_size.x == parent_grid_size.x || intersection_grid_size.y == parent_grid_size.y,
        "Non convex grids not supported");

    if (is_row_major_intersection(parent, intersection)) {
        if (intersection.start_coord.y == parent.start_coord.y) {
            return LogicalDeviceRange(
                {parent.start_coord.x, intersection.end_coord.y + 1}, {parent.end_coord.x, parent.end_coord.y});
        } else {
            return LogicalDeviceRange(
                {parent.start_coord.x, parent.start_coord.y}, {parent.end_coord.x, intersection.start_coord.y - 1});
        }
    } else {
        if (intersection.start_coord.x == parent.start_coord.x) {
            return LogicalDeviceRange(
                {intersection.end_coord.x + 1, parent.start_coord.y}, {parent.end_coord.x, parent.end_coord.y});
        } else {
            return LogicalDeviceRange(
                {parent.start_coord.x, parent.start_coord.y}, {intersection.start_coord.x - 1, parent.end_coord.y});
        }
    }
}

}  // namespace tt::tt_metal::distributed
