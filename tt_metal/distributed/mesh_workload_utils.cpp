// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <command_queue.hpp>

#include "tt_metal/impl/program/dispatch.hpp"

namespace tt::tt_metal::distributed {

namespace experimental {

void write_program_commands(
    CommandQueue& cq,
    ProgramCommandSequence& program_cmd_seq,
    uint32_t num_active_cores_in_program,
    SubDeviceId sub_device_id,
    bool stall_first,
    bool stall_before_program,
    bool blocking) {
    auto sub_device_index = sub_device_id.to_index();
    // Increment expected num workers inside single device CQs to ensure other paths dont break.
    // This is temporary, since data movement and events rely on single device CQs. Once MeshCommandQueue
    // supports all runtime features, this will be removed, and program dispatch commands will be written
    // directly through dedicated interfaces.

    uint32_t num_workers_in_cq = cq.get_expected_num_workers_completed_for_sub_device(sub_device_index);
    cq.set_expected_num_workers_completed_for_sub_device(
        sub_device_index, num_workers_in_cq + num_active_cores_in_program);
    // Write program command stream to device
    program_dispatch::write_program_command_sequence(
        program_cmd_seq,
        cq.device()->sysmem_manager(),
        cq.id(),
        dispatch_core_manager::instance().get_dispatch_core_type(cq.device()->id()),
        stall_first,
        stall_before_program);
}

// Use this function to send go signals to a device not running a program.
// In the MeshWorkload context, a go signal must be sent to each device when
// a workload is dispatched, in order to maintain consistent global state.
void write_go_signal(
    CommandQueue& cq,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    int num_unicast_txns = -1) {
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB =
        align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) + hal.get_alignment(HalMemType::HOST);

    auto& manager = cq.device()->sysmem_manager();
    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq.id());

    HugepageDeviceCommand go_signal_cmd_sequence(cmd_region, cmd_sequence_sizeB);
    go_msg_t run_program_go_signal;

    run_program_go_signal.signal = RUN_MSG_GO;
    run_program_go_signal.master_x = dispatch_core.x;
    run_program_go_signal.master_y = dispatch_core.y;
    run_program_go_signal.dispatch_message_offset = 0;

    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(cq.device()->id());
    uint32_t dispatch_message_addr = DispatchMemMap::get(dispatch_core_type)
                                         .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);

    go_signal_cmd_sequence.add_notify_dispatch_s_go_signal_cmd(
        0, /* wait */
        1 /* index_bitmask */);

    go_signal_cmd_sequence.add_dispatch_go_signal_mcast(
        expected_num_workers_completed,
        *reinterpret_cast<uint32_t*>(&run_program_go_signal),
        dispatch_message_addr,
        send_mcast ? cq.device()->num_noc_mcast_txns(SubDeviceId{0}) : 0,
        send_unicasts ? ((num_unicast_txns > 0) ? num_unicast_txns : cq.device()->num_noc_unicast_txns(SubDeviceId{0}))
                      : 0,
        0, /* noc_data_start_idx */
        DispatcherSelect::DISPATCH_SLAVE);

    manager.issue_queue_push_back(cmd_sequence_sizeB, cq.id());

    manager.fetch_queue_reserve_back(cq.id());
    manager.fetch_queue_write(cmd_sequence_sizeB, cq.id());
}

}  // namespace experimental

}  // namespace tt::tt_metal::distributed
