// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dev_msgs.h"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "dispatch_core_common.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "tt_align.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"

enum class CoreType;

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
    bool send_unicasts) {
    uint32_t pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) +
                                  MetalContext::instance().hal().get_alignment(HalMemType::HOST);

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    auto dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    auto sub_device_index = *sub_device_id;

    HugepageDeviceCommand go_signal_cmd_sequence(cmd_region, cmd_sequence_sizeB);
    go_msg_t run_program_go_signal;
    run_program_go_signal.signal = RUN_MSG_GO;
    run_program_go_signal.master_x = dispatch_core.x;
    run_program_go_signal.master_y = dispatch_core.y;
    run_program_go_signal.dispatch_message_offset =
        MetalContext::instance().dispatch_mem_map().get_dispatch_message_update_offset(sub_device_index);

    // When running with dispatch_s enabled:
    //   - dispatch_d must notify dispatch_s that a go signal can be sent
    //   - dispatch_s then mcasts the go signal to all workers.
    // When running without dispatch_s:
    //   - dispatch_d handles sending the go signal to all workers
    // There is no need for dispatch_d to barrier before sending the dispatch_s notification or go signal,
    // since this go signal is not preceeded by NOC txns for program config data
    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        uint16_t index_bitmask = 1 << sub_device_index;
        go_signal_cmd_sequence.add_notify_dispatch_s_go_signal_cmd(
            0,                                   /* wait */
            index_bitmask /* index_bitmask */);  // When running on sub devices, we must account for this
        dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SUBORDINATE;
    }
    go_signal_cmd_sequence.add_dispatch_go_signal_mcast(
        expected_num_workers_completed,
        *reinterpret_cast<uint32_t*>(&run_program_go_signal),
        MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(sub_device_index),
        send_mcast ? device->num_noc_mcast_txns(sub_device_id) : 0,
        send_unicasts ? device->num_virtual_eth_cores(sub_device_id) : 0,
        device->noc_data_start_index(sub_device_id, send_mcast, send_unicasts), /* noc_data_start_idx */
        dispatcher_for_go_signal);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);

    sysmem_manager.fetch_queue_reserve_back(cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}
}  // namespace tt::tt_metal::distributed
