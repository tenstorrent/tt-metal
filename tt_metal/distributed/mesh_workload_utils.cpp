// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <cstdint>
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "tt_align.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"

#include <umd/device/types/core_coordinates.hpp>

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
    const program_dispatch::ProgramDispatchMetadata& dispatch_md) {
    const auto& hal = MetalContext::instance().hal();
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    DeviceCommandCalculator calculator;
    if (tt_metal::MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        calculator.add_notify_dispatch_s_go_signal_cmd();
    }
    calculator.add_dispatch_go_signal_mcast();
    uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();
    cmd_sequence_sizeB +=
        dispatch_md.prefetcher_cache_info.is_cached ? 0 : align(sizeof(CQPrefetchCmd), pcie_alignment);

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    auto sub_device_index = *sub_device_id;

    HugepageDeviceCommand go_signal_cmd_sequence(cmd_region, cmd_sequence_sizeB);

    if (not dispatch_md.prefetcher_cache_info.is_cached) {
        go_signal_cmd_sequence.add_prefetch_set_ringbuffer_offset(
            dispatch_md.prefetcher_cache_info.offset + dispatch_md.prefetcher_cache_info.mesh_max_program_kernels_sizeB,
            true);
    }

    uint32_t go_msg_u32_val = hal.make_go_msg_u32(
        dev_msgs::RUN_MSG_GO,
        dispatch_core.x,
        dispatch_core.y,
        MetalContext::instance().dispatch_mem_map().get_dispatch_message_update_offset(sub_device_index));

    // When running with dispatch_s enabled:
    //   - dispatch_d must notify dispatch_s that a go signal can be sent
    //   - dispatch_s then mcasts the go signal to all workers.
    // When running without dispatch_s:
    //   - dispatch_d handles sending the go signal to all workers
    // There is no need for dispatch_d to barrier before sending the dispatch_s notification or go signal,
    // since this go signal is not preceded by NOC txns for program config data
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
        go_msg_u32_val,
        MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(sub_device_index),
        (send_mcast && device->has_noc_mcast_txns(sub_device_id)) ? *sub_device_id
                                                                  : CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET,
        send_unicasts ? device->num_virtual_eth_cores(sub_device_id) : 0,
        device->noc_data_start_index(sub_device_id, send_unicasts), /* noc_data_start_idx */
        dispatcher_for_go_signal);

    TT_ASSERT(go_signal_cmd_sequence.size_bytes() == go_signal_cmd_sequence.write_offset_bytes());

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);

    sysmem_manager.fetch_queue_reserve_back(cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}
}  // namespace tt::tt_metal::distributed
