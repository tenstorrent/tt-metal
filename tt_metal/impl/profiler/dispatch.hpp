// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.hpp"
#include "dispatch/device_command.hpp"
#include "dispatch/topology.hpp"
#include "hal_types.hpp"
#include "hostdevcommon/profiler_common.h"
#include "llrt/hal.hpp"

namespace tt {
namespace tt_metal {

namespace profiler_dispatch {

struct ProfilerDispatchParams {
    const CoreCoord virtual_core;
    DeviceAddr address = 0;
    IDevice* device = nullptr;
    uint32_t cq_id = 0;
    CoreType dispatch_core_type;
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    tt::stl::Span<const SubDeviceId> sub_device_ids;
};

// Used so the host knows how to properly copy data into user space from the completion queue (in hugepages)
struct ReadProfilerControlVectorDescriptor {
    void* dst;
    uint32_t dst_offset;
    uint32_t num_pages_read;
    uint32_t cur_dev_page_id;
    uint32_t starting_host_page_id;
};

void issue_read_profiler_control_vector_command_sequence(const ProfilerDispatchParams& dispatch_params) {
    // accounts for padding
    const uint32_t pcie_alignment = hal_ref.get_alignment(tt::tt_metal::HalMemType::HOST);
    const uint32_t num_worker_counters = dispatch_params.sub_device_ids.size();
    const uint32_t cmd_sequence_sizeB =
        pcie_alignment * num_worker_counters +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        pcie_alignment +                        // CQ_PREFETCH_CMD_STALL
        pcie_alignment +  // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST
        pcie_alignment;   // CQ_PREFETCH_CMD_RELAY_LINEAR or CQ_PREFETCH_CMD_RELAY_PAGED

    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    // We only need the write barrier + prefetch stall for the last wait cmd
    const uint32_t last_index = num_worker_counters - 1;
    for (uint32_t i = 0; i < last_index; ++i) {
        const uint8_t offset_index = *dispatch_params.sub_device_ids[i];
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
            0,
            DispatchMemMap::get(dispatch_params.dispatch_core_type, dispatch_params.device->num_hw_cqs())
                .get_dispatch_stream_index(offset_index),
            dispatch_params.expected_num_workers_completed[offset_index]);
    }
    const uint8_t offset_index = *dispatch_params.sub_device_ids[last_index];
    command_sequence.add_dispatch_wait_with_prefetch_stall(
        CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER,
        0,
        DispatchMemMap::get(dispatch_params.dispatch_core_type, dispatch_params.device->num_hw_cqs())
            .get_dispatch_stream_index(offset_index),
        dispatch_params.expected_num_workers_completed[offset_index]);

    command_sequence.add_dispatch_write_host(false, kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, false);

    command_sequence.add_prefetch_relay_linear(
        dispatch_params.device->get_noc_unicast_encoding(k_dispatch_downstream_noc, dispatch_params.virtual_core),
        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE,
        dispatch_params.address);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

}  // namespace profiler_dispatch

}  // namespace tt_metal
}  // namespace tt
