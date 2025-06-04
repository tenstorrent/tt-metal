// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <device.hpp>
#include <stdint.h>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "core_coord.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch/worker_config_buffer.hpp"
#include "sub_device_types.hpp"
#include "trace_buffer.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
class LaunchMessageRingBufferState;
class SystemMemoryManager;
class WorkerConfigBufferMgr;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::trace_dispatch {

struct TraceDispatchMetadata {
    uint32_t cmd_sequence_sizeB;
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor>& trace_worker_descriptors;
    std::vector<SubDeviceId>& sub_device_ids;
    uint32_t trace_buffer_page_size = 0;
    uint32_t trace_buffer_num_pages = 0;
    uint32_t trace_buffer_address = 0;

    TraceDispatchMetadata(
        uint32_t cmd_size,
        std::unordered_map<SubDeviceId, TraceWorkerDescriptor>& descriptors,
        std::vector<SubDeviceId>& sub_devices,
        uint32_t buf_page_size,
        uint32_t buf_num_pages,
        uint32_t buf_address) :
        cmd_sequence_sizeB(cmd_size),
        trace_worker_descriptors(descriptors),
        sub_device_ids(sub_devices),
        trace_buffer_page_size(buf_page_size),
        trace_buffer_num_pages(buf_num_pages),
        trace_buffer_address(buf_address) {}
};

void reset_host_dispatch_state_for_trace(
    uint32_t num_sub_devices,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state,
    DispatchArray<uint32_t>& expected_num_workers_completed,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state_reset,
    DispatchArray<uint32_t>& expected_num_workers_completed_reset,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr_reset);

void load_host_dispatch_state(
    uint32_t num_sub_devices,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state,
    DispatchArray<uint32_t>& expected_num_workers_completed,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state_reset,
    DispatchArray<uint32_t>& expected_num_workers_completed_reset,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr_reset);

void issue_trace_commands(
    IDevice* device,
    SystemMemoryManager& sysmem_manager,
    const TraceDispatchMetadata& dispatch_md,
    uint8_t cq_id,
    const DispatchArray<uint32_t>& expected_num_workers_completed,
    CoreCoord dispatch_core);

uint32_t compute_trace_cmd_size(uint32_t num_sub_devices);

void update_worker_state_post_trace_execution(
    const std::unordered_map<SubDeviceId, TraceWorkerDescriptor>& trace_worker_descriptors,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr,
    DispatchArray<uint32_t>& expected_num_workers_completed);

std::size_t compute_interleaved_trace_buf_page_size(uint32_t buf_size, const uint32_t num_banks);

}  // namespace tt::tt_metal::trace_dispatch
