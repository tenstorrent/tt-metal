// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <command_queue.hpp>
#include "device.hpp"
#include "dispatch/topology.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"

namespace tt {
namespace tt_metal {

// Used so the host knows how to properly copy data into user space from the completion queue (in hugepages)
struct ReadProfilerControlVectorDescriptor {
    void* dst;
};

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

void issue_read_profiler_control_vector_command_sequence(const ProfilerDispatchParams& dispatch_params);

void read_profiler_control_vector_from_completion_queue(
    ReadProfilerControlVectorDescriptor& read_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager);

}  // namespace profiler_dispatch

}  // namespace tt_metal
}  // namespace tt
