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
struct ReadL1DataDescriptor {
    void* dst = nullptr;
    uint32_t size_bytes = 0;
};

uint32_t calculate_max_prefetch_data_size_bytes(const CoreType& dispatch_core_type);

namespace device_dispatch {

struct L1ReadDispatchParams {
    const CoreCoord virtual_core;
    DeviceAddr address = 0;
    uint32_t size_bytes = 0;
    IDevice* device = nullptr;
    uint32_t cq_id = 0;
    CoreType dispatch_core_type;
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    tt::stl::Span<const SubDeviceId> sub_device_ids;
};

struct L1WriteDispatchParams : public L1ReadDispatchParams {
    const void* src = nullptr;
};

void issue_l1_write_command_sequence(const L1WriteDispatchParams& dispatch_params);

void issue_l1_read_command_sequence(const L1ReadDispatchParams& dispatch_params);

void read_l1_data_from_completion_queue(
    const ReadL1DataDescriptor& read_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager,
    std::atomic<bool>& exit_condition);

}  // namespace device_dispatch

}  // namespace tt_metal
}  // namespace tt
