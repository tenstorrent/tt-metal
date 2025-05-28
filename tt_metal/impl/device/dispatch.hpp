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
struct ReadCoreDataDescriptor {
    void* dst = nullptr;
    uint32_t size_bytes = 0;
};

uint32_t calculate_max_prefetch_data_size_bytes(const CoreType& dispatch_core_type);

namespace device_dispatch {

struct CoreDispatchParams {
    CoreCoord virtual_core;
    DeviceAddr address = 0;
    uint32_t size_bytes = 0;
    IDevice* device = nullptr;
    uint32_t cq_id = 0;
    CoreType dispatch_core_type;
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    tt::stl::Span<const SubDeviceId> sub_device_ids;
};

struct CoreReadDispatchParams : public CoreDispatchParams {};

void validate_core_read_write_bounds(
    IDevice* device, const CoreCoord& virtual_core, DeviceAddr address, uint32_t size_bytes);

DeviceAddr add_bank_offset_to_address(IDevice* device, const CoreCoord& virtual_core, DeviceAddr address);

void write_to_core(
    IDevice* device,
    const CoreCoord& virtual_core,
    const void* src,
    DeviceAddr address,
    uint32_t size_bytes,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {});

void issue_core_read_command_sequence(const CoreReadDispatchParams& dispatch_params);

void read_core_data_from_completion_queue(
    const ReadCoreDataDescriptor& read_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager,
    std::atomic<bool>& exit_condition);

}  // namespace device_dispatch

}  // namespace tt_metal
}  // namespace tt
