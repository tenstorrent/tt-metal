// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <command_queue_interface.hpp>
#include <stdint.h>
#include <sub_device_types.hpp>
#include <tt-metalium/device.hpp>

#include <tt_stl/span.hpp>
#include "dispatch/system_memory_manager.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

// Used so host knows data in completion queue is just an event ID
struct ReadEventDescriptor {
    uint32_t event_id;
    uint32_t global_offset;

    explicit ReadEventDescriptor(uint32_t event) : event_id(event), global_offset(0) {}

    void set_global_offset(uint32_t offset) { global_offset = offset; }
    uint32_t get_global_event_id() { return global_offset + event_id; }
};

namespace event_dispatch {

void issue_record_event_commands(
    IDevice* device,
    uint32_t event_id,
    uint8_t cq_id,
    uint32_t num_command_queues,
    SystemMemoryManager& manager,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    bool notify_host = true,
    bool clear_count = false);

void issue_wait_for_event_commands(
    uint8_t cq_id, uint8_t event_cq_id, SystemMemoryManager& sysmem_manager, uint32_t event_id);

void read_events_from_completion_queue(
    ReadEventDescriptor& event_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager);

}  // namespace event_dispatch

}  // namespace tt::tt_metal
