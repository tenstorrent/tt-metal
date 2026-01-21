// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/event/dispatch.hpp"

#include <tt_stl/span.hpp>
#include <tt_align.hpp>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "dispatch/command_queue_common.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "hal_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/strong_type.hpp>
#include "sub_device_types.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal::event_dispatch {

namespace {
uint32_t get_packed_write_max_unicast_sub_cmds(IDevice* device) {
    return device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
}
}  // namespace

void issue_record_event_commands(
    IDevice* device,
    ChipId device_id,
    uint32_t event_id,
    uint8_t cq_id,
    uint32_t num_command_queues,
    SystemMemoryManager& manager,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    bool notify_host,
    bool clear_count) {
    std::vector<uint32_t> event_payload(DispatchSettings::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = event_id;

    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t num_worker_counters = sub_device_ids.size();
    const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(device);

    // Calculate the packed event payload size
    uint32_t packed_event_payload_sizeB;
    {
        tt::tt_metal::DeviceCommandCalculator event_payload_calculator;
        event_payload_calculator.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
            num_command_queues, DispatchSettings::EVENT_PADDED_SIZE, packed_write_max_unicast_sub_cmds, false);
        packed_event_payload_sizeB =
            tt::align(event_payload_calculator.write_offset_bytes() - sizeof(CQPrefetchCmd), l1_alignment);
    }

    // Calculate the actual command size
    tt::tt_metal::DeviceCommandCalculator calculator;
    for (int i = 0; i < num_worker_counters; ++i) {
        calculator.add_dispatch_wait();
    }

    calculator.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_command_queues, DispatchSettings::EVENT_PADDED_SIZE, packed_write_max_unicast_sub_cmds, false);

    if (notify_host) {
        calculator.add_dispatch_write_linear_host_event(DispatchSettings::EVENT_PADDED_SIZE);
    }
    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();

    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    auto dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);

    for (uint32_t i = 0; i < num_worker_counters; ++i) {
        auto offset_index = *sub_device_ids[i];
        // recording an event does not have any side-effects on the dispatch completion count
        // hence clear_count is set to false, i.e. the number of workers on the dispatcher is
        // not reset
        // We only need the write barrier for the last wait cmd.
        /* write_barrier ensures that all writes initiated by the dispatcher are
                                        flushed before the event is recorded */
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | (clear_count ? CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM : 0) |
                ((i == num_worker_counters - 1) ? CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER : 0),
            0,
            MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
            expected_num_workers_completed[offset_index]);
    }

    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds(num_command_queues);
    std::vector<std::pair<const void*, uint32_t>> event_payloads(num_command_queues);

    for (auto cq_id = 0; cq_id < num_command_queues; cq_id++) {
        tt_cxy_pair dispatch_location = MetalContext::instance().get_dispatch_query_manager().get_dispatch_core(cq_id);
        CoreCoord dispatch_virtual_core = device->virtual_core_from_logical_core(dispatch_location, dispatch_core_type);
        unicast_sub_cmds[cq_id] = CQDispatchWritePackedUnicastSubCmd{
            .noc_xy_addr = device->get_noc_unicast_encoding(k_dispatch_downstream_noc, dispatch_virtual_core)};
        event_payloads[cq_id] = {event_payload.data(), event_payload.size() * sizeof(uint32_t)};
    }

    uint32_t completion_q0_last_event_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
    uint32_t address = cq_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_EVENT,
        num_command_queues,
        address,
        DispatchSettings::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads,
        packed_write_max_unicast_sub_cmds);

    if (notify_host) {
        bool flush_prefetch = true;
        uint16_t pad1 = (device_id << 8) | cq_id;
        command_sequence.add_dispatch_write_host<true>(
            flush_prefetch, DispatchSettings::EVENT_PADDED_SIZE, true, pad1, event_payload.data());
    }

    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);

    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void issue_wait_for_event_commands(
    uint8_t cq_id, uint8_t event_cq_id, SystemMemoryManager& sysmem_manager, uint32_t event_id) {
    tt::tt_metal::DeviceCommandCalculator calculator;
    calculator.add_dispatch_wait();
    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    uint32_t completion_q0_last_event_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);

    uint32_t last_completed_event_address =
        event_cq_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;

    command_sequence.add_dispatch_wait(
        CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_MEMORY, last_completed_event_address, 0, event_id);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);

    sysmem_manager.fetch_queue_reserve_back(cq_id);

    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void read_events_from_completion_queue(
    ReadEventDescriptor& event_descriptor,
    ChipId mmio_device_id,
    ChipId device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager) {
    // For mock devices, the sysmem_manager is a stubbed singleton
    // Mock cluster.read_sysmem returns zeros, so validate that and handle gracefully
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        sysmem_manager.set_last_completed_event(cq_id, event_descriptor.get_global_event_id());
        return;
    }

    uint32_t read_ptr = sysmem_manager.get_completion_queue_read_ptr(cq_id);
    thread_local static std::vector<uint32_t> dispatch_cmd_and_event(
        (sizeof(CQDispatchCmd) + DispatchSettings::EVENT_PADDED_SIZE) / sizeof(uint32_t));
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        dispatch_cmd_and_event.data(),
        sizeof(CQDispatchCmd) + DispatchSettings::EVENT_PADDED_SIZE,
        read_ptr,
        mmio_device_id,
        channel);

    CQDispatchCmd* dispatch_cmd = reinterpret_cast<CQDispatchCmd*>(dispatch_cmd_and_event.data());
    uint32_t expected_padding_value = HugepageDeviceCommand::random_padding_value();
    uint16_t expected_pad1 = (device_id << 8) | cq_id;

    TT_FATAL(
        dispatch_cmd->base.cmd_id == CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST && dispatch_cmd->write_linear_host.is_event &&
            dispatch_cmd->write_linear_host.length == sizeof(CQDispatchCmd) + DispatchSettings::EVENT_PADDED_SIZE &&
            dispatch_cmd->write_linear_host.pad1 == expected_pad1 &&
            dispatch_cmd->write_linear_host.pad2 == expected_padding_value,
        "Unexpected values for event in completion queue, got cmd id {}, is event {}, length {}, pad1 {} (expected "
        "{}), pad2 {} (expected {})",
        dispatch_cmd->base.cmd_id,
        dispatch_cmd->write_linear_host.is_event,
        dispatch_cmd->write_linear_host.length,
        dispatch_cmd->write_linear_host.pad1,
        expected_pad1,
        dispatch_cmd->write_linear_host.pad2,
        expected_padding_value);
    uint32_t event_completed = dispatch_cmd_and_event[sizeof(CQDispatchCmd) / sizeof(uint32_t)];

    TT_FATAL(
        event_completed == event_descriptor.event_id,
        "Event Order Issue: expected to read back completion signal for event {} but got {}!",
        event_descriptor.event_id,
        event_completed);
    sysmem_manager.completion_queue_pop_front(1, cq_id);
    sysmem_manager.set_last_completed_event(cq_id, event_descriptor.get_global_event_id());
    log_trace(
        LogAlways,
        "Completion queue popped event {} (global: {})",
        event_completed,
        event_descriptor.get_global_event_id());
}

}  // namespace tt::tt_metal::event_dispatch
