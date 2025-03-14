// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/event/dispatch.hpp"
#include <tt-metalium/dispatch_settings.hpp>
#include "tt_metal/impl/dispatch/device_command.hpp"
#include <tt-metalium/program_impl.hpp>
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include <tt_align.hpp>

#include "tt_cluster.hpp"

namespace tt::tt_metal {

namespace event_dispatch {

namespace {
uint32_t get_packed_write_max_unicast_sub_cmds(IDevice* device) {
    return device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
}
}  // namespace

void issue_record_event_commands(
    IDevice* device,
    uint32_t event_id,
    uint8_t cq_id,
    uint32_t num_command_queues,
    SystemMemoryManager& manager,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    bool notify_host) {
    std::vector<uint32_t> event_payload(DispatchSettings::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = event_id;

    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    uint32_t packed_event_payload_sizeB =
        align(sizeof(CQDispatchCmd) + num_command_queues * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment) +
        (align(DispatchSettings::EVENT_PADDED_SIZE, l1_alignment) * num_command_queues);
    uint32_t packed_write_sizeB = align(sizeof(CQPrefetchCmd) + packed_event_payload_sizeB, pcie_alignment);
    uint32_t num_worker_counters = sub_device_ids.size();

    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST) *
            num_worker_counters +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        packed_write_sizeB +       // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PACKED +
                                   // unicast subcmds + event payload
        align(
            sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + DispatchSettings::EVENT_PADDED_SIZE,
            pcie_alignment) *
            notify_host;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST + event ID ===> Write
                          // event notification back to host, if requested by user

    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    auto dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();

    uint32_t last_index = num_worker_counters - 1;
    for (uint32_t i = 0; i < num_worker_counters; ++i) {
        auto offset_index = *sub_device_ids[i];
        // recording an event does not have any side-effects on the dispatch completion count
        // hence clear_count is set to false, i.e. the number of workers on the dispatcher is
        // not reset
        // We only need the write barrier for the last wait cmd.
        /* write_barrier ensures that all writes initiated by the dispatcher are
                                        flushed before the event is recorded */
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM |
                ((i == num_worker_counters - 1) ? CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER : 0),
            0,
            DispatchMemMap::get(dispatch_core_type).get_dispatch_stream_index(offset_index),
            expected_num_workers_completed[offset_index]);
    }

    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds(num_command_queues);
    std::vector<std::pair<const void*, uint32_t>> event_payloads(num_command_queues);

    for (auto cq_id = 0; cq_id < num_command_queues; cq_id++) {
        tt_cxy_pair dispatch_location = DispatchQueryManager::instance().get_dispatch_core(cq_id);
        CoreCoord dispatch_virtual_core = device->virtual_core_from_logical_core(dispatch_location, dispatch_core_type);
        unicast_sub_cmds[cq_id] = CQDispatchWritePackedUnicastSubCmd{
            .noc_xy_addr = device->get_noc_unicast_encoding(k_dispatch_downstream_noc, dispatch_virtual_core)};
        event_payloads[cq_id] = {event_payload.data(), event_payload.size() * sizeof(uint32_t)};
    }

    uint32_t completion_q0_last_event_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
    uint32_t address = cq_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;
    const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(device);
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_command_queues,
        address,
        DispatchSettings::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads,
        packed_write_max_unicast_sub_cmds);

    if (notify_host) {
        bool flush_prefetch = true;
        command_sequence.add_dispatch_write_host<true>(
            flush_prefetch, DispatchSettings::EVENT_PADDED_SIZE, true, event_payload.data());
    }

    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);

    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void issue_wait_for_event_commands(
    uint8_t cq_id, uint8_t event_cq_id, SystemMemoryManager& sysmem_manager, uint32_t event_id) {
    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

    auto dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    uint32_t completion_q0_last_event_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);

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
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager) {
    uint32_t read_ptr = sysmem_manager.get_completion_queue_read_ptr(cq_id);
    thread_local static std::vector<uint32_t> dispatch_cmd_and_event(
        (sizeof(CQDispatchCmd) + DispatchSettings::EVENT_PADDED_SIZE) / sizeof(uint32_t));
    tt::Cluster::instance().read_sysmem(
        dispatch_cmd_and_event.data(),
        sizeof(CQDispatchCmd) + DispatchSettings::EVENT_PADDED_SIZE,
        read_ptr,
        mmio_device_id,
        channel);
    uint32_t event_completed = dispatch_cmd_and_event[sizeof(CQDispatchCmd) / sizeof(uint32_t)];

    TT_ASSERT(
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

}  // namespace event_dispatch

}  // namespace tt::tt_metal
