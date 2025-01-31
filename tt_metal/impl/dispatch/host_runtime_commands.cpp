// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_runtime_commands.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include <buffer.hpp>
#include <math.hpp>
#include <dev_msgs.h>
#include <hal.hpp>
#include "program_command_sequence.hpp"
#include "tt_metal/command_queue.hpp"
#include <assert.hpp>
#include <logger.hpp>
#include <tt_metal.hpp>
#include <host_api.hpp>
#include <circular_buffer_constants.h>
#include <circular_buffer.hpp>
#include "dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include <cq_commands.hpp>
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include <dispatch_core_manager.hpp>
#include <event.hpp>
#include <kernel.hpp>
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "umd/device/tt_xy_pair.h"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/dispatch_settings.hpp>

#include <hal.hpp>

using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace detail {

bool DispatchStateCheck(bool isFastDispatch) {
    static bool fd = isFastDispatch;
    TT_FATAL(fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!");
    return fd;
}

Buffer& GetBufferObject(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer) {
    return std::visit(
        [&](auto&& b) -> Buffer& {
            using type_buf = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<type_buf, std::shared_ptr<Buffer>>) {
                return *b;
            } else {
                return b.get();
            }
        },
        buffer);
}

void ValidateBufferRegion(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer, const BufferRegion& region) {
    Buffer& buffer_obj = GetBufferObject(buffer);

    TT_FATAL(
        buffer_obj.is_valid_region(region),
        "Buffer region with offset {} and size {} is invalid.",
        region.offset,
        region.size);
}
}  // namespace detail

enum DispatchWriteOffsets {
    DISPATCH_WRITE_OFFSET_ZERO = 0,
    DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE = 1,
    DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE = 2,
};

inline uint32_t get_packed_write_max_unicast_sub_cmds(IDevice* device) {
    return device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
}

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    IDevice* device,
    NOC noc_index,
    Program& program,
    CoreCoord& dispatch_core,
    SystemMemoryManager& manager,
    WorkerConfigBufferMgr& config_buffer_mgr,
    uint32_t expected_num_workers_completed,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    SubDeviceId sub_device_id) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    manager(manager),
    config_buffer_mgr(config_buffer_mgr),
    expected_num_workers_completed(expected_num_workers_completed),
    program(program),
    dispatch_core(dispatch_core),
    multicast_cores_launch_message_wptr(multicast_cores_launch_message_wptr),
    unicast_cores_launch_message_wptr(unicast_cores_launch_message_wptr),
    sub_device_id(sub_device_id) {
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    this->packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
    this->dispatch_message_addr =
        DispatchMemMap::get(this->dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE) +
        DispatchMemMap::get(this->dispatch_core_type).get_dispatch_message_offset(this->sub_device_id.to_index());
}

void EnqueueProgramCommand::process() {
    // Dispatch metadata contains runtime information based on
    // the kernel config ring buffer state
    program_dispatch::ProgramDispatchMetadata dispatch_metadata;

    // Compute the total number of workers this program uses
    uint32_t num_workers = 0;
    if (program.runs_on_noc_multicast_only_cores()) {
        num_workers += device->num_worker_cores(HalProgrammableCoreType::TENSIX, this->sub_device_id);
    }
    if (program.runs_on_noc_unicast_only_cores()) {
        num_workers += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, this->sub_device_id);
    }
    // Reserve space for this program in the kernel config ring buffer
    program_dispatch::reserve_space_in_kernel_config_buffer(
        this->config_buffer_mgr,
        program.get_program_config_sizes(),
        program.get_program_binary_status(device->id()),
        num_workers,
        this->expected_num_workers_completed,
        dispatch_metadata);

    RecordProgramRun(program);

    // Access the program dispatch-command cache
    auto& cached_program_command_sequence = program.get_cached_program_command_sequences().begin()->second;
    // Update the generated dispatch commands based on the state of the CQ and the ring buffer
    program_dispatch::update_program_dispatch_commands(
        program,
        cached_program_command_sequence,
        this->multicast_cores_launch_message_wptr,
        this->unicast_cores_launch_message_wptr,
        this->expected_num_workers_completed,
        this->dispatch_core,
        this->dispatch_core_type,
        this->sub_device_id,
        dispatch_metadata,
        program.get_program_binary_status(device->id()));
    // Issue dispatch commands for this program
    program_dispatch::write_program_command_sequence(
        cached_program_command_sequence,
        this->manager,
        this->command_queue_id,
        this->dispatch_core_type,
        dispatch_metadata.stall_first,
        dispatch_metadata.stall_before_program);
    // Kernel Binaries are committed to DRAM, the first time the program runs on device. Reflect this on host.
    program.set_program_binary_status(device->id(), ProgramBinaryStatus::Committed);
}

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id,
    IDevice* device,
    NOC noc_index,
    SystemMemoryManager& manager,
    uint32_t event_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    bool clear_count,
    bool write_barrier) :
    command_queue_id(command_queue_id),
    device(device),
    noc_index(noc_index),
    manager(manager),
    event_id(event_id),
    expected_num_workers_completed(expected_num_workers_completed),
    sub_device_ids(sub_device_ids),
    clear_count(clear_count),
    write_barrier(write_barrier) {}

void EnqueueRecordEventCommand::process() {
    std::vector<uint32_t> event_payload(DispatchSettings::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = this->event_id;

    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    uint8_t num_hw_cqs =
        this->device->num_hw_cqs();  // Device initialize asserts that there can only be a maximum of 2 HW CQs
    uint32_t packed_event_payload_sizeB =
        align(sizeof(CQDispatchCmd) + num_hw_cqs * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment) +
        (align(DispatchSettings::EVENT_PADDED_SIZE, l1_alignment) * num_hw_cqs);
    uint32_t packed_write_sizeB = align(sizeof(CQPrefetchCmd) + packed_event_payload_sizeB, pcie_alignment);
    uint32_t num_worker_counters = this->sub_device_ids.size();

    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST) *
            num_worker_counters +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        packed_write_sizeB +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PACKED + unicast subcmds + event
                              // payload
        align(
            sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + DispatchSettings::EVENT_PADDED_SIZE,
            pcie_alignment);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST + event ID

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    uint32_t dispatch_message_base_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);

    uint32_t last_index = num_worker_counters - 1;
    // We only need the write barrier for the last wait cmd
    for (uint32_t i = 0; i < last_index; ++i) {
        auto offset_index = this->sub_device_ids[i].to_index();
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr +
            DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, this->expected_num_workers_completed[offset_index], this->clear_count);
    }
    auto offset_index = this->sub_device_ids[last_index].to_index();
    uint32_t dispatch_message_addr =
        dispatch_message_base_addr +
        DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
    command_sequence.add_dispatch_wait(
        this->write_barrier,
        dispatch_message_addr,
        this->expected_num_workers_completed[offset_index],
        this->clear_count);

    CoreType core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds(num_hw_cqs);
    std::vector<std::pair<const void*, uint32_t>> event_payloads(num_hw_cqs);

    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair dispatch_location;
        if (device->is_mmio_capable()) {
            dispatch_location = dispatch_core_manager::instance().dispatcher_core(this->device->id(), channel, cq_id);
        } else {
            dispatch_location = dispatch_core_manager::instance().dispatcher_d_core(this->device->id(), channel, cq_id);
        }

        CoreCoord dispatch_virtual_core = this->device->virtual_core_from_logical_core(dispatch_location, core_type);
        unicast_sub_cmds[cq_id] = CQDispatchWritePackedUnicastSubCmd{
            .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, dispatch_virtual_core)};
        event_payloads[cq_id] = {event_payload.data(), event_payload.size() * sizeof(uint32_t)};
    }

    uint32_t completion_q0_last_event_addr = DispatchMemMap::get(core_type).get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr = DispatchMemMap::get(core_type).get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
    uint32_t address = this->command_queue_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;
    const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_hw_cqs,
        address,
        DispatchSettings::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads,
        packed_write_max_unicast_sub_cmds);

    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_host<true>(
        flush_prefetch, DispatchSettings::EVENT_PADDED_SIZE, true, event_payload.data());

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
    uint32_t command_queue_id,
    IDevice* device,
    SystemMemoryManager& manager,
    const Event& sync_event,
    bool clear_count) :
    command_queue_id(command_queue_id),
    device(device),
    manager(manager),
    sync_event(sync_event),
    clear_count(clear_count) {
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
    // TT_ASSERT(command_queue_id != sync_event.cq_id || event != sync_event.event_id,
    //     "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. Event ID: {} CQ ID: {}",
    //     event, command_queue_id);
}

void EnqueueWaitForEventCommand::process() {
    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    uint32_t completion_q0_last_event_addr =
        DispatchMemMap::get(this->dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr =
        DispatchMemMap::get(this->dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);

    uint32_t last_completed_event_address =
        sync_event.cq_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;

    command_sequence.add_dispatch_wait(false, last_completed_event_address, sync_event.event_id, this->clear_count);

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

EnqueueTraceCommand::EnqueueTraceCommand(
    uint32_t command_queue_id,
    IDevice* device,
    SystemMemoryManager& manager,
    std::shared_ptr<TraceDescriptor>& descriptor,
    Buffer& buffer,
    std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
    NOC noc_index,
    CoreCoord dispatch_core) :
    command_queue_id(command_queue_id),
    buffer(buffer),
    device(device),
    manager(manager),
    descriptor(descriptor),
    expected_num_workers_completed(expected_num_workers_completed),
    clear_count(true),
    noc_index(noc_index),
    dispatch_core(dispatch_core) {}

void EnqueueTraceCommand::process() {
    uint32_t num_sub_devices = descriptor->descriptors.size();
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t go_signals_cmd_size =
        align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * descriptor->descriptors.size();

    uint32_t cmd_sequence_sizeB =
        DispatchQueryManager::instance().dispatch_s_enabled() *
            hal.get_alignment(
                HalMemType::HOST) +  // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +        // go signal cmd
        (hal.get_alignment(
             HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
                                  // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s.
                                  // dispatch_s is responsible for resetting worker count and giving dispatch_d the
                                  // latest worker state. This is encapsulated in the dispatch_s wait command (only to
                                  // be sent when dispatch is distributed on 2 cores)
         (DispatchQueryManager::instance().distributed_dispatcher()) * hal.get_alignment(HalMemType::HOST)) *
            num_sub_devices +
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_EXEC_BUF

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    if (DispatchQueryManager::instance().dispatch_s_enabled()) {
        uint16_t index_bitmask = 0;
        for (const auto& id : descriptor->sub_device_ids) {
            index_bitmask |= 1 << id.to_index();
        }
        command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
        dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
    }
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    uint32_t dispatch_message_base_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    go_msg_t reset_launch_message_read_ptr_go_signal;
    reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
    reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)this->dispatch_core.x;
    reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)this->dispatch_core.y;
    for (const auto& [id, desc] : descriptor->descriptors) {
        const auto& noc_data_start_idx = device->noc_data_start_index(
            id,
            desc.num_traced_programs_needing_go_signal_multicast,
            desc.num_traced_programs_needing_go_signal_unicast);
        const auto& num_noc_mcast_txns =
            desc.num_traced_programs_needing_go_signal_multicast ? device->num_noc_mcast_txns(id) : 0;
        const auto& num_noc_unicast_txns =
            desc.num_traced_programs_needing_go_signal_unicast ? device->num_noc_unicast_txns(id) : 0;
        reset_launch_message_read_ptr_go_signal.dispatch_message_offset =
            (uint8_t)DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(id.to_index());
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr +
            DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(id.to_index());
        auto index = id.to_index();
        // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
        command_sequence.add_dispatch_go_signal_mcast(
            this->expected_num_workers_completed[index],
            *reinterpret_cast<uint32_t*>(&reset_launch_message_read_ptr_go_signal),
            dispatch_message_addr,
            num_noc_mcast_txns,
            num_noc_unicast_txns,
            noc_data_start_idx,
            dispatcher_for_go_signal);
        if (desc.num_traced_programs_needing_go_signal_multicast) {
            this->expected_num_workers_completed[index] +=
                device->num_worker_cores(HalProgrammableCoreType::TENSIX, id);
        }
        if (desc.num_traced_programs_needing_go_signal_unicast) {
            this->expected_num_workers_completed[index] +=
                device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, id);
        }
    }
    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed
    // this step, before sending kernel config data to workers or notifying dispatch_s that its safe to send the
    // go_signal. Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (const auto& id : descriptor->sub_device_ids) {
        auto index = id.to_index();
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr + DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(index);
        if (DispatchQueryManager::instance().distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                false,
                dispatch_message_addr,
                this->expected_num_workers_completed[index],
                this->clear_count,
                false,
                true,
                1);
        }
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, this->expected_num_workers_completed[index], this->clear_count);
        if (this->clear_count) {
            this->expected_num_workers_completed[index] = 0;
        }
    }

    uint32_t page_size = buffer.page_size();
    uint32_t page_size_log2 = __builtin_ctz(page_size);
    TT_ASSERT((page_size & (page_size - 1)) == 0, "Page size must be a power of 2");

    command_sequence.add_prefetch_exec_buf(buffer.address(), page_size_log2, buffer.num_pages());

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    const bool stall_prefetcher = true;
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id, stall_prefetcher);
}

EnqueueTerminateCommand::EnqueueTerminateCommand(
    uint32_t command_queue_id, IDevice* device, SystemMemoryManager& manager) :
    command_queue_id(command_queue_id), device(device), manager(manager) {}

void EnqueueTerminateCommand::process() {
    // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
    // CQ_PREFETCH_CMD_TERMINATE
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);

    // dispatch and prefetch terminate commands each needs to be a separate fetch queue entry
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand dispatch_d_command_sequence(cmd_region, cmd_sequence_sizeB);
    dispatch_d_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_MASTER);
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
    if (DispatchQueryManager::instance().dispatch_s_enabled()) {
        // Terminate dispatch_s if enabled
        cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
        HugepageDeviceCommand dispatch_s_command_sequence(cmd_region, cmd_sequence_sizeB);
        dispatch_s_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_SLAVE);
        this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
    }
    cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand prefetch_command_sequence(cmd_region, cmd_sequence_sizeB);
    prefetch_command_sequence.add_prefetch_terminate();
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

inline namespace v0 {

void EnqueueWriteBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    std::vector<uint32_t>& src,
    bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, std::move(buffer), src.data(), blocking);
}

void EnqueueReadBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* dst,
    bool blocking) {
    Buffer& buffer_obj = detail::GetBufferObject(buffer);
    BufferRegion region(0, buffer_obj.size());
    EnqueueReadSubBuffer(cq, buffer, dst, region, blocking);
}

void EnqueueReadSubBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    const BufferRegion& region,
    bool blocking) {
    detail::DispatchStateCheck(true);
    detail::ValidateBufferRegion(buffer, region);

    std::visit(
        [&](auto&& b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (
                std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.enqueue_read_buffer(b, dst, region, blocking);
            }
        },
        buffer);
}

void EnqueueWriteBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    HostDataType src,
    bool blocking) {
    Buffer& buffer_obj = detail::GetBufferObject(buffer);
    BufferRegion region(0, buffer_obj.size());
    EnqueueWriteSubBuffer(cq, buffer, std::move(src), region, blocking);
}

void EnqueueWriteSubBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    HostDataType src,
    const BufferRegion& region,
    bool blocking) {
    detail::DispatchStateCheck(true);
    detail::ValidateBufferRegion(buffer, region);

    cq.enqueue_write_buffer(std::move(buffer), std::move(src), region, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    ZoneScoped;
    detail::DispatchStateCheck(true);

    IDevice* device = cq.device();
    detail::CompileProgram(device, program);
    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);
    cq.enqueue_program(program, blocking);
    // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem
    // leaks on device.
    program.release_buffers();
}

void EnqueueRecordEvent(
    CommandQueue& cq, const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.enqueue_record_event(event, false, sub_device_ids);
}

void EnqueueWaitForEvent(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Worker thread.
    log_trace(
        tt::LogMetal,
        "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(),
        event->cq_id,
        event->event_id,
        cq.device()->id(),
        cq.id());
    cq.enqueue_wait_for_event(event);
}

void EventSynchronize(const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    log_trace(
        tt::LogMetal,
        "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})",
        event->device->id(),
        event->cq_id,
        event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::RunTimeOptions::get_instance().get_test_mode_enabled() &&
            tt::watcher_server_killed_due_to_error()) {
            TT_FATAL(
                false,
                "Command Queue could not complete EventSynchronize. See {} for details.",
                tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

bool EventQuery(const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(
        tt::LogMetal,
        "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed,
        event->device->id(),
        event->cq_id,
        event->event_id);
    return event_completed;
}

void Finish(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.finish(sub_device_ids);
    TT_ASSERT(
        !(cq.is_dprint_server_hung()), "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(
        !(cq.is_noc_hung()),
        "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
        tt::watcher_get_log_file_name());
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(cq.device()->get_trace(trace_id) != nullptr, "Trace instance {} must exist on device", trace_id);
    cq.enqueue_trace(trace_id, blocking);
}

}  // namespace v0

v1::CommandQueueHandle v1::GetCommandQueue(IDevice* device, std::uint8_t cq_id) {
    return v1::CommandQueueHandle{device, cq_id};
}

v1::CommandQueueHandle v1::GetDefaultCommandQueue(IDevice* device) { return GetCommandQueue(device, 0); }

void v1::EnqueueReadBuffer(CommandQueueHandle cq, const BufferHandle& buffer, std::byte* dst, bool blocking) {
    v0::EnqueueReadBuffer(GetDevice(cq)->command_queue(GetId(cq)), *buffer, dst, blocking);
}

void v1::EnqueueWriteBuffer(CommandQueueHandle cq, const BufferHandle& buffer, const std::byte* src, bool blocking) {
    v0::EnqueueWriteBuffer(GetDevice(cq)->command_queue(GetId(cq)), *buffer, src, blocking);
}

void v1::EnqueueProgram(CommandQueueHandle cq, ProgramHandle& program, bool blocking) {
    v0::EnqueueProgram(GetDevice(cq)->command_queue(GetId(cq)), program, blocking);
}

void v1::Finish(CommandQueueHandle cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    v0::Finish(GetDevice(cq)->command_queue(GetId(cq)));
}

IDevice* v1::GetDevice(CommandQueueHandle cq) { return cq.device; }

std::uint8_t v1::GetId(CommandQueueHandle cq) { return cq.id; }

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const EnqueueCommandType& type) {
    switch (type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: os << "ENQUEUE_READ_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: os << "ENQUEUE_WRITE_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_PROGRAM: os << "ENQUEUE_PROGRAM"; break;
        case EnqueueCommandType::ENQUEUE_TRACE: os << "ENQUEUE_TRACE"; break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT: os << "ENQUEUE_RECORD_EVENT"; break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: os << "ENQUEUE_WAIT_FOR_EVENT"; break;
        case EnqueueCommandType::FINISH: os << "FINISH"; break;
        case EnqueueCommandType::FLUSH: os << "FLUSH"; break;
        default: TT_THROW("Invalid command type!");
    }
    return os;
}
