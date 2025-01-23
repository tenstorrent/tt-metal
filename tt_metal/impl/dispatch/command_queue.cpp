// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <command_queue.hpp>

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
#include <hardware_command_queue.hpp>
#include "program_command_sequence.hpp"
#include "tt_metal/command_queue.hpp"
#include <assert.hpp>
#include <logger.hpp>
#include <tt_metal.hpp>
#include <host_api.hpp>
#include <circular_buffer_constants.h>
#include <circular_buffer.hpp>
#include <dprint_server.hpp>
#include "tt_metal/impl/debug/watcher_server.hpp"
#include <cq_commands.hpp>
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include <dispatch_core_manager.hpp>
#include <event.hpp>
#include <kernel.hpp>
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "umd/device/tt_xy_pair.h"

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

    if (buffer_obj.is_valid_partial_region(region)) {
        TT_FATAL(
            !is_sharded(buffer_obj.buffer_layout()),
            "Reading from and writing to partial buffer regions is only supported for non-sharded buffers.");
    }
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
    this->dispatch_message_addr = dispatch_constants::get(
        this->dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE) +
        dispatch_constants::get(this->dispatch_core_type).get_dispatch_message_offset(this->sub_device_id.to_index());
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
    program_dispatch::write_program_command_sequence(cached_program_command_sequence, this->manager, this->command_queue_id, this->dispatch_core_type, dispatch_metadata.stall_first, dispatch_metadata.stall_before_program);
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
    std::vector<uint32_t> event_payload(dispatch_constants::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = this->event_id;

    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    uint8_t num_hw_cqs =
        this->device->num_hw_cqs();  // Device initialize asserts that there can only be a maximum of 2 HW CQs
    uint32_t packed_event_payload_sizeB =
        align(sizeof(CQDispatchCmd) + num_hw_cqs * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment) +
        (align(dispatch_constants::EVENT_PADDED_SIZE, l1_alignment) * num_hw_cqs);
    uint32_t packed_write_sizeB = align(sizeof(CQPrefetchCmd) + packed_event_payload_sizeB, pcie_alignment);
    uint32_t num_worker_counters = this->sub_device_ids.size();

    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST) * num_worker_counters +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        packed_write_sizeB +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PACKED + unicast subcmds + event
                              // payload
        align(
            sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE,
            pcie_alignment);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST + event ID

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    uint32_t dispatch_message_base_addr = dispatch_constants::get(
        dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);

    uint32_t last_index = num_worker_counters - 1;
    // We only need the write barrier for the last wait cmd
    for (uint32_t i = 0; i < last_index; ++i) {
        auto offset_index = this->sub_device_ids[i].to_index();
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, this->expected_num_workers_completed[offset_index], this->clear_count);

    }
    auto offset_index = this->sub_device_ids[last_index].to_index();
    uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
    command_sequence.add_dispatch_wait(
            this->write_barrier, dispatch_message_addr, this->expected_num_workers_completed[offset_index], this->clear_count);

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

    uint32_t completion_q0_last_event_addr = dispatch_constants::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr = dispatch_constants::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
    uint32_t address = this->command_queue_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;
    const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_hw_cqs,
        address,
        dispatch_constants::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads,
        packed_write_max_unicast_sub_cmds);

    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_host<true>(
        flush_prefetch, dispatch_constants::EVENT_PADDED_SIZE, true, event_payload.data());

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
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    uint32_t completion_q0_last_event_addr = dispatch_constants::get(this->dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr = dispatch_constants::get(this->dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);

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
    std::shared_ptr<detail::TraceDescriptor>& descriptor,
    Buffer& buffer,
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> & expected_num_workers_completed,
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
    uint32_t go_signals_cmd_size = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * descriptor->descriptors.size();

    uint32_t cmd_sequence_sizeB =
        this->device->dispatch_s_enabled() * hal.get_alignment(HalMemType::HOST) + // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +  // go signal cmd
        (hal.get_alignment(HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
        // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s. dispatch_s is responsible for resetting worker count
        // and giving dispatch_d the latest worker state. This is encapsulated in the dispatch_s wait command (only to be sent when dispatch is distributed
        // on 2 cores)
        (this->device->distributed_dispatcher()) * hal.get_alignment(HalMemType::HOST)) * num_sub_devices +
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_EXEC_BUF

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    if (this->device->dispatch_s_enabled()) {
        uint16_t index_bitmask = 0;
        for (const auto &id : descriptor->sub_device_ids) {
            index_bitmask |= 1 << id.to_index();
        }
        command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
        dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
    }
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    uint32_t dispatch_message_base_addr = dispatch_constants::get(
        dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    go_msg_t reset_launch_message_read_ptr_go_signal;
    reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
    reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)this->dispatch_core.x;
    reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)this->dispatch_core.y;
    for (const auto& [id, desc] : descriptor->descriptors) {
        const auto& noc_data_start_idx = device->noc_data_start_index(id, desc.num_traced_programs_needing_go_signal_multicast, desc.num_traced_programs_needing_go_signal_unicast);
        const auto& num_noc_mcast_txns = desc.num_traced_programs_needing_go_signal_multicast ? device->num_noc_mcast_txns(id) : 0;
        const auto& num_noc_unicast_txns = desc.num_traced_programs_needing_go_signal_unicast ? device->num_noc_unicast_txns(id) : 0;
        reset_launch_message_read_ptr_go_signal.dispatch_message_offset = (uint8_t)dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(id.to_index());
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(id.to_index());
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
            this->expected_num_workers_completed[index] += device->num_worker_cores(HalProgrammableCoreType::TENSIX, id);
        }
        if (desc.num_traced_programs_needing_go_signal_unicast) {
            this->expected_num_workers_completed[index] += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, id);
        }
    }
    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed this step, before sending kernel config data to workers
    // or notifying dispatch_s that its safe to send the go_signal.
    // Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (const auto &id : descriptor->sub_device_ids) {
        auto index = id.to_index();
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(index);
        if (this->device->distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, this->expected_num_workers_completed[index], this->clear_count, false, true, 1);
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
    if (this->device->dispatch_s_enabled()) {
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

void EnqueueAddBufferToProgramImpl(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    Program& program) {
    std::visit(
        [&program](auto&& b) {
            using buffer_type = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
                program.add_buffer(b);
            }
        },
        buffer);
}

void EnqueueAddBufferToProgram(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    Program& program,
    bool blocking) {
    EnqueueAddBufferToProgramImpl(std::move(buffer), program);
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit(
            [&resolved_runtime_args](auto&& a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, Buffer*>) {
                    resolved_runtime_args.push_back(a->address());
                } else {
                    resolved_runtime_args.push_back(a);
                }
            },
            arg);
    }
    runtime_args_md.kernel->set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueSetRuntimeArgs(
    CommandQueue& cq,
    const std::shared_ptr<Kernel>& kernel,
    const CoreCoord& core_coord,
    std::shared_ptr<RuntimeArgs> runtime_args_ptr,
    bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata{
        .core_coord = core_coord,
        .runtime_args_ptr = std::move(runtime_args_ptr),
        .kernel = kernel,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::SET_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer) {
    *(static_cast<uint32_t*>(dst_buf_addr)) = buffer->address();
}

void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking) {
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::GET_BUF_ADDR, .blocking = blocking, .shadow_buffer = buffer, .dst = dst_buf_addr});
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

    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .dst = dst,
        .region = region});
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
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    const BufferRegion& region,
    bool blocking) {
    detail::DispatchStateCheck(true);
    detail::ValidateBufferRegion(buffer, region);

    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .src = std::move(src),
        .region = region});
}

void EnqueueProgram(
    CommandQueue& cq, Program& program, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(
        CommandInterface{.type = EnqueueCommandType::ENQUEUE_PROGRAM, .blocking = blocking, .program = &program});
}

void EnqueueRecordEvent(CommandQueue& cq, const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_RECORD_EVENT,
        .blocking = false,
        .event = event,
        .sub_device_ids = sub_device_ids
    });
}

void EnqueueWaitForEvent(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT,
        .blocking = false,
        .event = event,
    });
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
        if (tt::llrt::RunTimeOptions::get_instance().get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
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
    cq.run_command(
        CommandInterface{.type = EnqueueCommandType::FINISH, .blocking = true, .sub_device_ids = sub_device_ids});
    TT_ASSERT(
        !(cq.hw_command_queue().is_dprint_server_hung()),
        "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(
        !(cq.hw_command_queue().is_noc_hung()),
        "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
        tt::watcher_get_log_file_name());
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(cq.device()->get_trace(trace_id) != nullptr, "Trace instance {} must exist on device", trace_id);
    cq.run_command(
        CommandInterface{.type = EnqueueCommandType::ENQUEUE_TRACE, .blocking = blocking, .trace_id = trace_id});
}

}  // namespace v0

void EnqueueReadBufferImpl(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    const BufferRegion& region,
    bool blocking) {
    std::visit(
        [&](auto&& b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (
                std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.hw_command_queue().enqueue_read_buffer(b, dst, region, blocking);
            }
        },
        buffer);
}

void EnqueueWriteBufferImpl(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    HostDataType src,
    const BufferRegion& region,
    bool blocking) {
    cq.hw_command_queue().enqueue_write_buffer(std::move(buffer), std::move(src), region, blocking);
}

void EnqueueProgramImpl(
    CommandQueue& cq, Program& program, bool blocking) {
    ZoneScoped;

    IDevice* device = cq.device();
    detail::CompileProgram(device, program);
    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);
    cq.hw_command_queue().enqueue_program(program, blocking);
    // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem
    // leaks on device.
    program.release_buffers();

}

void EnqueueRecordEventImpl(CommandQueue& cq, const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    cq.hw_command_queue().enqueue_record_event(event, false, sub_device_ids);
}

void EnqueueWaitForEventImpl(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    event->wait_until_ready();  // Block until event populated. Worker thread.
    log_trace(
        tt::LogMetal,
        "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(),
        event->cq_id,
        event->event_id,
        cq.device()->id(),
        cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}

void FinishImpl(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids) { cq.hw_command_queue().finish(sub_device_ids); }

void EnqueueTraceImpl(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    cq.hw_command_queue().enqueue_trace(trace_id, blocking);
}

CommandQueue::CommandQueue(IDevice* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device), cq_id(id), mode(mode), worker_state(CommandQueueState::IDLE) {
    if (this->async_mode()) {
        num_async_cqs++;
        // The main program thread launches the Command Queue
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
    }
}

CommandQueue::CommandQueue(Trace& trace) :
    device_ptr(nullptr),
    parent_thread_id(0),
    cq_id(-1),
    mode(CommandQueueMode::TRACE),
    worker_state(CommandQueueState::IDLE) {}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (not this->trace_mode()) {
        TT_FATAL(this->worker_queue.empty(), "{} worker queue must be empty on destruction", this->name());
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() { return this->device()->hw_command_queue(this->cq_id); }

void CommandQueue::dump() {
    int cid = 0;
    log_info(LogMetalTrace, "Dumping {}, mode={}", this->name(), this->get_mode());
    for (const auto& cmd : this->worker_queue) {
        log_info(LogMetalTrace, "[{}]: {}", cid, cmd.type);
        cid++;
    }
}

std::string CommandQueue::name() {
    if (this->mode == CommandQueueMode::TRACE) {
        return "TraceQueue";
    }
    return "CQ" + std::to_string(this->cq_id);
}

void CommandQueue::wait_until_empty() {
    log_trace(LogDispatch, "{} WFI start", this->name());
    if (this->async_mode()) {
        // Insert a flush token to push all prior commands to completion
        // Necessary to avoid implementing a peek and pop on the lock-free queue
        this->worker_queue.push(CommandInterface{.type = EnqueueCommandType::FLUSH});
    }
    while (true) {
        if (this->worker_queue.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    log_trace(LogDispatch, "{} WFI complete", this->name());
}

void CommandQueue::set_mode(const CommandQueueMode& mode) {
    TT_ASSERT(
        not this->trace_mode(),
        "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
    if (this->mode == mode) {
        // Do nothing if requested mode matches current CQ mode.
        return;
    }
    this->mode = mode;
    if (this->async_mode()) {
        num_async_cqs++;
        num_passthrough_cqs--;
        // Record parent thread-id and start worker.
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
        num_async_cqs--;
        // Wait for all cmds sent in async mode to complete and stop worker.
        this->wait_until_empty();
        this->stop_worker();
    }
}

void CommandQueue::start_worker() {
    if (this->worker_state == CommandQueueState::RUNNING) {
        return;  // worker already running, exit
    }
    this->worker_state = CommandQueueState::RUNNING;
    this->worker_thread = std::make_unique<std::thread>(std::thread(&CommandQueue::run_worker, this));
    tt::log_debug(tt::LogDispatch, "{} started worker thread", this->name());
}

void CommandQueue::stop_worker() {
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "{} stopped worker thread", this->name());
}

void CommandQueue::run_worker() {
    // forever loop checking for commands in the worker queue
    // Track the worker thread id, for cases where a command calls a sub command.
    // This is to detect cases where commands may be nested.
    worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if (this->worker_queue.empty()) {
            if (this->worker_state == CommandQueueState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::shared_ptr<CommandInterface> command(this->worker_queue.pop());
            run_command_impl(*command);
        }
    }
}

void CommandQueue::run_command(CommandInterface&& command) {
    log_trace(LogDispatch, "{} received {} in {} mode", this->name(), command.type, this->mode);
    if (this->async_mode()) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == parent_thread_id) {
            // In async mode when parent pushes cmd, feed worker through queue.
            bool blocking = command.blocking.has_value() and *command.blocking;
            this->worker_queue.push(std::move(command));
            if (blocking) {
                TT_ASSERT(not this->trace_mode(), "Blocking commands cannot be traced!");
                this->wait_until_empty();
            }
        } else {
            // Handle case where worker pushes command to itself (passthrough)
            TT_ASSERT(
                std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id,
                "Only main thread or worker thread can run commands through the SW command queue");
            run_command_impl(command);
        }
    } else if (this->trace_mode()) {
        // In trace mode push to the trace queue
        this->worker_queue.push(std::move(command));
    } else if (this->passthrough_mode()) {
        this->run_command_impl(command);
    } else {
        TT_THROW("Unsupported CommandQueue mode!");
    }
}

void CommandQueue::run_command_impl(const CommandInterface& command) {
    log_trace(LogDispatch, "{} running {}", this->name(), command.type);
    switch (command.type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            TT_ASSERT(command.region.has_value(), "Must specify region value!");
            EnqueueReadBufferImpl(
                *this,
                command.buffer.value(),
                command.dst.value(),
                command.region.value(),
                command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER:
            TT_ASSERT(command.src.has_value(), "Must provide a src!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            TT_ASSERT(command.region.has_value(), "Must specify region value!");
            EnqueueWriteBufferImpl(
                *this,
                command.buffer.value(),
                command.src.value(),
                command.region.value(),
                command.blocking.value());
            break;
        case EnqueueCommandType::GET_BUF_ADDR:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst address!");
            TT_ASSERT(command.shadow_buffer.has_value(), "Must provide a shadow buffer!");
            EnqueueGetBufferAddrImpl(command.dst.value(), command.shadow_buffer.value());
            break;
        case EnqueueCommandType::SET_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueSetRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.program != nullptr, "Must provide a program!");
            EnqueueAddBufferToProgramImpl(command.buffer.value(), *command.program);
            break;
        case EnqueueCommandType::ENQUEUE_PROGRAM:
            TT_ASSERT(command.program != nullptr, "Must provide a program!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueProgramImpl(*this, *command.program, command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_TRACE:
            EnqueueTraceImpl(*this, command.trace_id.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueRecordEventImpl(*this, command.event.value(), command.sub_device_ids);
            break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueWaitForEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::FINISH: FinishImpl(*this, command.sub_device_ids); break;
        case EnqueueCommandType::FLUSH:
            // Used by CQ to push prior commands
            break;
        default: TT_THROW("Invalid command type");
    }
    log_trace(LogDispatch, "{} running {} complete", this->name(), command.type);
}

v1::CommandQueueHandle v1::GetCommandQueue(IDevice* device, std::uint8_t cq_id) {
    return v1::CommandQueueHandle{device, cq_id};
}

v1::CommandQueueHandle v1::GetDefaultCommandQueue(IDevice* device) { return GetCommandQueue(device, 0); }

void v1::EnqueueReadBuffer(CommandQueueHandle cq, const BufferHandle& buffer, std::byte *dst, bool blocking) {
    v0::EnqueueReadBuffer(GetDevice(cq)->command_queue(GetId(cq)), *buffer, dst, blocking);
}

void v1::EnqueueWriteBuffer(CommandQueueHandle cq, const BufferHandle& buffer, const std::byte *src, bool blocking) {
    v0::EnqueueWriteBuffer(GetDevice(cq)->command_queue(GetId(cq)), *buffer, src, blocking);
}

void v1::EnqueueProgram(CommandQueueHandle cq, ProgramHandle &program, bool blocking) {
    v0::EnqueueProgram(GetDevice(cq)->command_queue(GetId(cq)), program, blocking);
}

void v1::Finish(CommandQueueHandle cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    v0::Finish(GetDevice(cq)->command_queue(GetId(cq)));
}

IDevice* v1::GetDevice(CommandQueueHandle cq) {
    return cq.device;
}

std::uint8_t v1::GetId(CommandQueueHandle cq) {
    return cq.id;
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type) {
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

std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type) {
    switch (type) {
        case CommandQueue::CommandQueueMode::PASSTHROUGH: os << "PASSTHROUGH"; break;
        case CommandQueue::CommandQueueMode::ASYNC: os << "ASYNC"; break;
        case CommandQueue::CommandQueueMode::TRACE: os << "TRACE"; break;
        default: TT_THROW("Invalid CommandQueueMode type!");
    }
    return os;
}
